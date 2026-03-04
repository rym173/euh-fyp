import json
import random
import time
import re
import ast
import io
import tokenize
from pathlib import Path
import numpy as np

from .eoh_diag_interface_EC import InterfaceECDiag
from ..selection import prob_rank
from ..management import pop_greedy
from ...llm.interface_LLM import InterfaceLLM


def _leading_spaces(line):
    return len(line) - len(line.lstrip(" "))


def _find_subsequence_start(lines, subseq, preferred_start):
    if not subseq:
        return max(0, min(preferred_start, len(lines)))

    def _match_at(start):
        if start < 0 or start + len(subseq) > len(lines):
            return False
        for i, val in enumerate(subseq):
            if lines[start + i] != val:
                return False
        return True

    if _match_at(preferred_start):
        return preferred_start

    left = max(0, preferred_start - 120)
    right = min(len(lines) - len(subseq), preferred_start + 120)
    for start in range(left, right + 1):
        if _match_at(start):
            return start

    for start in range(0, len(lines) - len(subseq) + 1):
        if _match_at(start):
            return start
    return None


def _apply_unified_diff(original, diff_text):
    try:
        original_lines = original.splitlines()
        diff_lines = diff_text.splitlines()
        if not diff_lines or not diff_lines[0].startswith("---"):
            return None

        idx = 0
        while idx < len(diff_lines) and diff_lines[idx].startswith("---"):
            idx += 1
        if idx < len(diff_lines) and diff_lines[idx].startswith("+++"):
            idx += 1

        out = []
        cursor = 0
        while idx < len(diff_lines):
            line = diff_lines[idx]
            if not line.startswith("@@"):
                return None

            # Allow optional lengths and optional trailing context after @@.
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?:.*)$", line)
            if not m:
                return None
            expected_start = max(cursor, int(m.group(1)) - 1)
            idx += 1

            hunk_lines = []
            while idx < len(diff_lines) and not diff_lines[idx].startswith("@@"):
                hline = diff_lines[idx]
                if hline.startswith((" ", "+", "-")):
                    hunk_lines.append(hline)
                elif hline.startswith("\\ No newline at end of file"):
                    pass
                else:
                    return None
                idx += 1

            old_block = [hl[1:] for hl in hunk_lines if hl.startswith((" ", "-"))]
            hunk_start = _find_subsequence_start(original_lines, old_block, expected_start)
            if hunk_start is None:
                return None

            out.extend(original_lines[cursor:hunk_start])
            orig_idx = hunk_start
            ref_indent = None
            for hline in hunk_lines:
                tag = hline[0]
                content = hline[1:]
                if tag == " ":
                    if orig_idx >= len(original_lines):
                        return None
                    current = original_lines[orig_idx]
                    out.append(current)
                    ref_indent = _leading_spaces(current)
                    orig_idx += 1
                elif tag == "-":
                    if orig_idx >= len(original_lines):
                        return None
                    ref_indent = _leading_spaces(original_lines[orig_idx])
                    orig_idx += 1
                elif tag == "+":
                    if content.strip() and ref_indent is not None:
                        content = (" " * ref_indent) + content.lstrip(" ")
                    out.append(content)
                else:
                    return None

            cursor = orig_idx

        out.extend(original_lines[cursor:])
        return "\n".join(out)
    except Exception:
        return None


def _validate_patch_diff(diff_text):
    changed = 0
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            changed += 1
        elif line.startswith('-') and not line.startswith('---'):
            changed += 1
    return changed <= 12


def _extract_json(text):
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # scan for the first decodable JSON object
    decoder = json.JSONDecoder()
    for i, ch in enumerate(stripped):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[i:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _normalize_patch_payload(patch):
    if not isinstance(patch, str):
        return None, "invalid"
    text = patch.strip()
    if not text:
        return None, "invalid"

    # Remove markdown fences if present.
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    lines = [ln.rstrip("\r") for ln in lines]
    if not lines:
        return None, "invalid"

    first = lines[0].lstrip()
    # Some outputs are full code mislabeled as diff.
    if first.startswith("import ") or first.startswith("from ") or first.startswith("def "):
        return "\n".join(lines).strip(), "full_code"

    # Drop git preamble lines.
    while lines and (
        lines[0].startswith("diff --git ")
        or lines[0].startswith("index ")
        or lines[0].startswith("new file mode ")
        or lines[0].startswith("deleted file mode ")
        or lines[0].startswith("--- /dev/null")
    ):
        lines = lines[1:]

    # If hunk-only diff is returned, inject generic headers.
    if lines and lines[0].startswith("@@"):
        lines = ["--- a/code.py", "+++ b/code.py"] + lines

    # If headers exist later, trim leading noise.
    if lines and not lines[0].startswith("---"):
        for i, ln in enumerate(lines):
            if ln.startswith("---"):
                lines = lines[i:]
                break

    if not lines:
        return None, "invalid"
    return "\n".join(lines).strip(), "diff"


def _validate_code(code):
    # Only allow numpy import
    for line in code.splitlines():
        if line.strip().startswith('import') or line.strip().startswith('from'):
            if line.strip() != 'import numpy as np':
                return False
    # Compile
    try:
        compiled = compile(code, '<string>', 'exec')
    except Exception:
        return False
    # Exec and smoke test
    env = {}
    try:
        exec(compiled, env)
        if 'score' not in env:
            return False
        score = env['score']
        bins = np.array([10.0, 7.0, 5.0])
        out = score(5, bins)
        if not isinstance(out, np.ndarray):
            return False
        if out.shape != bins.shape:
            return False
    except Exception:
        return False
    return True


def _as_float(value, default):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _collect_numeric_constant_indices(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return []

    indices = []
    idx = 0

    class _Visitor(ast.NodeVisitor):
        def visit_Constant(self, node):
            nonlocal idx
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                val = float(node.value)
                if np.isfinite(val) and abs(val) > 1e-12 and abs(val) != 1.0:
                    indices.append(idx)
                idx += 1
            self.generic_visit(node)

    _Visitor().visit(tree)
    return indices


def _mutate_numeric_constant(code, target_idx, factor):
    try:
        tree = ast.parse(code)
    except Exception:
        return None

    class _Mutator(ast.NodeTransformer):
        def __init__(self, target_idx, factor):
            self.target_idx = target_idx
            self.factor = factor
            self.idx = 0
            self.changed = False

        def visit_Constant(self, node):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if self.idx == self.target_idx:
                    val = node.value
                    new_val = float(val) * self.factor
                    if isinstance(val, int):
                        new_val = int(round(new_val))
                        if new_val == val:
                            new_val = val + (1 if self.factor > 1 else -1)
                    node = ast.copy_location(ast.Constant(value=new_val), node)
                    self.changed = True
                self.idx += 1
            return node

    mutator = _Mutator(target_idx, factor)
    tree = mutator.visit(tree)
    ast.fix_missing_locations(tree)
    if not mutator.changed:
        return None

    try:
        new_code = ast.unparse(tree)
    except Exception:
        return None

    if "import numpy as np" not in new_code:
        new_code = "import numpy as np\n\n" + new_code
    return new_code


def _generate_numeric_variants(code, max_variants=12):
    targets = _collect_numeric_constant_indices(code)
    if not targets:
        return []
    factors = [0.6, 0.75, 0.85, 1.15, 1.3, 1.5]
    variants = []
    seen = {code.strip()}
    for target_idx in targets[:10]:
        for factor in factors:
            mutated = _mutate_numeric_constant(code, target_idx, factor)
            if not mutated:
                continue
            key = mutated.strip()
            if key in seen:
                continue
            seen.add(key)
            variants.append(mutated)
            if len(variants) >= max_variants:
                return variants
    return variants


def _inject_before_return_scores(code, new_line):
    lines = code.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "return scores":
            indent = line[: len(line) - len(line.lstrip(" "))]
            lines = lines[:i] + [indent + new_line] + lines[i:]
            return "\n".join(lines)
    return None


def _replace_first_match(code, match, replacement):
    if not match:
        return None
    return code[: match.start()] + replacement + code[match.end() :]


def _replace_score_expression(code, expr_transform):
    match = re.search(r"(?m)^(\s*scores\s*=\s*)(.+)$", code)
    if not match:
        return None
    prefix = match.group(1)
    expr = match.group(2).strip()
    new_expr = expr_transform(expr)
    if not new_expr or new_expr == expr:
        return None
    new_line = prefix + new_expr
    return _replace_first_match(code, match, new_line)


def _generate_structural_variants(code, max_variants=24):
    variants = []
    seen = {code.strip()}

    def _add_variant(new_code):
        if not new_code:
            return False
        key = new_code.strip()
        if not key or key in seen:
            return False
        seen.add(key)
        variants.append(new_code)
        return len(variants) >= max_variants

    # Structural post-transforms before return.
    if _add_variant(_inject_before_return_scores(code, "scores = scores - np.mean(scores)")):
        return variants
    if _add_variant(_inject_before_return_scores(code, "scores = scores / (1.0 + np.std(scores))")):
        return variants
    if _add_variant(_inject_before_return_scores(code, "scores = np.tanh(scores)")):
        return variants
    if _add_variant(
        _inject_before_return_scores(
            code,
            "scores = scores + 0.05 * (bins / (np.max(bins) + 1e-9))",
        )
    ):
        return variants

    # Change penalty shape when penalty variables appear in score expression.
    score_match = re.search(r"(?m)^(\s*scores\s*=\s*)(.+)$", code)
    score_expr = score_match.group(2) if score_match else ""
    penalty_vars = sorted(
        set(re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*penalty[A-Za-z0-9_]*)\s*=", code))
    )
    for pen in penalty_vars[:6]:
        if pen not in score_expr:
            continue
        wrapped_log = _replace_score_expression(
            code,
            lambda expr, p=pen: expr.replace(
                p, f"np.log1p(np.maximum({p}, 0.0))", 1
            ),
        )
        if _add_variant(wrapped_log):
            return variants
        wrapped_sqrt = _replace_score_expression(
            code,
            lambda expr, p=pen: expr.replace(
                p, f"np.sqrt(np.maximum({p}, 0.0))", 1
            ),
        )
        if _add_variant(wrapped_sqrt):
            return variants
        wrapped_sq = _replace_score_expression(
            code,
            lambda expr, p=pen: expr.replace(
                p, f"np.square(np.maximum({p}, 0.0))", 1
            ),
        )
        if _add_variant(wrapped_sq):
            return variants

    # Add branch-sensitive residual term to score expression.
    branch_variant = _replace_score_expression(
        code,
        lambda expr: (
            f"({expr}) + 0.15 * np.where(bins >= item, "
            f"(bins - item) / (np.max(bins) + 1e-9), -1.0)"
        ),
    )
    if _add_variant(branch_variant):
        return variants

    # Exponent knobs in nonlinear terms.
    for match in list(re.finditer(r"\*\*\s*(\d+)", code))[:10]:
        try:
            exp = int(match.group(1))
        except Exception:
            continue
        if exp < 1 or exp > 8:
            continue
        for new_exp in (max(1, exp - 1), min(8, exp + 1)):
            if new_exp == exp:
                continue
            new_code = _replace_first_match(code, match, f"** {new_exp}")
            if _add_variant(new_code):
                return variants

    # Threshold knobs (small shifts + comparator strictness).
    threshold_matches = list(re.finditer(r"([<>]=?)\s*(-?\d+(?:\.\d+)?)", code))
    for match in threshold_matches[:10]:
        op = match.group(1)
        num_txt = match.group(2)
        try:
            val = float(num_txt)
        except Exception:
            continue
        if not np.isfinite(val) or abs(val) > 10:
            continue
        for factor in (0.85, 1.15):
            new_val = val * factor
            new_num = f"{new_val:.6g}"
            replaced = _replace_first_match(code, match, f"{op} {new_num}")
            if _add_variant(replaced):
                return variants
        strict_op = None
        if op == ">":
            strict_op = ">="
        elif op == "<":
            strict_op = "<="
        if strict_op is not None:
            replaced = _replace_first_match(code, match, f"{strict_op} {num_txt}")
            if _add_variant(replaced):
                return variants

    return variants


def _generate_fallback_variants(code, structural_max=24, numeric_max=24):
    variants = []
    seen = {code.strip()}

    for candidate in _generate_structural_variants(code, structural_max):
        key = candidate.strip()
        if key and key not in seen:
            seen.add(key)
            variants.append(candidate)

    for candidate in _generate_numeric_variants(code, numeric_max):
        key = candidate.strip()
        if key and key not in seen:
            seen.add(key)
            variants.append(candidate)

    return variants


def _token_signature_without_numbers(code):
    try:
        tokens = []
        reader = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(reader):
            tok_type = tok.type
            if tok_type in (
                tokenize.NUMBER,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.COMMENT,
                tokenize.ENDMARKER,
            ):
                continue
            tokens.append((tok_type, tok.string))
        return tokens
    except Exception:
        return None


def _is_numeric_only_edit(original_code, patched_code):
    if not isinstance(original_code, str) or not isinstance(patched_code, str):
        return False
    if original_code.strip() == patched_code.strip():
        return False

    original_sig = _token_signature_without_numbers(original_code)
    patched_sig = _token_signature_without_numbers(patched_code)
    if original_sig is None or patched_sig is None:
        return False
    return original_sig == patched_sig


def _is_patch_better(parent_eval, patched_eval, eps, worst_delta, fail_delta):
    parent_eval = parent_eval or {}
    patched_eval = patched_eval or {}

    parent_signals = parent_eval.get("signals") or {}
    patched_signals = patched_eval.get("signals") or {}

    parent_mean = _as_float(parent_signals.get("mean_objective", parent_eval.get("objective")), float("inf"))
    patched_mean = _as_float(patched_signals.get("mean_objective", patched_eval.get("objective")), float("inf"))

    parent_worst = _as_float(parent_signals.get("worst_objective", parent_mean), float("inf"))
    patched_worst = _as_float(patched_signals.get("worst_objective", patched_mean), float("inf"))

    parent_fail = _as_float(parent_signals.get("fail_rate"), 1.0)
    patched_fail = _as_float(patched_signals.get("fail_rate"), 1.0)

    improved_mean = patched_mean < (parent_mean - eps)
    non_worse_mean = patched_mean <= (parent_mean + eps)
    safe_tail = patched_worst <= (parent_worst + worst_delta)
    safe_fail = patched_fail <= (parent_fail + fail_delta)
    better_tail = patched_worst < (parent_worst - eps)
    better_fail = patched_fail < (parent_fail - eps)

    if improved_mean and safe_tail and safe_fail:
        return True
    if non_worse_mean and safe_tail and better_fail:
        return True
    if non_worse_mean and safe_fail and better_tail:
        return True
    return False


class EOH_DIAG:
    def __init__(self, paras, problem, select, manage, **kwargs):
        self.prob = problem
        self.select = select
        self.manage = manage

        self.use_local_llm = paras.llm_use_local
        self.llm_local_url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        self.pop_size = paras.ec_pop_size
        self.n_pop = paras.ec_n_pop
        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        if paras.ec_m > self.pop_size or paras.ec_m == 1:
            print("m should not be larger than pop size or smaller than 2, adjust it to m=2")
            paras.ec_m = 2
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode
        self.ndelay = 1

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path
        self.exp_n_proc = paras.exp_n_proc
        self.timeout = paras.eva_timeout
        self.use_numba = paras.eva_numba_decorator
        self.diag_top_k = 3
        self.diag_quick_instances = 2
        self.diag_quick_items = 1500
        self.diag_accept_eps = 1e-5
        self.diag_accept_worst_delta = 5e-4
        self.diag_accept_fail_delta = 0.01
        self.diag_patch_trials = 4
        self.diag_beam_size = 3
        self.diag_beam_full_k = 2
        self.diag_numeric_variants = 60
        self.diag_structural_variants = 36
        self.diag_min_objective_gain_strict = 0.002
        self.diag_min_objective_gain_relaxed = 0.001
        self.diag_min_objective_gain = self.diag_min_objective_gain_strict
        self.diag_stagnation_window = 2
        self.diag_stagnant_generations = 0
        self.diag_best_seen_objective = None
        self.diag_quick_relax_delta = 0.001
        self.diag_force_full_eval_every = 4
        self.diag_reject_numeric_only_llm = True

        self.diag_llm = InterfaceLLM(
            self.api_endpoint,
            self.api_key,
            self.llm_model,
            self.use_local_llm,
            self.llm_local_url,
            self.debug_mode,
        )

        # Determinism
        self.random_seed = getattr(paras, "exp_random_seed", 2024)
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        print("- EoH parameters loaded -")

    def _update_adaptive_min_gain(self, current_best):
        if current_best is None:
            return

        if self.diag_best_seen_objective is None:
            self.diag_best_seen_objective = current_best
            self.diag_stagnant_generations = 0
            self.diag_min_objective_gain = self.diag_min_objective_gain_strict
            return

        improved = current_best < (self.diag_best_seen_objective - self.diag_accept_eps)
        if improved:
            self.diag_best_seen_objective = current_best
            self.diag_stagnant_generations = 0
            self.diag_min_objective_gain = self.diag_min_objective_gain_strict
            return

        self.diag_stagnant_generations += 1
        if self.diag_stagnant_generations >= self.diag_stagnation_window:
            self.diag_min_objective_gain = self.diag_min_objective_gain_relaxed

    def _decode_diag_candidate(self, base_code, response):
        if not response:
            return None, "invalid_diag_json", response

        diag_json = _extract_json(response)
        repaired_response = response
        if diag_json is None:
            repair_prompt = (
                "Reformat the following output into valid JSON only, no markdown fences, "
                "matching keys: failure_modes, evidence, edit_plan, patch_type, patch.\n"
                "OUTPUT:\n" + response
            )
            repaired = self.diag_llm.get_response(repair_prompt)
            if repaired:
                repaired_response = repaired
                diag_json = _extract_json(repaired)

        if not (diag_json and isinstance(diag_json, dict)):
            return None, "invalid_diag_json", repaired_response

        patch_type = str(diag_json.get("patch_type", "")).strip().lower()
        patch = diag_json.get("patch")

        if patch_type == "diff" and isinstance(patch, str):
            patch_norm, patch_kind = _normalize_patch_payload(patch)
            if patch_kind == "full_code":
                patched_code = patch_norm
            elif patch_kind == "diff" and patch_norm is not None and _validate_patch_diff(patch_norm):
                patched_code = _apply_unified_diff(base_code, patch_norm)
            else:
                patched_code = None
        elif patch_type == "full_code" and isinstance(patch, str):
            patch_norm, patch_kind = _normalize_patch_payload(patch)
            if patch_kind == "diff" and patch_norm is not None and _validate_patch_diff(patch_norm):
                patched_code = _apply_unified_diff(base_code, patch_norm)
            else:
                patched_code = patch_norm
        else:
            patched_code = None

        if not (patched_code and _validate_code(patched_code)):
            return None, "invalid_patch_code", repaired_response

        if self.diag_reject_numeric_only_llm and _is_numeric_only_edit(base_code, patched_code):
            return None, "rejected_numeric_only_patch", repaired_response

        return {
            "diag_json": diag_json,
            "patched_code": patched_code,
            "response": repaired_response,
        }, None, repaired_response

    def _diag_prompt(self, summary):
        return (
            "You are a code diagnostics assistant. Given a heuristic 'score' function and performance signals, "
            "produce a STRUCTURAL patch that changes decision behavior. Reply ONLY with strict JSON matching this schema:\n"
            "{\n"
            "  \"failure_modes\": [string],\n"
            "  \"evidence\": [string],\n"
            "  \"edit_plan\": string,\n"
            "  \"patch_type\": \"diff\" | \"full_code\",\n"
            "  \"patch\": string\n"
            "}\n"
            "Constraints:\n"
            "- Keep function name and signature score(item: int, bins: np.ndarray) -> np.ndarray\n"
            "- Prefer patch_type=full_code unless a compact valid diff is certain\n"
            "- Change <= 12 lines if patch_type=diff\n"
            "- Patch must include at least one structural change (new term/transform/threshold/branch/ranking logic)\n"
            "- Do NOT provide coefficient-only edits (changing numeric constants alone is invalid)\n"
            "- No placeholders like TODO/pass/your code here\n"
            "- Must return variable 'scores'\n"
            "- Do not import anything besides numpy\n"
            "- Reply with JSON only (no markdown fences)\n"
            "- If patch_type=diff, patch MUST be unified diff and start with:\n"
            "  --- a/code.py\n"
            "  +++ b/code.py\n"
            "- Do NOT include diff --git or index lines\n"
            "- If you cannot provide a valid diff, use patch_type=full_code\n"
            "\nSUMMARY JSON:\n"
            + json.dumps(summary)
        )

    def _run_diagnosis(self, best_ind, gen_id, rank_id):
        if not hasattr(self.prob, 'evaluate_with_signals'):
            return None, None, None, None, {"reason": "signals_not_supported"}

        diag_dir = Path(self.output_path) / "results" / "diag_logs"
        diag_dir.mkdir(parents=True, exist_ok=True)

        parent_eval = self.prob.evaluate_with_signals(best_ind['code'])
        parent_eval_quick = self.prob.evaluate_with_signals(
            best_ind['code'],
            max_instances=self.diag_quick_instances,
            max_items=self.diag_quick_items,
        )
        summary = {
            "objective": parent_eval.get("objective"),
            "signals": parent_eval.get("signals"),
            "quick_objective": parent_eval_quick.get("objective"),
            "quick_signals": parent_eval_quick.get("signals"),
            "per_instance": parent_eval.get("per_instance", []),
            "algorithm": best_ind.get("algorithm"),
            "code": best_ind.get("code"),
        }

        patch_applied = None
        last_response = ""
        result_meta = {
            "generation": gen_id,
            "rank": rank_id,
            "parent_objective": parent_eval.get("objective"),
            "parent_quick_objective": parent_eval_quick.get("objective"),
            "patched_quick_objective": None,
            "patched_objective": None,
            "accepted": False,
            "reason": "no_patch",
            "min_gain_used": self.diag_min_objective_gain,
        }

        base_prompt = self._diag_prompt(summary)
        feedback_reason = None
        beam_focus_hints = [
            "Focus on thresholding and branch logic near full bins.",
            "Focus on penalty-shape transforms for smoother risk control.",
            "Focus on nonlinear interaction between load-balance and fit quality.",
        ]
        for trial in range(self.diag_patch_trials):
            trial_candidates = []
            trial_responses = []
            decode_fail_counts = {}

            for beam_idx in range(self.diag_beam_size):
                if trial == 0 and beam_idx == 0:
                    prompt = base_prompt
                else:
                    hint = beam_focus_hints[beam_idx % len(beam_focus_hints)]
                    prompt = (
                        base_prompt
                        + "\nPrevious patch was rejected with reason: "
                        + str(feedback_reason)
                        + f". Candidate {beam_idx + 1}/{self.diag_beam_size}: {hint} "
                        + "Provide a different full_code patch with a structural change."
                    )

                response = self.diag_llm.get_response(prompt)
                decoded, decode_reason, repaired_response = self._decode_diag_candidate(
                    best_ind.get("code", ""),
                    response,
                )
                trial_responses.append(f"[trial {trial + 1} beam {beam_idx + 1}]\n{repaired_response or ''}")
                if decoded is None:
                    decode_fail_counts[decode_reason] = decode_fail_counts.get(decode_reason, 0) + 1
                    continue
                trial_candidates.append(decoded)

            if trial_responses:
                last_response = "\n\n".join(trial_responses)

            if not trial_candidates:
                if decode_fail_counts:
                    feedback_reason = max(decode_fail_counts.items(), key=lambda x: x[1])[0]
                else:
                    feedback_reason = "invalid_patch_code"
                result_meta["reason"] = feedback_reason
                continue

            parent_quick_obj = _as_float(parent_eval_quick.get("objective"), float("inf"))
            force_full_eval = (
                self.diag_force_full_eval_every > 0
                and (trial + 1) % self.diag_force_full_eval_every == 0
            )
            quick_pass = []
            for candidate in trial_candidates:
                patched_eval_quick = self.prob.evaluate_with_signals(
                    candidate["patched_code"],
                    max_instances=self.diag_quick_instances,
                    max_items=self.diag_quick_items,
                )
                patched_quick_obj = _as_float(patched_eval_quick.get("objective"), float("inf"))
                quick_ok = _is_patch_better(
                    parent_eval_quick,
                    patched_eval_quick,
                    self.diag_accept_eps,
                    self.diag_accept_worst_delta,
                    self.diag_accept_fail_delta,
                )
                quick_soft_ok = patched_quick_obj <= (parent_quick_obj + self.diag_quick_relax_delta)
                candidate["patched_eval_quick"] = patched_eval_quick
                candidate["quick_obj"] = patched_quick_obj
                candidate["quick_ok"] = bool(quick_ok or quick_soft_ok)
                if candidate["quick_ok"]:
                    quick_pass.append(candidate)

            if not quick_pass and force_full_eval:
                quick_pass = sorted(trial_candidates, key=lambda c: c.get("quick_obj", float("inf")))

            if not quick_pass:
                best_quick = min((c.get("quick_obj", float("inf")) for c in trial_candidates), default=float("inf"))
                if np.isfinite(best_quick):
                    result_meta["patched_quick_objective"] = float(best_quick)
                feedback_reason = "rejected_quick_gate"
                result_meta["reason"] = feedback_reason
                continue

            ranked_quick = sorted(quick_pass, key=lambda c: c.get("quick_obj", float("inf")))
            full_candidates = ranked_quick[: max(1, self.diag_beam_full_k)]
            accepted_candidates = []
            parent_obj = _as_float(parent_eval.get("objective"), float("inf"))
            for candidate in full_candidates:
                patched_eval = self.prob.evaluate_with_signals(candidate["patched_code"])
                patched_obj = _as_float(patched_eval.get("objective"), float("inf"))
                gain = parent_obj - patched_obj
                full_gate_ok = _is_patch_better(
                    parent_eval,
                    patched_eval,
                    self.diag_accept_eps,
                    self.diag_accept_worst_delta,
                    self.diag_accept_fail_delta,
                )
                min_gain_ok = gain >= self.diag_min_objective_gain
                candidate["patched_eval"] = patched_eval
                candidate["patched_obj"] = patched_obj
                candidate["gain"] = gain
                if full_gate_ok and min_gain_ok:
                    candidate["reject_reason"] = None
                    accepted_candidates.append(candidate)
                else:
                    if not full_gate_ok:
                        candidate["reject_reason"] = "rejected_full_gate"
                    else:
                        candidate["reject_reason"] = "rejected_min_gain"

            if accepted_candidates:
                best_candidate = min(accepted_candidates, key=lambda c: c.get("patched_obj", float("inf")))
                result_meta["accepted"] = True
                result_meta["reason"] = "accepted"
                result_meta["patched_quick_objective"] = best_candidate["patched_eval_quick"].get("objective")
                result_meta["patched_objective"] = best_candidate["patched_eval"].get("objective")
                patch_applied = best_candidate["patched_code"]
                new_ind = {
                    "algorithm": best_ind.get("algorithm"),
                    "code": best_candidate["patched_code"],
                    "objective": best_candidate["patched_eval"].get("objective"),
                    "other_inf": None,
                    "diagnosis": best_candidate["diag_json"],
                    "parent_objective": best_ind.get('objective'),
                    "patched": True,
                }
                return new_ind, summary, best_candidate.get("response", last_response), patch_applied, result_meta

            best_full = min(full_candidates, key=lambda c: c.get("patched_obj", float("inf")))
            result_meta["patched_quick_objective"] = best_full["patched_eval_quick"].get("objective")
            result_meta["patched_objective"] = best_full["patched_eval"].get("objective")
            feedback_reason = best_full.get("reject_reason") or "rejected_full_gate"
            result_meta["reason"] = feedback_reason

        # Deterministic fallback: structural knobs + numeric local search.
        fallback_best = None
        for candidate_code in _generate_fallback_variants(
            best_ind.get("code", ""),
            structural_max=self.diag_structural_variants,
            numeric_max=self.diag_numeric_variants,
        ):
            if not _validate_code(candidate_code):
                continue
            patched_eval_quick = self.prob.evaluate_with_signals(
                candidate_code,
                max_instances=self.diag_quick_instances,
                max_items=self.diag_quick_items,
            )
            quick_ok = _is_patch_better(
                parent_eval_quick,
                patched_eval_quick,
                self.diag_accept_eps,
                self.diag_accept_worst_delta,
                self.diag_accept_fail_delta,
            )
            parent_quick_obj = _as_float(parent_eval_quick.get("objective"), float("inf"))
            patched_quick_obj = _as_float(patched_eval_quick.get("objective"), float("inf"))
            quick_soft_ok = patched_quick_obj <= (parent_quick_obj + self.diag_quick_relax_delta)
            if not (quick_ok or quick_soft_ok):
                continue
            patched_eval = self.prob.evaluate_with_signals(candidate_code)
            full_ok = _is_patch_better(
                parent_eval,
                patched_eval,
                self.diag_accept_eps,
                self.diag_accept_worst_delta,
                self.diag_accept_fail_delta,
            )
            if not full_ok:
                continue
            cand_obj = patched_eval.get("objective")
            if cand_obj is None:
                continue
            gain = _as_float(parent_eval.get("objective"), float("inf")) - _as_float(cand_obj, float("inf"))
            if gain < self.diag_min_objective_gain:
                continue
            if fallback_best is None or cand_obj < fallback_best["objective"]:
                fallback_best = {
                    "code": candidate_code,
                    "objective": cand_obj,
                    "quick_objective": patched_eval_quick.get("objective"),
                }

        if fallback_best is not None:
            prior_reason = result_meta.get("reason")
            result_meta["accepted"] = True
            result_meta["reason"] = "accepted_numeric_fallback"
            result_meta["patched_quick_objective"] = fallback_best.get("quick_objective")
            result_meta["patched_objective"] = fallback_best.get("objective")
            patch_applied = fallback_best["code"]
            new_ind = {
                "algorithm": best_ind.get("algorithm"),
                "code": fallback_best["code"],
                "objective": fallback_best["objective"],
                "other_inf": None,
                "diagnosis": {
                    "patch_type": "full_code",
                    "source": "mixed_fallback",
                    "base_reason": prior_reason,
                },
                "parent_objective": best_ind.get('objective'),
                "patched": True,
            }
            return new_ind, summary, last_response, patch_applied, result_meta

        # Log even if no improvement
        return None, summary, last_response, patch_applied, result_meta

    def run(self):
        print("- Evolution Start -")
        time_start = time.time()

        interface_prob = self.prob
        interface_ec = InterfaceECDiag(
            self.pop_size,
            self.m,
            self.api_endpoint,
            self.api_key,
            self.llm_model,
            self.use_local_llm,
            self.llm_local_url,
            self.debug_mode,
            interface_prob,
            select=self.select,
            n_p=self.exp_n_proc,
            timeout=self.timeout,
            use_numba=self.use_numba,
        )

        population = []
        if self.use_seed:
            with open(self.seed_path) as file:
                data = json.load(file)
            population = interface_ec.population_generation_seed(data, self.exp_n_proc)
            filename = self.output_path + "/results/pops/population_generation_0.json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)
            n_start = 0
        else:
            if self.load_pop:
                print("load initial population from " + self.load_pop_path)
                with open(self.load_pop_path) as file:
                    data = json.load(file)
                for individual in data:
                    population.append(individual)
                print("initial population has been loaded!")
                n_start = self.load_pop_id
            else:
                print("creating initial population:")
                population = interface_ec.population_generation()
                population = self.manage.population_management(population, self.pop_size)

                print("Pop initial: ")
                for off in population:
                    print(" Obj: ", off['objective'], end="|")
                print()
                print("initial population has been created!")
                filename = self.output_path + "/results/pops/population_generation_0.json"
                with open(filename, 'w') as f:
                    json.dump(population, f, indent=5)
                n_start = 0

        n_op = len(self.operators)
        if population:
            initial_best = min(ind.get("objective", float("inf")) for ind in population)
            self._update_adaptive_min_gain(initial_best)

        for pop in range(n_start, self.n_pop):
            print(
                f"Diag min gain (gen {pop + 1}): {self.diag_min_objective_gain:.4f} "
                f"[stagnation={self.diag_stagnant_generations}]"
            )
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]
                offsprings = []
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                for off in offsprings:
                    population.append(off)
                    print(" Obj: ", off['objective'], end="|")
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()

            # Diagnosis for top-k candidates and only keep the best accepted patch.
            if hasattr(interface_prob, 'evaluate_with_signals') and population:
                diag_dir = Path(self.output_path) / "results" / "diag_logs"
                diag_dir.mkdir(parents=True, exist_ok=True)

                accepted_patches = []
                top_k = min(self.diag_top_k, len(population))
                for rank in range(top_k):
                    candidate = population[rank]
                    new_ind, summary, response, patch_applied, result_meta = self._run_diagnosis(
                        candidate, pop + 1, rank + 1
                    )

                    stamp = f"gen_{pop+1}_rank_{rank+1}_{int(time.time()*1000)}"
                    (diag_dir / f"{stamp}_input.json").write_text(json.dumps(summary, indent=2))
                    (diag_dir / f"{stamp}_raw.txt").write_text(response or "")
                    (diag_dir / f"{stamp}_patch.txt").write_text(patch_applied or "")
                    (diag_dir / f"{stamp}_result.json").write_text(json.dumps(result_meta, indent=2))

                    if new_ind is not None:
                        accepted_patches.append(new_ind)

                if accepted_patches:
                    best_patch = min(accepted_patches, key=lambda ind: ind.get("objective", float("inf")))
                    population.append(best_patch)
                    population = self.manage.population_management(population, min(len(population), self.pop_size))

            if population:
                best_after_gen = min(ind.get("objective", float("inf")) for ind in population)
                self._update_adaptive_min_gain(best_after_gen)

            filename = self.output_path + "/results/pops/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population, f, indent=5)

            filename = self.output_path + "/results/pops_best/population_generation_" + str(pop + 1) + ".json"
            with open(filename, 'w') as f:
                json.dump(population[0], f, indent=5)

            print(f"--- {pop + 1} of {self.n_pop} populations finished. Time Cost:  {((time.time()-time_start)/60):.1f} m")
            print("Pop Objs: ", end=" ")
            for i in range(len(population)):
                print(str(population[i]['objective']) + " ", end="")
            print()
