import json
import random
import time
import re
from pathlib import Path
import numpy as np

from .eoh_diag_interface_EC import InterfaceECDiag
from ..selection import prob_rank
from ..management import pop_greedy
from ...llm.interface_LLM import InterfaceLLM


def _apply_unified_diff(original, diff_text):
    original_lines = original.splitlines()
    diff_lines = diff_text.splitlines()
    if not diff_lines:
        return None
    if not diff_lines[0].startswith('---'):
        return None
    idx = 0
    # Skip file headers
    while idx < len(diff_lines) and diff_lines[idx].startswith('---'):
        idx += 1
    if idx < len(diff_lines) and diff_lines[idx].startswith('+++'):
        idx += 1

    out = []
    orig_idx = 0
    while idx < len(diff_lines):
        line = diff_lines[idx]
        if line.startswith('@@'):
            # Parse hunk header
            m = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", line)
            if not m:
                return None
            start_old = int(m.group(1)) - 1
            # Copy unchanged lines before hunk
            out.extend(original_lines[orig_idx:start_old])
            orig_idx = start_old
            idx += 1
            # Apply hunk
            while idx < len(diff_lines) and not diff_lines[idx].startswith('@@'):
                hline = diff_lines[idx]
                if hline.startswith(' '):
                    out.append(original_lines[orig_idx])
                    orig_idx += 1
                elif hline.startswith('-'):
                    orig_idx += 1
                elif hline.startswith('+'):
                    out.append(hline[1:])
                idx += 1
        else:
            return None
    # Copy remaining
    out.extend(original_lines[orig_idx:])
    return "\n".join(out)


def _validate_patch_diff(diff_text):
    changed = 0
    for line in diff_text.splitlines():
        if line.startswith('+') and not line.startswith('+++'):
            changed += 1
        elif line.startswith('-') and not line.startswith('---'):
            changed += 1
    return changed <= 12


def _extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        # try to extract JSON block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


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

        self.diag_llm = InterfaceLLM(
            self.api_endpoint,
            self.api_key,
            self.llm_model,
            self.use_local_llm,
            self.llm_local_url,
            self.debug_mode,
        )

        # Determinism
        random.seed(2024)
        np.random.seed(2024)

        print("- EoH parameters loaded -")

    def _diag_prompt(self, summary):
        return (
            "You are a code diagnostics assistant. Given a heuristic 'score' function and performance signals, "
            "produce a SMALL targeted patch. Reply ONLY with strict JSON matching this schema:\n"
            "{\n"
            "  \"failure_modes\": [string],\n"
            "  \"evidence\": [string],\n"
            "  \"edit_plan\": string,\n"
            "  \"patch_type\": \"diff\" | \"full_code\",\n"
            "  \"patch\": string\n"
            "}\n"
            "Constraints:\n"
            "- Keep function name and signature score(item: int, bins: np.ndarray) -> np.ndarray\n"
            "- Change <= 12 lines if patch_type=diff\n"
            "- No placeholders like TODO/pass/your code here\n"
            "- Must return variable 'scores'\n"
            "- Do not import anything besides numpy\n"
            "\nSUMMARY JSON:\n"
            + json.dumps(summary)
        )

    def _run_diagnosis(self, best_ind, gen_id):
        if not hasattr(self.prob, 'evaluate_with_signals'):
            return None

        diag_dir = Path(self.output_path) / "results" / "diag_logs"
        diag_dir.mkdir(parents=True, exist_ok=True)

        signals = self.prob.evaluate_with_signals(best_ind['code'])
        summary = {
            "objective": signals.get("objective"),
            "signals": signals.get("signals"),
            "per_instance": signals.get("per_instance", []),
            "algorithm": best_ind.get("algorithm"),
            "code": best_ind.get("code"),
        }

        prompt = self._diag_prompt(summary)
        response = self.diag_llm.get_response(prompt)

        diag_json = _extract_json(response) if response else None
        patch_applied = None
        patched_obj = None

        if diag_json and isinstance(diag_json, dict):
            patch_type = diag_json.get("patch_type")
            patch = diag_json.get("patch")

            if patch_type == "diff" and isinstance(patch, str) and _validate_patch_diff(patch):
                patched_code = _apply_unified_diff(best_ind['code'], patch)
            elif patch_type == "full_code" and isinstance(patch, str):
                patched_code = patch
            else:
                patched_code = None

            if patched_code and _validate_code(patched_code):
                patched_eval = self.prob.evaluate_with_signals(patched_code)
                patched_obj = patched_eval.get("objective")
                if patched_obj is not None and patched_obj < best_ind.get('objective', float('inf')):
                    new_ind = {
                        "algorithm": best_ind.get("algorithm"),
                        "code": patched_code,
                        "objective": patched_obj,
                        "other_inf": None,
                        "diagnosis": diag_json,
                        "parent_objective": best_ind.get('objective'),
                        "patched": True,
                    }
                    patch_applied = patched_code
                    return new_ind, summary, response, patch_applied, patched_obj

        # Log even if no improvement
        return None, summary, response, patch_applied, patched_obj

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

        for pop in range(n_start, self.n_pop):
            for i in range(n_op):
                op = self.operators[i]
                print(f" OP: {op}, [{i + 1} / {n_op}] ", end="|")
                op_w = self.operator_weights[i]
                if (np.random.rand() < op_w):
                    parents, offsprings = interface_ec.get_algorithm(population, op)
                for off in offsprings:
                    population.append(off)
                    print(" Obj: ", off['objective'], end="|")
                size_act = min(len(population), self.pop_size)
                population = self.manage.population_management(population, size_act)
                print()

            # diagnosis for top-1 candidate (bp_online only)
            if hasattr(interface_prob, 'evaluate_with_signals') and population:
                best = population[0]
                new_ind, summary, response, patch_applied, patched_obj = self._run_diagnosis(best, pop + 1)

                # logging
                diag_dir = Path(self.output_path) / "results" / "diag_logs"
                diag_dir.mkdir(parents=True, exist_ok=True)
                stamp = f"gen_{pop+1}_{int(time.time())}"
                (diag_dir / f"{stamp}_input.json").write_text(json.dumps(summary, indent=2))
                (diag_dir / f"{stamp}_raw.txt").write_text(response or "")
                (diag_dir / f"{stamp}_patch.txt").write_text(patch_applied or "")
                (diag_dir / f"{stamp}_result.json").write_text(
                    json.dumps({
                        "parent_objective": best.get('objective'),
                        "patched_objective": patched_obj,
                    }, indent=2)
                )

                if new_ind is not None:
                    population.append(new_ind)
                    population = self.manage.population_management(population, min(len(population), self.pop_size))

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
