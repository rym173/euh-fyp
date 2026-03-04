import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from eoh import eoh
from eoh.utils.getParas import Paras


def _parse_list(value, cast=str):
    items = [x.strip() for x in value.split(",") if x.strip()]
    return [cast(x) for x in items]


def _collect_objective_trajectory(output_dir):
    pops_best = output_dir / "results" / "pops_best"
    if not pops_best.exists():
        return []
    files = sorted(
        pops_best.glob("population_generation_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    trajectory = []
    for f in files:
        try:
            payload = json.loads(f.read_text())
            obj = payload.get("objective")
            if obj is not None:
                trajectory.append(float(obj))
        except Exception:
            continue
    return trajectory


def _build_paras(method, seed, output_path, args):
    paras = Paras()
    paras.set_paras(
        method=method,
        problem=args.problem,
        llm_api_endpoint=args.llm_api_endpoint,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        ec_operators=args.ec_operators,
        ec_operator_weights=args.ec_operator_weights,
        ec_pop_size=args.ec_pop_size,
        ec_n_pop=args.ec_n_pop,
        exp_n_proc=args.exp_n_proc,
        eva_numba_decorator=args.eva_numba_decorator,
        exp_debug_mode=args.exp_debug_mode,
        exp_output_path=str(output_path),
        exp_random_seed=seed,
    )
    paras.eva_timeout = args.eva_timeout
    return paras


def _aggregate_by_method(rows):
    grouped = {}
    for row in rows:
        if row.get("error"):
            continue
        grouped.setdefault(row["method"], []).append(row)

    agg = {}
    for method, vals in grouped.items():
        final_objs = [r["final_best_objective"] for r in vals if r.get("final_best_objective") is not None]
        runtimes = [r["runtime_sec"] for r in vals if r.get("runtime_sec") is not None]
        agg[method] = {
            "n_runs": len(vals),
            "final_best_mean": statistics.mean(final_objs) if final_objs else None,
            "final_best_std": statistics.pstdev(final_objs) if len(final_objs) > 1 else 0.0 if final_objs else None,
            "final_best_min": min(final_objs) if final_objs else None,
            "final_best_max": max(final_objs) if final_objs else None,
            "runtime_mean_sec": statistics.mean(runtimes) if runtimes else None,
        }
    return agg


def _seed_winners(rows):
    by_seed = {}
    for row in rows:
        if row.get("error") or row.get("final_best_objective") is None:
            continue
        by_seed.setdefault(row["seed"], {})[row["method"]] = row["final_best_objective"]

    winners = []
    for seed, vals in sorted(by_seed.items()):
        if "eoh" in vals and "eoh_diag" in vals:
            eoh_obj = vals["eoh"]
            diag_obj = vals["eoh_diag"]
            if diag_obj < eoh_obj:
                winner = "eoh_diag"
            elif eoh_obj < diag_obj:
                winner = "eoh"
            else:
                winner = "tie"
            winners.append(
                {
                    "seed": seed,
                    "eoh_final_best": eoh_obj,
                    "eoh_diag_final_best": diag_obj,
                    "winner": winner,
                }
            )
    return winners


def main():
    parser = argparse.ArgumentParser(
        description="Run EOH and EOH_DIAG on the same problem instances across multiple seeds."
    )
    parser.add_argument("--problem", default="bp_online")
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=2024)
    parser.add_argument("--output-root", default="./examples/bp_online/comparison_runs")

    parser.add_argument("--llm-api-endpoint", default="http://vllm-nodeport.vllm-ns.svc.cluster.local:8000")
    parser.add_argument("--llm-api-key", default="EMPTY")
    parser.add_argument("--llm-model", default="Qwen3.5-122B-A10B-FP8")

    parser.add_argument("--ec-operators", default="e1,e2,m1,m2")
    parser.add_argument("--ec-operator-weights", default="1,1,1,1")
    parser.add_argument("--ec-pop-size", type=int, default=6)
    parser.add_argument("--ec-n-pop", type=int, default=6)
    parser.add_argument("--exp-n-proc", type=int, default=2)
    parser.add_argument("--eva-timeout", type=int, default=90)
    parser.add_argument("--eva-numba-decorator", action="store_true")
    parser.add_argument("--exp-debug-mode", action="store_true")
    args = parser.parse_args()

    args.ec_operators = _parse_list(args.ec_operators, str)
    args.ec_operator_weights = _parse_list(args.ec_operator_weights, float)
    if len(args.ec_operators) != len(args.ec_operator_weights):
        raise ValueError("Length mismatch: ec_operators vs ec_operator_weights")

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    methods = ["eoh", "eoh_diag"]
    runs = []

    print(f"Output root: {root}")
    print(f"Methods: {methods}")
    print(f"Seeds: {[args.seed_start + i for i in range(args.num_seeds)]}")
    print(f"Operators: {args.ec_operators}")

    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        for method in methods:
            run_dir = root / f"seed_{seed}" / method
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== Running method={method} seed={seed} ===")
            print(f"Run directory: {run_dir}")

            start = time.time()
            row = {
                "method": method,
                "seed": seed,
                "output_path": str(run_dir),
                "runtime_sec": None,
                "final_best_objective": None,
                "trajectory": [],
                "error": None,
            }
            try:
                paras = _build_paras(method, seed, run_dir, args)
                evolution = eoh.EVOL(paras)
                evolution.run()
                row["trajectory"] = _collect_objective_trajectory(run_dir)
                if row["trajectory"]:
                    row["final_best_objective"] = row["trajectory"][-1]
            except Exception as exc:
                row["error"] = str(exc)
                print(f"Run failed for method={method}, seed={seed}: {exc}")
            finally:
                row["runtime_sec"] = round(time.time() - start, 3)
                runs.append(row)

    summary = {
        "config": {
            "problem": args.problem,
            "num_seeds": args.num_seeds,
            "seed_start": args.seed_start,
            "methods": methods,
            "llm_api_endpoint": args.llm_api_endpoint,
            "llm_model": args.llm_model,
            "ec_operators": args.ec_operators,
            "ec_operator_weights": args.ec_operator_weights,
            "ec_pop_size": args.ec_pop_size,
            "ec_n_pop": args.ec_n_pop,
            "exp_n_proc": args.exp_n_proc,
            "eva_timeout": args.eva_timeout,
            "output_root": str(root),
        },
        "runs": runs,
        "aggregate_by_method": _aggregate_by_method(runs),
        "seed_winners": _seed_winners(runs),
    }

    summary_json = root / "comparison_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2))

    runs_csv = root / "comparison_runs.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "seed",
                "final_best_objective",
                "runtime_sec",
                "output_path",
                "error",
            ],
        )
        writer.writeheader()
        for r in runs:
            writer.writerow(
                {
                    "method": r["method"],
                    "seed": r["seed"],
                    "final_best_objective": r["final_best_objective"],
                    "runtime_sec": r["runtime_sec"],
                    "output_path": r["output_path"],
                    "error": r["error"],
                }
            )

    print("\n=== Comparison finished ===")
    print(f"Summary JSON: {summary_json}")
    print(f"Runs CSV: {runs_csv}")
    print("Aggregate:")
    print(json.dumps(summary["aggregate_by_method"], indent=2))


if __name__ == "__main__":
    main()
