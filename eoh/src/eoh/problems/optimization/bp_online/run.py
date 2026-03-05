import numpy as np
import importlib
import time
from .get_instance import GetData
from .prompts import GetPrompts
import types
import warnings
import sys

class BPONLINE():
    def __init__(self):
        getdate = GetData()
        self.instances, self.lb = getdate.get_instances()
        self.prompts = GetPrompts()
        self.instance_baselines = self._build_instance_baselines()

    def get_valid_bin_indices(self,item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]

    def _first_fit_bins(self, items: np.ndarray, capacity: float) -> int:
        remaining = []
        for item in items:
            placed = False
            for idx in range(len(remaining)):
                if remaining[idx] >= item:
                    remaining[idx] -= item
                    placed = True
                    break
            if not placed:
                remaining.append(capacity - item)
        return len(remaining)

    def _best_fit_bins(self, items: np.ndarray, capacity: float) -> int:
        remaining = []
        for item in items:
            best_idx = -1
            best_after = None
            for idx, space in enumerate(remaining):
                if space >= item:
                    space_after = space - item
                    if best_after is None or space_after < best_after:
                        best_after = space_after
                        best_idx = idx
            if best_idx >= 0:
                remaining[best_idx] -= item
            else:
                remaining.append(capacity - item)
        return len(remaining)

    def _build_instance_baselines(self):
        baselines = {}
        for name, dataset in self.instances.items():
            for inst_id, instance in dataset.items():
                capacity = instance["capacity"]
                items = np.array(instance["items"])
                ff_bins = self._first_fit_bins(items, capacity)
                bf_bins = self._best_fit_bins(items, capacity)
                baselines[(name, str(inst_id))] = {
                    "first_fit_bins": int(ff_bins),
                    "best_fit_bins": int(bf_bins),
                }
        return baselines


    def online_binpack(self,items: tuple, bins: np.ndarray, alg):
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        n = 1
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = alg.score(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
            n=n+1
            
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins

    def online_binpack_with_metrics(self, items: tuple, bins: np.ndarray, alg, capacity: float):
        """Online binpacking with diagnostics metrics."""
        packing = [[] for _ in bins]
        total_items = 0
        tight_fit_count = 0
        tie_count = 0
        score_std_sum = 0.0
        score_std_count = 0
        used_bins = np.zeros(len(bins), dtype=bool)
        opened_count = 0
        opened_history = []
        for item in items:
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            if len(valid_bin_indices) == 0:
                continue
            priorities = np.asarray(alg.score(item, bins[valid_bin_indices]))
            if priorities.ndim == 0:
                raise ValueError("score output must be a vector")
            if priorities.shape[0] != len(valid_bin_indices):
                raise ValueError("score output shape mismatch with valid bins")
            top_priority = np.max(priorities)
            tie_count += int(np.sum(priorities == top_priority) > 1)
            score_std_sum += float(np.std(priorities))
            score_std_count += 1
            best_bin = valid_bin_indices[np.argmax(priorities)]
            if not used_bins[best_bin]:
                used_bins[best_bin] = True
                opened_count += 1
            bins[best_bin] -= item
            packing[best_bin].append(item)
            total_items += 1
            if bins[best_bin] <= 0.05 * capacity:
                tight_fit_count += 1
            opened_history.append(opened_count)

        packing = [bin_items for bin_items in packing if bin_items]
        if opened_history:
            n_hist = len(opened_history)
            idx_early = max(0, n_hist // 3 - 1)
            idx_mid = max(0, (2 * n_hist) // 3 - 1)
            idx_late = n_hist - 1
            phase_opened = (
                int(opened_history[idx_early]),
                int(opened_history[idx_mid]),
                int(opened_history[idx_late]),
            )
        else:
            phase_opened = (0, 0, 0)

        metrics = {
            "tight_fit_count": int(tight_fit_count),
            "total_items": int(total_items),
            "tie_count": int(tie_count),
            "score_std_sum": float(score_std_sum),
            "score_std_count": int(score_std_count),
            "phase_opened": phase_opened,
        }
        return packing, bins, metrics


    # @funsearch.run
    def evaluateGreedy(self,alg) -> float:
        # algorithm_module = importlib.import_module("ael_alg")
        # alg = importlib.reload(algorithm_module)  
        """Evaluate heuristic function on a set of online binpacking instances."""
        # List storing number of bins used for each instance.
        #num_bins = []
        # Perform online binpacking for each instance.
        # for name in instances:
        #     #print(name)

        fitness_all = []
        for name, dataset in self.instances.items():
            num_bins_list = []
            for _, instance in dataset.items():

                capacity = instance['capacity']
                items = np.array(instance['items'])

                # items = items/capacity
                # capacity = 1.0

                # Create num_items bins so there will always be space for all items,
                # regardless of packing order. Array has shape (num_items,).
                bins = np.array([capacity for _ in range(instance['num_items'])])
                # Pack items into bins and return remaining capacity in bins_packed, which
                # has shape (num_items,).
                _, bins_packed = self.online_binpack(items, bins, alg)
                # If remaining capacity in a bin is equal to initial capacity, then it is
                # unused. Count number of used bins.
                num_bins = (bins_packed != capacity).sum()

                num_bins_list.append(-num_bins)

            # avg_num_bins = -self.evaluateGreedy(dataset, algorithm)
            avg_num_bins = -np.mean(np.array(num_bins_list))
            fitness = (avg_num_bins - self.lb[name]) / self.lb[name]
            fitness_all.append(fitness)


        # Score of heuristic function is negative of average number of bins used
        # across instances (as we want to minimize number of bins).

        if not fitness_all:
            return None
        return float(np.mean(np.array(fitness_all)))



    # def evaluate(self):
    #     try:

    #         for name, dataset in self.instances.items():
    #             # Parallelize the loop
    #             num_bins = Parallel(n_jobs=4,timeout=30)(delayed(self.evaluateGreedy)(instance) for _, instance in dataset.items())
    #             # avg_num_bins = -self.evaluateGreedy(dataset, algorithm)
    #             avg_num_bins = -np.mean(num_bins)
    #             excess = (avg_num_bins - self.lb[name]) / self.lb[name]
    #             #print(name)
    #             #print(f'\t Average number of bins: {avg_num_bins}')
    #             #print(f'\t Lower bound on optimum: {self.lb[name]}')
    #             #print(f'\t Excess: {100 * excess:.2f}%')        
    #         return excess
    #     except Exception as e:
    #         #print("Error:", str(e))  # Print the error message
    #         return None
        
    def evaluate(self, code_string):
        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")
                
                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.evaluateGreedy(heuristic_module)

                return fitness
        except Exception as e:
            #print("Error:", str(e))
            return None

    def evaluate_with_signals(self, code_string, max_instances=None, max_items=None, instance_keys=None):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                heuristic_module = types.ModuleType("heuristic_module_diag")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                eval_instances = []
                if instance_keys is not None:
                    for name, inst_id in instance_keys:
                        dataset = self.instances.get(name, {})
                        instance = dataset.get(inst_id)
                        if instance is not None:
                            eval_instances.append((name, str(inst_id), instance))
                else:
                    for name, dataset in self.instances.items():
                        for inst_id, instance in dataset.items():
                            eval_instances.append((name, str(inst_id), instance))
                    if max_instances is not None:
                        eval_instances = eval_instances[:max_instances]

                objectives = []
                runtimes_ms = []
                bins_opened_list = []
                leftover_means = []
                first_fit_gaps = []
                best_fit_gaps = []
                tie_rates = []
                score_stds = []
                phase_early_fracs = []
                phase_mid_fracs = []
                phase_late_fracs = []
                tight_fit_counts = 0
                total_items = 0
                failures = 0
                per_instance = []

                for name, inst_id, instance in eval_instances:
                    try:
                        capacity = instance['capacity']
                        items = np.array(instance['items'])
                        if max_items is not None:
                            items = items[:max_items]
                        bins = np.array([capacity for _ in range(len(items))])

                        start = time.perf_counter()
                        _, bins_packed, metrics = self.online_binpack_with_metrics(
                            items, bins, heuristic_module, capacity
                        )
                        elapsed = (time.perf_counter() - start) * 1000.0

                        bins_opened = int((bins_packed != capacity).sum())
                        obj = (bins_opened - self.lb[name]) / self.lb[name]

                        if max_items is None:
                            baseline = self.instance_baselines.get((name, inst_id), {})
                            ff_bins = int(baseline.get("first_fit_bins", max(1, bins_opened)))
                            bf_bins = int(baseline.get("best_fit_bins", max(1, bins_opened)))
                        else:
                            ff_bins = max(1, self._first_fit_bins(items, capacity))
                            bf_bins = max(1, self._best_fit_bins(items, capacity))

                        ff_gap = (bins_opened - ff_bins) / max(1, ff_bins)
                        bf_gap = (bins_opened - bf_bins) / max(1, bf_bins)

                        tie_rate = metrics["tie_count"] / max(1, metrics["total_items"])
                        score_std_mean = metrics["score_std_sum"] / max(1, metrics["score_std_count"])
                        phase_early, phase_mid, phase_late = metrics["phase_opened"]
                        denom_bins = max(1, bins_opened)
                        phase_early_frac = phase_early / denom_bins
                        phase_mid_frac = max(0.0, phase_mid - phase_early) / denom_bins
                        phase_late_frac = max(0.0, phase_late - phase_mid) / denom_bins

                        objectives.append(obj)
                        runtimes_ms.append(elapsed)
                        bins_opened_list.append(bins_opened)
                        first_fit_gaps.append(ff_gap)
                        best_fit_gaps.append(bf_gap)
                        tie_rates.append(tie_rate)
                        score_stds.append(score_std_mean)
                        phase_early_fracs.append(phase_early_frac)
                        phase_mid_fracs.append(phase_mid_frac)
                        phase_late_fracs.append(phase_late_frac)

                        used_bins = bins_packed[bins_packed != capacity]
                        if len(used_bins) > 0:
                            leftover_means.append(float(np.mean(used_bins)))
                        else:
                            leftover_means.append(0.0)

                        tight_fit_counts += metrics["tight_fit_count"]
                        total_items += metrics["total_items"]

                        per_instance.append({
                            "dataset": name,
                            "instance": inst_id,
                            "objective": float(obj),
                            "bins_opened": bins_opened,
                            "leftover_mean": float(leftover_means[-1]),
                            "runtime_ms": float(elapsed),
                            "first_fit_gap": float(ff_gap),
                            "best_fit_gap": float(bf_gap),
                            "tie_rate": float(tie_rate),
                            "score_std_mean": float(score_std_mean),
                            "phase_opened_frac_early": float(phase_early_frac),
                            "phase_opened_frac_mid": float(phase_mid_frac),
                            "phase_opened_frac_late": float(phase_late_frac),
                        })
                    except Exception:
                        failures += 1

                if len(objectives) == 0:
                    objective = float("inf")
                    mean_obj = float("inf")
                    std_obj = float("inf")
                    worst_obj = float("inf")
                else:
                    objective = float(np.mean(objectives))
                    mean_obj = float(np.mean(objectives))
                    std_obj = float(np.std(objectives))
                    worst_obj = float(np.max(objectives))

                fail_rate = failures / max(1, (failures + len(objectives)))
                mean_runtime_ms = float(np.mean(runtimes_ms)) if runtimes_ms else 0.0
                bins_opened_mean = float(np.mean(bins_opened_list)) if bins_opened_list else 0.0
                leftover_mean = float(np.mean(leftover_means)) if leftover_means else 0.0
                leftover_std = float(np.std(leftover_means)) if leftover_means else 0.0
                tight_fit_rate = float(tight_fit_counts / max(1, total_items))
                mean_first_fit_gap = float(np.mean(first_fit_gaps)) if first_fit_gaps else 0.0
                mean_best_fit_gap = float(np.mean(best_fit_gaps)) if best_fit_gaps else 0.0
                tie_rate = float(np.mean(tie_rates)) if tie_rates else 0.0
                score_std_mean = float(np.mean(score_stds)) if score_stds else 0.0
                phase_early_frac = float(np.mean(phase_early_fracs)) if phase_early_fracs else 0.0
                phase_mid_frac = float(np.mean(phase_mid_fracs)) if phase_mid_fracs else 0.0
                phase_late_frac = float(np.mean(phase_late_fracs)) if phase_late_fracs else 0.0

                per_instance_sorted = sorted(per_instance, key=lambda x: x["objective"], reverse=True)[:5]

                return {
                    "objective": objective,
                    "signals": {
                        "mean_objective": mean_obj,
                        "std_objective": std_obj,
                        "worst_objective": worst_obj,
                        "fail_rate": float(fail_rate),
                        "mean_runtime_ms": mean_runtime_ms,
                        "bins_opened_mean": bins_opened_mean,
                        "leftover_mean": leftover_mean,
                        "leftover_std": leftover_std,
                        "tight_fit_rate": tight_fit_rate,
                        "mean_first_fit_gap": mean_first_fit_gap,
                        "mean_best_fit_gap": mean_best_fit_gap,
                        "tie_rate": tie_rate,
                        "score_std_mean": score_std_mean,
                        "phase_opened_frac_early": phase_early_frac,
                        "phase_opened_frac_mid": phase_mid_frac,
                        "phase_opened_frac_late": phase_late_frac,
                    },
                    "per_instance": per_instance_sorted,
                    "per_instance_all": per_instance,
                    "evaluated_instance_count": len(per_instance),
                }
        except Exception:
            return {
                "objective": float("inf"),
                "signals": {
                    "mean_objective": float("inf"),
                    "std_objective": float("inf"),
                    "worst_objective": float("inf"),
                    "fail_rate": 1.0,
                    "mean_runtime_ms": 0.0,
                    "bins_opened_mean": 0.0,
                    "leftover_mean": 0.0,
                    "leftover_std": 0.0,
                    "tight_fit_rate": 0.0,
                    "mean_first_fit_gap": 0.0,
                    "mean_best_fit_gap": 0.0,
                    "tie_rate": 0.0,
                    "score_std_mean": 0.0,
                    "phase_opened_frac_early": 0.0,
                    "phase_opened_frac_mid": 0.0,
                    "phase_opened_frac_late": 0.0,
                },
                "per_instance": [],
                "per_instance_all": [],
                "evaluated_instance_count": 0,
            }
