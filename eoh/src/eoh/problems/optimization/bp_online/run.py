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

    def get_valid_bin_indices(self,item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]


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
        for item in items:
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            if len(valid_bin_indices) == 0:
                continue
            priorities = alg.score(item, bins[valid_bin_indices])
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
            total_items += 1
            if bins[best_bin] <= 0.05 * capacity:
                tight_fit_count += 1

        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins, tight_fit_count, total_items


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


        # Score of heuristic function is negative of average number of bins used
        # across instances (as we want to minimize number of bins).

        return fitness



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

    def evaluate_with_signals(self, code_string):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                heuristic_module = types.ModuleType("heuristic_module_diag")
                exec(code_string, heuristic_module.__dict__)
                sys.modules[heuristic_module.__name__] = heuristic_module

                objectives = []
                runtimes_ms = []
                bins_opened_list = []
                leftover_means = []
                tight_fit_counts = 0
                total_items = 0
                failures = 0
                per_instance = []

                for name, dataset in self.instances.items():
                    for inst_id, instance in dataset.items():
                        try:
                            capacity = instance['capacity']
                            items = np.array(instance['items'])
                            bins = np.array([capacity for _ in range(instance['num_items'])])

                            start = time.perf_counter()
                            _, bins_packed, tight_fit, items_count = self.online_binpack_with_metrics(
                                items, bins, heuristic_module, capacity
                            )
                            elapsed = (time.perf_counter() - start) * 1000.0

                            bins_opened = (bins_packed != capacity).sum()
                            obj = (bins_opened - self.lb[name]) / self.lb[name]

                            objectives.append(obj)
                            runtimes_ms.append(elapsed)
                            bins_opened_list.append(bins_opened)

                            used_bins = bins_packed[bins_packed != capacity]
                            if len(used_bins) > 0:
                                leftover_means.append(float(np.mean(used_bins)))
                            else:
                                leftover_means.append(0.0)

                            tight_fit_counts += tight_fit
                            total_items += items_count

                            per_instance.append({
                                "dataset": name,
                                "instance": str(inst_id),
                                "objective": float(obj),
                                "bins_opened": int(bins_opened),
                                "leftover_mean": float(leftover_means[-1]),
                                "runtime_ms": float(elapsed),
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
                    },
                    "per_instance": per_instance_sorted,
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
                },
                "per_instance": [],
            }
