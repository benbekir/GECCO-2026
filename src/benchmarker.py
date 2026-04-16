import os
import time
import json
import sys
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
# NOTE: this block is only here for convenience, so that it is possible to run this file directly
# its possible to remove this block and just run this file using "python -m src.benchmarker"
from pathlib import Path
from typing import TYPE_CHECKING
# Support direct execution (python src/benchmarker.py) by ensuring repo root is on sys.path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if TYPE_CHECKING:
    from src.core.fjssp_algorithm import FJSSPAlgorithm
import matplotlib.pyplot as plt
from src.util.benchmark_parser import WorkerBenchmarkParser
from src.util.evaluation import workload_balance

class BenchmarkRunner:
    def __init__(self, instances_dir: str):
        self.instances_dir = instances_dir
        self.files = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]

    def run_benchmark(self, algorithms: dict[str, "FJSSPAlgorithm"], k: int, filter: list[str] = []) -> tuple[list[str], list[str]]:
        """
        Runs benchmarks and returns a list of paths to the generated CSV files.
        """
        parser = WorkerBenchmarkParser()
        created_result_files = []
        created_history_files = []

        for name, algorithm in algorithms.items():
            print(f"Running {name}...")
            algo_results = []
            algo_histories = {}
            
            for filename in self.files:
                if filter and filename not in filter:
                    continue

                filepath = os.path.join(self.instances_dir, filename)
                
                for run_idx in range(k):
                    start_time = time.time()
                    encoding = parser.parse_benchmark(filepath)
                    best_candidate, history = algorithm.solve(encoding)
                    
                    _, machines, workers = best_candidate.get_sequences()
                    c_workload_balance = workload_balance(machines, workers, encoding.durations())
                    runtime = time.time() - start_time

                    if run_idx == 0:
                        algo_histories[filename] = history
                    algo_results.append({
                        "Algorithm": name,
                        "Instance": filename,
                        "Run_ID": run_idx + 1,
                        "Makespan": best_candidate.makespan,
                        "Workload Balance": round(c_workload_balance, 2),
                        "Runtime": round(runtime, 2)
                    })

            output_path = f"results/{name}.csv"
            df = pd.DataFrame(algo_results)
            df.to_csv(output_path, index=False)
            created_result_files.append(output_path)
            
            hist_path = f"results/{name}_history.json"
            with open(hist_path, 'w') as f:
                json.dump(algo_histories, f)
            created_history_files.append(hist_path)

        return created_result_files, created_history_files

    def perform_weighted_ranking(self, file_paths: list[str]):
        """
        Reads results from provided file paths and performs ranking.
        """
        # Load all results
        dfs = [pd.read_csv(fp) for fp in file_paths]
        df = pd.concat(dfs, ignore_index=True)

        instances = df['Instance'].unique()
        algorithms = df['Algorithm'].unique()
        global_ps_scores = {algo: [] for algo in algorithms}

        for inst in instances:
            # Filter data for this specific instance
            inst_df = df[df['Instance'] == inst]
            
            for algo_a in algorithms:
                # Extract all makespans for Algo A as a list/series
                data_a = inst_df[inst_df['Algorithm'] == algo_a]['Makespan'].values
                if len(data_a) == 0: continue
                
                for algo_b in algorithms:
                    if algo_a == algo_b: continue
                    
                    data_b = inst_df[inst_df['Algorithm'] == algo_b]['Makespan'].values
                    if len(data_b) == 0: continue
                    
                    # Statistical comparison
                    stat, _ = mannwhitneyu(data_a, data_b, alternative='two-sided')
                    
                    # Probability of Superiority (PS)
                    n1, n2 = len(data_a), len(data_b)
                    ps = ((n1 * n2) - stat) / (n1 * n2)
                    global_ps_scores[algo_a].append(ps)

        # Final Leaderboard Logic
        leaderboard = sorted([(algo, np.mean(scores)) for algo, scores in global_ps_scores.items() if scores], 
                             key=lambda x: x[1], reverse=True)

        for rank, (name, score) in enumerate(leaderboard, 1):
            print(f"{rank}. {name:15} | Global PS: {score:.4f}")

    def plot_convergence(self, history_files: list[str], instance_name: str):
        plt.figure(figsize=(12, 6))
        
        for file_path in history_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            algo_name = os.path.basename(file_path).replace("_history.json", "")
            # we cant plot greedy algorithm converge
            if algo_name == "GREEDY":
                continue
            
            if instance_name in data:
                history = data[instance_name]
                
                # Handle both list of values or list of [iter, value] pairs
                if isinstance(history[0], list):
                    iters = [h[0] for h in history]
                    values = [h[1] for h in history]
                else:
                    values = history
                    iters = list(range(len(history)))

                # Normalize x-axis to 0-100% progress
                max_i = max(iters) if iters else 1
                progress = [(i / max_i) * 100 for i in iters]
                
                plt.plot(progress, values, label=algo_name, linewidth=2)

        plt.title(f"Convergence Comparison: {instance_name}", fontsize=14)
        plt.xlabel("Search Progress (%)", fontsize=12)
        plt.ylabel("Makespan", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f"results/plot_{instance_name}.png")
        plt.show()

def main() -> None:
    from src.algorithms.ga import GASolver, Strategy
    from src.algorithms.lahc import LAHCSolver
    from src.algorithms.greedy import GreedyFJSSPWSolver
    from src.algorithms.spea import SPEA2Solver

    K = 10
    runner = BenchmarkRunner("instances/fjssp-w")
    algorithms: dict[str, FJSSPAlgorithm] = {
        "LAHC": LAHCSolver(L=170, max_iters=54_755),
        "GA_PLUS": GASolver(Strategy=Strategy.PLUS, M=10, L=50, max_generations=100),
        "GREEDY": GreedyFJSSPWSolver(),
        "SPEA-II": SPEA2Solver(pop_size=315,archive_size=128,max_generations=500,mutation_rate=0.02828977853657342,mutation_limit=55,nuke_limit=80)
    }

    target = ["2b_Hurink_edata_1_workers.fjs"]
    res_files, hist_files = runner.run_benchmark(algorithms, k=K, filter=target)
    runner.perform_weighted_ranking(res_files)
    runner.plot_convergence(hist_files, "2b_Hurink_edata_1_workers.fjs")
    
if __name__ == "__main__":
    main()