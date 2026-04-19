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
        parser = WorkerBenchmarkParser()
        created_result_files = list[str]()
        created_history_files = list[str]()

        os.makedirs("results", exist_ok=True)

        for name, algorithm in algorithms.items():
            print(f"Starting Algorithm: {name}")
            
            output_path = f"results/{name}.csv"
            hist_path = f"results/{name}_history.json"
            
            created_result_files.append(output_path)
            created_history_files.append(hist_path)

            for filename in self.files:
                if filter and filename not in filter:
                    continue

                filepath = os.path.join(self.instances_dir, filename)
                print(f"Running File: {filename}...", flush=True)
                
                for run_idx in range(k):
                    print(f"\tRun: {run_idx + 1}/{k}")
                    start_time = time.time()
                    encoding = parser.parse_benchmark(filepath)
                    
                    # Computationally expensive part
                    best_candidate, history = algorithm.solve(encoding)
                    
                    _, machines, workers = best_candidate.get_sequences()
                    c_workload_balance = workload_balance(machines, workers, encoding.durations())
                    runtime = time.time() - start_time

                    # 1. Update history JSON immediately
                    if run_idx == 0:
                        self._update_history_json(hist_path, filename, history)

                    # 2. Create a single-row result dictionary
                    run_data = {
                        "Algorithm": name,
                        "Instance": filename,
                        "Run_ID": run_idx + 1,
                        "Makespan": best_candidate.makespan,
                        "Workload Balance": round(c_workload_balance, 2),
                        "Runtime": round(runtime, 2)
                    }

                    df = pd.DataFrame([run_data]) # Note: list wrapping the dict
                    file_exists = os.path.isfile(output_path)
                    df.to_csv(output_path, mode='a', index=False, header=not file_exists)
                    
                    print(f"\tRun {run_idx + 1} saved (Makespan: {best_candidate.makespan})")

                print(f"  Completed all runs for {filename}.")

        return created_result_files, created_history_files

    def _update_history_json(self, path, instance_name, history_data):
        """Helper to update the JSON history file incrementally."""
        data = {}
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                data = {}
                
        data[instance_name] = history_data
        
        with open(path, 'w') as f:
            json.dump(data, f)

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

    def plot_bars(self, file_paths: list[str]):
        """
        Reads results from CSV files, calculates averages, and plots a grouped bar chart.
        """
        dfs = []
        for fp in file_paths:
            if os.path.exists(fp):
                dfs.append(pd.read_csv(fp))
        
        if not dfs:
            print("No data found to plot bars.")
            return
            
        df = pd.concat(dfs, ignore_index=True)

        # 2. Group by Instance and Algorithm to get the mean makespan
        # This handles the case where K > 1 automatically
        summary_df = df.groupby(['Instance', 'Algorithm'])['Makespan'].mean().reset_index()

        # 3. Pivot the data so instances are rows and algorithms are columns
        pivot_df = summary_df.pivot(index='Instance', columns='Algorithm', values='Makespan')

        # 4. Plotting
        ax = pivot_df.plot(kind='bar', figsize=(12, 7), width=0.8)

        plt.title('Algorithm Comparison: Average Makespan per Instance', fontsize=14)
        plt.xlabel('Instance', fontsize=12)
        plt.ylabel('Mean Makespan (Lower is Better)', fontsize=12)
        
        # Improve layout
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        
        # Save and show
        output_name = 'results/overall_comparison_bars.png'
        plt.savefig(output_name, dpi=300)
        print(f"Bar chart saved to {output_name}")
        plt.show()

def main() -> None:
    from src.algorithms.ml import MLSolver, Strategy
    from src.algorithms.lahc import LAHCSolver
    from src.algorithms.greedy import GreedyFJSSPWSolver
    from src.algorithms.spea import SPEA2Solver
    from src.algorithms.spea_lahc import HybridSPEALAHC

    K = 10
    runner = BenchmarkRunner("instances/fjssp-w")
    algorithms: dict[str, FJSSPAlgorithm] = {
        #"LAHC": LAHCSolver(L=170, max_iters=54_755),
        #"ML": MLSolver(Strategy=Strategy.PLUS, M=152, L=304, max_generations=500),
        #"GREEDY": GreedyFJSSPWSolver(),
        "SPEA-II": SPEA2Solver(pop_size=315,archive_size=128,max_generations=500,mutation_rate=0.02828977853657342,mutation_limit=55,nuke_limit=80),
        #"HYBRID": HybridSPEALAHC(pop_size=30, max_generations=100, lahc_iters=500, archive_size=20, lahc_l=50)
    }
    target = [
        "2a_Hurink_sdata_1_workers.fjs",
        "2b_Hurink_edata_1_workers.fjs",
        "3_DPpaulli_1_workers.fjs",
        "4_ChambersBarnes_10_workers.fjs",
        "5_Kacem_3_workers.fjs",
        "6_Fattahi_14_workers.fjs"]
    res_files, hist_files = runner.run_benchmark(algorithms, k=K, filter=target)
    #res_files = ["results/other.csv", "results/SPEA-II.csv", "results/GREEDY.csv", "results/LAHC.csv", "results/HYBRID.csv"]
    #runner.perform_weighted_ranking(res_files)
    #hist_files = ["results/ML_history.json", "results/SPEA-II_history.json", "results/LAHC_history.json"]
    #runner.plot_convergence(hist_files, "1_Brandimarte_7_workers.fjs")
    #runner.plot_bars(res_files)
    
if __name__ == "__main__":
    main()