import os
import time
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
        self.results = []
        self.convergence_histories = {}

    def run_benchmark(self, algorithms: dict[str, "FJSSPAlgorithm"], k: int, filter_list: list[str] = []):
        parser = WorkerBenchmarkParser()
        total = len(filter_list) if filter_list else len(self.files)

        for name, algorithm in algorithms.items():
            print(f"Running {name}...", end=" ", flush=True)
            self.results = [] 
            progress = 0

            for filename in self.files:
                if filter_list and filename not in filter_list:
                    continue

                progress += 1
                filepath = os.path.join(self.instances_dir, filename)
                print(f"\nInstance {filename} ({progress}/{total})")
                raw_makespan=[]
                for run_idx in range(k):
                    start_time = time.time()
                    
                    encoding = parser.parse_benchmark(filepath)
                    best_candidate, history = algorithm.solve(encoding)
                    raw_makespan.append(best_candidate.makespan)
                    # Extract metrics
                    _, machines, workers = best_candidate.get_sequences()
                    c_workload_balance = workload_balance(machines, workers, encoding.durations())
                    duration = time.time() - start_time
                    avg_makespan = sum(raw_makespan) / k
                    # Store each run as its own row
                    self.results.append({
                        "Algorithm": name,
                        "Instance": filename,
                        "Run_ID": run_idx + 1,
                        "Makespan": best_candidate.makespan,
                        "Workload Balance": c_workload_balance,
                        "Raw Makespans":raw_makespan,
                        "Average Makespan": avg_makespan,
                        "Runtime (s)": round(duration, 4)
                    })
                    if run_idx == 0:
                        self.convergence_histories[name] = history
                print(f"Done {k} runs.")

            self.save_results(output_file=f"results/{name}.csv")

    def get_summary(self):
        return pd.DataFrame(self.results)

    def save_results(self, output_file="benchmark_results.csv"):
        df = self.get_summary()
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

def perform_weighted_ranking(summary_df):
    instances = summary_df['Instance'].unique()
    algorithms = summary_df['Algorithm'].unique()
    
    global_ps_scores = {algo: [] for algo in algorithms}
    
    print("\n--- Pairwise Probability of Superiority (PS) ---")
    
    for inst in instances:
        print(f"\nInstance: {inst}")
        
        for algo_a in algorithms:
            algo_a_scores = []
            data_a = summary_df[(summary_df['Instance'] == inst) & 
                               (summary_df['Algorithm'] == algo_a)]['Raw Makespans'].values[0]
            n1 = len(data_a)
            
            for algo_b in algorithms:
                if algo_a == algo_b:
                    continue
                
                data_b = summary_df[(summary_df['Instance'] == inst) & 
                                   (summary_df['Algorithm'] == algo_b)]['Raw Makespans'].values[0]
                n2 = len(data_b)
                
                stat, p_value = mannwhitneyu(data_a, data_b, alternative='two-sided')
                
                # Probability that a random run of A is better than a random run of B.
                ps = ( (n1 * n2) - stat ) / (n1 * n2)
                algo_a_scores.append(ps)
                global_ps_scores[algo_a].append(ps)
            
            avg_inst_ps = np.mean(algo_a_scores) if algo_a_scores else 0.5
            print(f"  {algo_a:8} | Avg PS vs Others: {avg_inst_ps:.3f}")

    leaderboard = []
    for algo, scores in global_ps_scores.items():
        leaderboard.append((algo, np.mean(scores)))
    
    leaderboard.sort(key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*40)
    print(f"{'FINAL ALGORITHM RANKING':^40}")
    print("="*40)
    for rank, (name, score) in enumerate(leaderboard, 1):
        print(f"{rank}. {name:10} | Overall Superiority: {score:.4f}")
    print("="*40)

def plot_convergence(histories, instance_name):
    plt.figure(figsize=(12, 6))
    
    for name, history in histories.items():
        if not history:
            continue

        if isinstance(history[0], (list, tuple)):
            iters = [h[0] for h in history]
            values = [h[1] for h in history]
        else:
            values = history
            iters = list(range(len(history)))

        max_iter = max(iters) if max(iters) > 0 else 1
        progress = [(i / max_iter) * 100 for i in iters]
        
        plt.plot(progress, values, label=name, linewidth=2)

    plt.title(f"Normalized Convergence: {instance_name}", fontsize=14)
    plt.xlabel("Search Progress (%)", fontsize=12)
    plt.ylabel("Makespan", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'convergence_{instance_name}.png', dpi=300)
    plt.show()

def main() -> None:
    from src.algorithms.ga import GASolver, Strategy
    from src.algorithms.lahc import LAHCSolver
    from src.algorithms.greedy import GreedyFJSSPWSolver
    from src.algorithms.spea import SPEA2Solver

    K = 1
    runner = BenchmarkRunner("instances/fjssp-w")
    algorithms: dict[str, FJSSPAlgorithm] = {
        "LAHC": LAHCSolver(L=50, max_iters=10_000),
        "GA_PLUS": GASolver(Strategy=Strategy.PLUS, M=10, L=50, max_generations=100),
        #"GREEDY": GreedyFJSSPWSolver(),
        "SPEA-II": SPEA2Solver(pop_size=315,archive_size=128,max_generations=500,mutation_rate=0.02828977853657342,mutation_limit=55,nuke_limit=80)
    }

    target = ["3_DPpaulli_15_workers.fjs"]
    runner.run_benchmark(algorithms, k=K, filter_list=target)
    
    summary_df = runner.get_summary()
    print("\nFinal Comparison:")
    print(summary_df.pivot(index="Instance", columns="Algorithm", values="Average Makespan"))
    perform_weighted_ranking(summary_df)
    runner.save_results()
    plot_convergence(runner.convergence_histories,target[0])
    
if __name__ == "__main__":
    main()