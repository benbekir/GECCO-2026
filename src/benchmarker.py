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

from src.util.benchmark_parser import WorkerBenchmarkParser
from src.util.evaluation import workload_balance


class BenchmarkRunner:
    def __init__(self, instances_dir: str):
        self.instances_dir = instances_dir
        self.files = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]
        self.results = []

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

                for run_idx in range(k):
                    start_time = time.time()
                    
                    encoding = parser.parse_benchmark(filepath)
                    best_candidate, _ = algorithm.solve(encoding)
                    
                    # Extract metrics
                    _, machines, workers = best_candidate.get_sequences()
                    c_workload_balance = workload_balance(machines, workers, encoding.durations())
                    duration = time.time() - start_time
                    
                    # Store each run as its own row
                    self.results.append({
                        "Algorithm": name,
                        "Instance": filename,
                        "Run_ID": run_idx + 1,
                        "Makespan": best_candidate.makespan,
                        "Workload Balance": c_workload_balance,
                        "Runtime (s)": round(duration, 4)
                    })
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

def main() -> None:
    from src.algorithms.ga import GASolver, Strategy
    from src.algorithms.lahc import LAHCSolver
    from src.algorithms.greedy import GreedyFJSSPWSolver
    #from src.algorithms.spea import SPEA2Solver

    K = 10
    runner = BenchmarkRunner("instances/fjssp-w")
    algorithms: dict[str, FJSSPAlgorithm] = {
        "LAHC": LAHCSolver(L=50, max_iters=10_000),
        "GA_PLUS": GASolver(Strategy=Strategy.PLUS, M=10, L=50, max_generations=100),
        "GREEDY": GreedyFJSSPWSolver(),
        #"SPEA-II": SPEA2Solver(pop_size=315,archive_size=128,max_generations=500,mutation_rate=0.02828977853657342,mutation_limit=55,nuke_limit=80)
    }

    target = ["3_DPpaulli_15_workers.fjs"]
    runner.run_benchmark(algorithms, k=K, filter_list=target)
    
    summary_df = runner.get_summary()
    print("\nFinal Comparison:")
    print(summary_df.pivot(index="Instance", columns="Algorithm", values="Average Makespan"))
    perform_weighted_ranking(summary_df)
    runner.save_results()

    import matplotlib.pyplot as plt

    pivot_df = summary_df.pivot(index='Instance', columns='Algorithm', values='Average Makespan')
    ax = pivot_df.plot(kind='bar', figsize=(10, 6), width=0.8)

    plt.title(f'Comparison of Algorithm Performance ({K} runs)', fontsize=14)
    plt.xlabel('Instance Name', fontsize=12)
    plt.ylabel('Average Makespan', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Algorithm')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('algorithm_comparison.png', dpi=300)
    plt.show()
    
if __name__ == "__main__":
    main()