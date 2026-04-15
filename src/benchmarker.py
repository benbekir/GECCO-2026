import os
import time
import sys
import pandas as pd

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

    def run_benchmark(self, algorithms: dict[str, "FJSSPAlgorithm"], k: int,filter_list: list[str] = None):
        """
        algorithms: A dictionary where key is the name and value is the function 
                    that takes a filepath and returns (best_candidate, history)
        """
        parser = WorkerBenchmarkParser()
        progress = 0
        total = len(self.files)

        for name, algorithm in algorithms.items():
            print(f"Running {name}...", end=" ", flush=True)

            for filename in self.files:
                progress += 1
                filepath = os.path.join(self.instances_dir, filename)
                print(f"\nInstance {filename} ({progress}/{total})")

                start_time = time.time()
                avg_candidate_makespan = 0
                avg_workload_balance = 0
                for _ in range(k):
                    encoding = parser.parse_benchmark(filepath)
                    best_candidate, _ = algorithm.solve(encoding)
                    _, machines, workers = best_candidate.get_sequences()
                    c_workload_balance = workload_balance(machines, workers, encoding.durations())
                    avg_candidate_makespan += best_candidate.makespan / k
                    avg_workload_balance += c_workload_balance / k
                avg_duration = (time.time() - start_time) / k
                
                self.results.append({
                    "Instance": filename,
                    "Average Makespan": avg_candidate_makespan,
                    "Average Workload Balance": avg_workload_balance,
                    "Average Runtime (s)": round(avg_duration, 2)
                })
                print("Done.")

            self.save_results(output_file=f"results/{name}.csv")
            pass

    def get_summary(self):
        return pd.DataFrame(self.results)

    def save_results(self, output_file="benchmark_results.csv"):
        df = self.get_summary()
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

def main() -> None:
    from src.algorithms.ga import GASolver, Strategy
    from src.algorithms.lahc import LAHCSolver
    from src.algorithms.greedy import GreedyFJSSPWSolver
    from src.algorithms.spea import SPEA2Solver

    K = 10

    runner = BenchmarkRunner("instances/fjssp-w")

    algorithms: dict[str, FJSSPAlgorithm] = {
        "LAHC": LAHCSolver(L=50, max_iters=10_000),
        "GA_PLUS": GASolver(Strategy=Strategy.PLUS, M=10, L=50, max_generations=100),
        "GREEDY": GreedyFJSSPWSolver(),
        "SPEA-II": SPEA2Solver(pop_size=315,archive_size=128,max_generations=500,mutation_rate=0.02828977853657342,mutation_limit=55,nuke_limit=80)
    }

    target = ["2d_Hurink_vdata_30_workers.fjs", "2a_Hurink_sdata_54_workers.fjs", "3_DPpaulli_15_workers.fjs"]
    runner.run_benchmark(algorithms, k=K, filter_list=target)
    
    summary_df = runner.get_summary()
    print("\nFinal Comparison:")
    print(summary_df.pivot(index="Instance", columns="Algorithm", values="Average Makespan"))
    
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