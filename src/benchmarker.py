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

    def save_results(self, output_file="benchmark_results.csv"):
        df = pd.DataFrame(self.results)
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

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
    target = ["2d_Hurink_vdata_30_workers.fjs", "2a_Hurink_sdata_54_workers.fjs"]
    runner.run_benchmark(algorithms, k=K)
    
if __name__ == "__main__":
    main()