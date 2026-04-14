from pathlib import Path
import os
import sys
from typing import TYPE_CHECKING
# Support direct execution (python src/benchmarker.py) by ensuring repo root is on sys.path.
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if TYPE_CHECKING:
    from src.core.fjssp_algorithm import FJSSPAlgorithm
import optuna
from src.algorithms.spea import SPEA2Solver
from src.core.fjssp_algorithm import FJSSPAlgorithm
from src.util.benchmark_parser import WorkerBenchmarkParser
from src.algorithms.spea import SPEA2Solver


instances_dir = "instances/fjssp-w"
files = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]

parser=WorkerBenchmarkParser()

TARGET_FILES = [
    "0_BehnkeGeiger_42_workers.fjs",
    "2c_Hurink_rdata_28_workers.fjs",
    "3_DPpaulli_15_workers.fjs"
]
def run_benchmark(self, algorithms: dict[str, "FJSSPAlgorithm"], k: int):
        """
        algorithms: A dictionary where key is the name and value is the function 
                    that takes a filepath and returns (best_candidate, history)
        """
        parser = WorkerBenchmarkParser()
        progress = 0
        total = len(self.files)
        for filename in self.files:
            progress += 1
            print(f"Running instance {progress}/{total}...")

            filepath = os.path.join(self.instances_dir, filename)
            print(f"\n--- Benchmarking Instance: {filename} ---")
            
            for name, algorithm in algorithms.items():
                print(f"Running {name}...", end=" ", flush=True)
                
                avg_candidate_makespan = 0
                for _ in range(k):
                    encoding = parser.parse_benchmark(filepath)
                    best_candidate, _ = algorithm.solve(encoding)
                    avg_candidate_makespan += best_candidate.makespan / k

            return avg_candidate_makespan
        
def objective(trial):
     population=trial.suggest_int("population_size",150,400)
     archive=trial.suggest_int("archive_size",50,200)
     mutation_rate=trial.suggest_float("mutation_rate",0.01,0.2)
     results=[]
     solver=SPEA2Solver(pop_size=population,archive_size=archive,max_generations=100,mutation_rate=mutation_rate)
     for file in TARGET_FILES:
            filepath = os.path.join(instances_dir, file)
            encoding = parser.parse_benchmark(filepath)
            best_candidate,_=solver.solve(encoding)
            results.append(best_candidate.makespan / encoding.n_operations())
     return sum(results) / len(results)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)
print(f"Best params: {study.best_params}")
                