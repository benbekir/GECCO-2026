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
from src.algorithms.lahc import LAHCSolver
from src.algorithms.spea_lahc import HybridSPEALAHC
from src.algorithms.ml import MLSolver
from src.algorithms.ml import Strategy
from src.util.benchmark_parser import WorkerBenchmarkParser
from enum import Enum
import json

instances_dir = "instances/fjssp-w"
files = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]

parser=WorkerBenchmarkParser()

class Algorithms(Enum):
     SPEA2=1
     LAHC=2
     GOGETA=3
     ML=4
     HYBRID=5
     
TARGET_FILES = [
    # 100 60 90
    "0_BehnkeGeiger_60_workers.fjs",
    # 30 10 15 
    "1_Brandimarte_12_workers.fjs",
    # 20 5 7 
    "2a_Hurink_sdata_18_workers.fjs",
    # 15 15 22 
    "2a_Hurink_sdata_40_workers.fjs",
    # 15 10 15
    "2c_Hurink_rdata_28_workers.fjs",
    # 6 6 9
    "2b_Hurink_edata_1_workers.fjs"
]
def create_objective(algorithm_choice:Algorithms):

    def objective(trial: optuna.trial.Trial):
        solver: FJSSPAlgorithm = None
        if algorithm_choice==Algorithms.SPEA2:
            population=trial.suggest_int("population_size",150,400)
            archive=trial.suggest_int("archive_size",50,200)
            mutation_rate=trial.suggest_float("mutation_rate",0.01,0.2)
            mut_limit = trial.suggest_int("tracker_limit_mutation", 15, 60, step=5)
            nuke_limit = trial.suggest_int("tracker_limit_nuke", 60, 150, step=10)
            print(f"Chose population={population}, archive={archive}, mutation_rate={mutation_rate}, mut_limit={mut_limit}, nuke_limit={nuke_limit}.")
            solver=SPEA2Solver(pop_size=population,archive_size=archive,max_generations=150,mutation_rate=mutation_rate,mutation_limit=mut_limit,nuke_limit=nuke_limit)

        elif algorithm_choice==Algorithms.LAHC:
            L=trial.suggest_int("L",10,500)
            max_iters=trial.suggest_int("Max_iterations",5000,75000,log=True)
            print(f"Chose L={L}, max_iters={max_iters}.")
            solver=LAHCSolver(L=L,max_iters=max_iters)

        elif algorithm_choice==Algorithms.ML:
            strategy = trial.suggest_categorical("strategy", choices=(Strategy.PLUS, Strategy.COMMA))
            M = trial.suggest_int("M",10,200)
            L = trial.suggest_int("L",50,700)
            print(f"Chose Strategy={strategy}, M={M}, L={L}.")
            solver=MLSolver(strategy=strategy,M=M,L=L,max_generations=500)
        
        elif algorithm_choice==Algorithms.HYBRID:
            TOTAL_BUDGET = 100_000
    
            pop_size = trial.suggest_int("pop_size", 20, 100)
            max_generations = trial.suggest_int("max_generations", 20, 100)
            
            remaining_budget = TOTAL_BUDGET - pop_size
            possible_iters = remaining_budget // (max_generations * pop_size)
            lahc_iters = max(1, possible_iters - 1)
            
            archive_size = trial.suggest_int("archive_size", 10, 50)
            lahc_l = trial.suggest_int("lahc_l", 10, 100)
            solver = HybridSPEALAHC(pop_size, archive_size, max_generations, lahc_iters, lahc_l)
             
        results=[]
        for file in TARGET_FILES:
            print(f"Running {file}...")
            filepath = os.path.join(instances_dir, file)
            encoding = parser.parse_benchmark(filepath)
            best_candidate,_=solver.solve(encoding)
            results.append(best_candidate.makespan / encoding.n_operations())
        return sum(results) / len(results)
    return objective

if __name__ == "__main__":
    MY_CHOICE = Algorithms.HYBRID
    
    study = optuna.create_study(direction="minimize")
    func = create_objective(MY_CHOICE)
    study.optimize(func, n_trials=1)
    
    print(f"Best params for {MY_CHOICE.name}: {study.best_params}")

    output_data = {
        "algorithm": MY_CHOICE.name,
        "best_value": study.best_value,
        "best_params": study.best_params
    }
    if "strategy" in output_data["best_params"]:
        output_data["best_params"]["strategy"] = str(output_data["best_params"]["strategy"])

    filename = f"results/best_params_{MY_CHOICE.name.lower()}.json"

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Parameters successfully saved to {filename}")
                