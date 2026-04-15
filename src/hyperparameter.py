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
from enum import Enum
from src.algorithms.lahc import LAHCSolver
from src.algorithms.ga import GASolver
from src.algorithms.ga import Strategy
import json

instances_dir = "instances/fjssp-w"
files = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]

parser=WorkerBenchmarkParser()

class Algorithms(Enum):
     SPEA2=1
     LAHC=2
     GOGETA=3
     GA=4

     
TARGET_FILES = [
    "0_BehnkeGeiger_42_workers.fjs",
    "2c_Hurink_rdata_28_workers.fjs",
    "3_DPpaulli_15_workers.fjs"
]
def create_objective(algorithm_choice:Algorithms):
    def objective(trial):
        if algorithm_choice==Algorithms.SPEA2:
            population=trial.suggest_int("population_size",150,400)
            archive=trial.suggest_int("archive_size",50,200)
            mutation_rate=trial.suggest_float("mutation_rate",0.01,0.2)
            mut_limit = trial.suggest_int("tracker_limit_mutation", 15, 60, step=5)
            nuke_limit = trial.suggest_int("tracker_limit_nuke", 60, 150, step=10)
            solver=SPEA2Solver(pop_size=population,archive_size=archive,max_generations=150,mutation_rate=mutation_rate,mutation_limit=mut_limit,nuke_limit=nuke_limit)

        elif algorithm_choice==Algorithms.LAHC:
            L=trial.suggest_int("L",10,500)
            max_iters=trial.suggest_int("Max_iterations",5000,50000,log=True)
            solver=LAHCSolver(L=L,max_iters=max_iters)

        elif algorithm_choice==Algorithms.GA:
            strategy = trial.suggest_categorical("strategy",[Strategy.PLUS, Strategy.COMMA])
            M = trial.suggest_int("M",10,50)
            L = trial.suggest_int("L",50,200)
            solver= GASolver(strategy=strategy,M=M,L=L,max_generations=500)
             
        results=[]
        for file in TARGET_FILES:
                filepath = os.path.join(instances_dir, file)
                encoding = parser.parse_benchmark(filepath)
                best_candidate,_=solver.solve(encoding)
                results.append(best_candidate.makespan / encoding.n_operations())
        return sum(results) / len(results)
    return objective

if __name__ == "__main__":
    MY_CHOICE = Algorithms.GA
    
    study = optuna.create_study(direction="minimize")
    
    func = create_objective(MY_CHOICE)
    
    study.optimize(func, n_trials=50)
    
    print(f"Best params for {MY_CHOICE.name}: {study.best_params}")

    output_data = {
        "algorithm": MY_CHOICE.name,
        "best_value": study.best_value,
        "best_params": study.best_params
    }
    if "strategy" in output_data["best_params"]:
        output_data["best_params"]["strategy"] = str(output_data["best_params"]["strategy"])

    filename = f"best_params_{MY_CHOICE.name.lower()}.json"

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Parameters successfully saved to {filename}")
                