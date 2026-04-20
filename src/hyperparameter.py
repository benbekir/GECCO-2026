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

def save_best_callback(study, trial):
    """Callback to save results whenever a new best trial is found."""
    if study.best_trial.number == trial.number:
        filename = f"results/best_params_{study.user_attrs['algorithm_name'].lower()}.json"
        
        # Prepare data for JSON
        best_params = study.best_params.copy()
        if "strategy" in best_params:
            best_params["strategy"] = str(best_params["strategy"])

        output_data = {
            "best_value": study.best_value,
            "best_params": best_params
        }

        os.makedirs("results", exist_ok=True)
        with open(filename, "w") as f:
            json.dump(output_data, f, indent=4)
        print(f"--> New best found! Trial {trial.number} saved to {filename}")

def create_objective(algorithm_choice: Algorithms):
    parser = WorkerBenchmarkParser()
    instances_dir = "instances/fjssp-w"

    def objective(trial: optuna.trial.Trial):
        solver: FJSSPAlgorithm = None
        if algorithm_choice == Algorithms.SPEA2:
            pop = trial.suggest_int("population_size", 150, 400)
            arc = trial.suggest_int("archive_size", 50, 200)
            mut = trial.suggest_float("mutation_rate", 0.01, 0.2)
            mut_lim = trial.suggest_int("tracker_limit_mutation", 15, 60, step=5)
            nuke_lim = trial.suggest_int("tracker_limit_nuke", 60, 150, step=10)
            tabu_lim=trial.suggest_int("tracker_tabu_limit",20,80,step=5)
            tabu_duration=trial.suggest_int("tracker_tabu_duration",10,50,step=5)
            solver = SPEA2Solver(pop, arc, 150, mut, mut_lim, nuke_lim,tabu_lim,tabu_duration)

        elif algorithm_choice == Algorithms.LAHC:
            L = trial.suggest_int("L", 10, 500)
            max_iters = trial.suggest_int("Max_iterations", 5000, 75000, log=True)
            solver = LAHCSolver(L=L, max_iters=max_iters)

        elif algorithm_choice == Algorithms.ML:
            strat = trial.suggest_categorical("strategy", [Strategy.PLUS, Strategy.COMMA])
            M = trial.suggest_int("M", 10, 200)
            L = trial.suggest_int("L", 50, 700)
            solver = MLSolver(strategy=strat, M=M, L=L, max_generations=500)

        elif algorithm_choice == Algorithms.HYBRID:
            max_gens = trial.suggest_int("max_generations", 100, 150)
            arc_size = trial.suggest_int("archive_size", 10, 50)
            lahc_iters = trial.suggest_int("lahc_iters", 20, 75)
            lahc_l = trial.suggest_int("lahc_l", 2, lahc_iters)
            solver = HybridSPEALAHC(75, arc_size, max_gens, lahc_iters, lahc_l)

        results = []
        for file in TARGET_FILES:
            print(f"File: {file}")
            filepath = os.path.join(instances_dir, file)
            encoding = parser.parse_benchmark(filepath)
            best_candidate, _ = solver.solve(encoding)
            # Normalized makespan (relative to number of operations)
            results.append(best_candidate.makespan / encoding.n_operations())
        
        return sum(results) / len(results)

    return objective

if __name__ == "__main__":
    MY_CHOICE = Algorithms.SPEA2
    
    study = optuna.create_study(direction="minimize")
    study.set_user_attr("algorithm_name", MY_CHOICE.name)
    
    func = create_objective(MY_CHOICE)
    
    print(f"Starting Optuna for {MY_CHOICE.name}...")
    print("Limits: 50 trials OR 10 hours.")

    study.optimize(
        func, 
        n_trials=50, 
        timeout=36000, 
        callbacks=[save_best_callback]
    )
    
    print("\nOptimization Complete.")
    print(f"Best params: {study.best_params}")