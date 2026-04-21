import os
import time
import json
import sys
import pandas as pd
import numpy as np
import argparse
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
from src.util.evaluation import translate

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

            for filename in self.files:
                if filter and filename not in filter:
                    continue
                
                output_path = f"results/{name}_{filename}.json"
                hist_path = f"results/{name}_{filename}_history.json"
                
                created_result_files.append(output_path)
                created_history_files.append(hist_path)

                filepath = os.path.join(self.instances_dir, filename)
                print(f"Running File: {filename}...", flush=True)
                
                instance_results = []
                
                for run_idx in range(k):
                    print(f"\tRun: {run_idx + 1}/{k}")
                    start_time = time.time()
                    encoding = parser.parse_benchmark(filepath)
                    
                    best_candidate, history = algorithm.solve(encoding)

                    sequence, machines, workers = best_candidate.get_sequences()
                    start_times, m_fixed, w_fixed = translate(sequence, machines, workers, encoding.durations())
                    
                    runtime = time.time() - start_time

                    if run_idx == 0:
                        self._update_history_json(hist_path, filename, history)

                    run_data = {
                        "Algorithm": name,
                        "Instance": filename,
                        "Run_ID": run_idx + 1,
                        "Makespan": float(best_candidate.makespan),
                        "Balance": round(best_candidate.get_balance(), 2),
                        "Evaluations": algorithm.get_evaluations(),
                        "Runtime": round(runtime, 2),
                        "start_times": [int(t) for t in start_times],
                        "machine_assignments": [int(m) for m in m_fixed],
                        "worker_assignments": [int(w) for w in w_fixed]
                    }
                    
                    instance_results.append(run_data)
                    
                    with open(output_path, 'w') as f:
                        json.dump(instance_results, f, indent=4)
                    
                    print(f"\tRun {run_idx + 1} saved (Makespan: {best_candidate.makespan})")

                print(f"Completed all runs for {filename}.")

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
    
    def merge_results(self, algorithm_name: str):
        """
        Merges individual result JSONs into a MASTER list,
        and individual history JSONs into a MASTER dictionary.
        """
        merged_data = []
        merged_history = {}
        
        search_pattern = f"{algorithm_name}_"
        master_results_name = f"{algorithm_name}.json"
        master_history_name = f"{algorithm_name}_history.json"
        
        all_files = os.listdir("results")
        result_files = [f for f in all_files if f.startswith(search_pattern) and f.endswith(".json") 
                        and "_history" not in f and f != master_results_name]
        history_files = [f for f in all_files if f.startswith(search_pattern) and f.endswith("_history.json") 
                         and f != master_history_name]

        print(f"Merging {len(result_files)} result files...")
        for filename in result_files:
            path = os.path.join("results", filename)
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        merged_data.extend(data)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse result file {filename}")

        print(f"Merging {len(history_files)} history files...")
        for filename in history_files:
            path = os.path.join("results", filename)
            with open(path, 'r') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        # This merges the dictionaries. If keys (instance names) 
                        # are the same, the last one loaded wins.
                        merged_history.update(data)
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse history file {filename}")

        # 4. Save both Master files
        with open(os.path.join("results", master_results_name), 'w') as f:
            json.dump(merged_data, f, indent=4)
            
        with open(os.path.join("results", master_history_name), 'w') as f:
            json.dump(merged_history, f, indent=4)
        
        print(f"Done! Created {master_results_name} and {master_history_name}")

    def perform_weighted_ranking(self, file_paths: list[str]):
        """
        Reads results from provided file paths and performs ranking.
        """
        df = self._read_json(file_paths)

        instances = df['Instance'].unique()
        algorithms = df['Algorithm'].unique()
        global_ps_scores = {algo: [] for algo in algorithms}

        for inst in instances:
            inst_df = df[df['Instance'] == inst]
            
            for algo_a in algorithms:
                data_a = inst_df[inst_df['Algorithm'] == algo_a]['Makespan'].values
                if len(data_a) == 0: continue
                
                for algo_b in algorithms:
                    if algo_a == algo_b: continue
                    
                    data_b = inst_df[inst_df['Algorithm'] == algo_b]['Makespan'].values
                    if len(data_b) == 0: continue
                    
                    stat, _ = mannwhitneyu(data_a, data_b, alternative='two-sided')
                    n1, n2 = len(data_a), len(data_b)
                    ps = ((n1 * n2) - stat) / (n1 * n2)
                    global_ps_scores[algo_a].append(ps)

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
            if algo_name == "GREEDY":
                continue
            
            if instance_name in data:
                history = data[instance_name]
                
                if isinstance(history[0], list):
                    iters = [h[0] for h in history]
                    values = [h[1] for h in history]
                else:
                    values = history
                    iters = list(range(len(history)))

                max_i = max(iters) if iters else 1
                progress = [(i / max_i) * 100 for i in iters]
                
                plt.plot(progress, values, label=algo_name, linewidth=2)

        plt.title(f"Convergence Comparison: {instance_name}", fontsize=14)
        plt.xlabel("Search Progress (%)", fontsize=12)
        plt.ylabel("Makespan", fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(f"results/plots/plot_{instance_name}.png")
        plt.show()

    def plot_bars(self, file_paths: list[str]):
        """
        Reads results from JSON files, calculates averages, and plots a grouped bar chart.
        """
        df = self._read_json(file_paths)
        summary_df = df.groupby(['Instance', 'Algorithm'])['Makespan'].mean().reset_index()
        pivot_df = summary_df.pivot(index='Instance', columns='Algorithm', values='Makespan')

        ax = pivot_df.plot(kind='bar', figsize=(12, 7), width=0.8)
        plt.title('Algorithm Comparison: Average Makespan per Instance', fontsize=14)
        plt.xlabel('Instance', fontsize=12)
        plt.ylabel('Mean Makespan (Lower is Better)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        output_name = 'results/plots/alg_makespan_comparison.png'
        plt.savefig(output_name, dpi=300)
        print(f"Bar chart saved to {output_name}")
        plt.show()

    def _read_json(self, file_paths: list[str]) -> pd.DataFrame:
        all_data = []
        for fp in file_paths:
            if os.path.exists(fp):
                with open(fp, 'r') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_data.extend(data)
                        elif isinstance(data, dict):
                            all_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse {fp}. Skipping.")
        
        if not all_data:
            raise Exception("No data found to plot bars.")
            
        return pd.DataFrame(all_data)

def main() -> None:
    from src.algorithms.ml import MLSolver, Strategy
    from src.algorithms.lahc import LAHCSolver
    from src.algorithms.greedy import GreedyFJSSPWSolver
    from src.algorithms.spea import SPEA2Solver
    from src.algorithms.spea_lahc import HybridSPEALAHC

    parser = argparse.ArgumentParser(description="GECCO 2026 Benchmarking Suite")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument("--k", type=int, default=10, help="Number of runs per instance")
    run_parser.add_argument("--alg", nargs="*", help="Specific algorithm(s) to run (e.g., LAHC HYBRID)")
    run_parser.add_argument("--files", nargs="*", help="Files to run the benchmarks on")

    merge_parser = subparsers.add_parser("merge", help="Merge instance JSONs into one master file")
    merge_parser.add_argument("--alg", type=str, required=True, help="Algorithm name to merge (e.g., LAHC)")

    conv_parser = subparsers.add_parser("convergence", help="Plot convergence")
    conv_parser.add_argument("files", nargs="*", default=["results/HYBRID_history.json", "results/SPEA-II_history.json", "results/LAHC_history.json"], 
                             help="JSON history files")
    conv_parser.add_argument("--instance", type=str, default="2c_Hurink_rdata_28_workers.fjs", help="Instance name to plot")

    plot_parser = subparsers.add_parser("plot", help="Plot comparison bars")
    plot_parser.add_argument("files", nargs="*", default=["results/LAHC.json", "results/SPEA-II.json", "results/OtherResearcher.json", "results/GREEDY.json"], 
                             help="JSON result files")

    rank_parser = subparsers.add_parser("rank", help="Perform weighted ranking")
    rank_parser.add_argument("files", nargs="*", default=["results/LAHC.json", "results/SPEA-II.json", "results/OtherResearcher.json", "results/GREEDY.json"], 
                             help="JSON result files")

    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args(["--help"])

    runner = BenchmarkRunner("instances/fjssp-w")

    if args.command == "run":
        available_algorithms = {
            "LAHC": lambda: LAHCSolver(L=3000, max_iters=300_000),
            "HYBRID": lambda: HybridSPEALAHC(pop_size=40, max_generations=200, lahc_iters=300, archive_size=20, lahc_l=75, mutation_rate=0.03),
            "SPEA-II": lambda: SPEA2Solver(pop_size=378,archive_size=110,max_generations=500,mutation_rate=0.03663189655760893,mutation_limit=40,nuke_limit=70,TABU_THRESHOLD=20,TABU_DURATION=15),
            "ML": lambda: MLSolver(strategy=Strategy.PLUS, M=10, L=50, max_generations=50),
            "GREEDY": lambda: GreedyFJSSPWSolver()
        }

        target_names = args.alg if args.alg else list(available_algorithms.keys())
        selected_algorithms = {}
        for name in target_names:
            if name in available_algorithms:
                selected_algorithms[name] = available_algorithms[name]()
            else:
                print(f"Warning: Algorithm '{name}' not found. Skipping.")

        if not selected_algorithms:
            print("Error: No valid algorithms selected.")
            return

        runner.run_benchmark(selected_algorithms, k=args.k, filter=args.files)

    elif args.command == "merge":
        runner.merge_results(args.alg)

    elif args.command == "convergence":
        runner.plot_convergence(args.files, args.instance)

    elif args.command == "plot":
        runner.plot_bars(args.files)

    elif args.command == "rank":
        runner.perform_weighted_ranking(args.files)

if __name__ == "__main__":
    main()