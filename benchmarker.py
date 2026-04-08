import os
import time
import pandas as pd
import ga
import Algorithms

class BenchmarkRunner:
    def __init__(self, instances_dir: str):
        self.instances_dir = instances_dir
        self.files = [f for f in os.listdir(instances_dir) if f.endswith('.fjs')]
        self.results = []

    def run_benchmark(self, algorithms: dict):
        """
        algorithms: A dictionary where key is the name and value is the function 
                    that takes a filepath and returns (best_candidate, history)
        """
        for filename in self.files:
            filepath = os.path.join(self.instances_dir, filename)
            print(f"\n--- Benchmarking Instance: {filename} ---")
            
            for name, algo_func in algorithms.items():
                print(f"Running {name}...", end=" ", flush=True)
                
                start_time = time.time()
                best_candidate, history = algo_func(filepath)
                duration = time.time() - start_time
                
                self.results.append({
                    "Instance": filename,
                    "Algorithm": name,
                    "Best Makespan": best_candidate.time,
                    "Runtime (s)": round(duration, 2),
                    "Final Gen": history[-1][0] if history else 0
                })
                print(f"Done. Makespan: {best_candidate.time}")

    def get_summary(self):
        return pd.DataFrame(self.results)

    def save_results(self, output_file="benchmark_results.csv"):
        df = self.get_summary()
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    runner = BenchmarkRunner("instances/fjssp-w")

    algorithms = {
        "SPEA-II-GA": lambda path: Algorithms.run(),
    }

    runner.run_benchmark(algorithms)
    
    summary_df = runner.get_summary()
    print("\nFinal Comparison:")
    print(summary_df.pivot(index="Instance", columns="Algorithm", values="Best Makespan"))
    
    runner.save_results()