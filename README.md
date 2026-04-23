# Competition
This repository contains a meta-heuristic solution for the [Competition on Flexible Job Shop Scheduling Problems with Worker Flexibility under Uncertainty](https://gecco-2026.sigevo.org/Competition?itemId=8261) held by the Genetic and Evolutionary Computation Conference ([GECCO](https://gecco-2026.sigevo.org/HomePage)).

The link to the original competition repository can be found [HERE](https://github.com/jrc-rodec/FJSSP-W-Competition).

# Goal
The goal is to find a schedule that minimizes the makespan and the workload balance while respecting the constraints of machine and worker availability.

# Project structure
The project is structured as follows:

- **instances/**: Input benchmark datasets.
    - **fjssp/**: Original FJSSP benchmark instances.
    - **fjssp-w/**: Competition FJSSP-W instances (with worker flexibility).
- **src/**:
    - **algorithms/**: Meta-heuristic implementations including `SPEA-II`, `LAHC`, `SPEA-II + LAHC Hybrid`, and `(μ,λ) and (μ+λ) Evolution Strategies`.
    - **core/**: Core domain objects and abstractions used across algorithms.
    - **util/**: Utility modules provided by the official competition organizers for parsing, encoding and evaluation.
    - **benchmarker.py**: Benchmark execution and experiment orchestration.
    - **hyperparameter.py**: Hyperparameter search/tuning utilities.
- **results/**: Saved outputs from experiments.
    - **\*.json / \*.csv**: Algorithm results and makespan convergence histories.
    - **params/**: Best-found parameter configurations.
    - **plots/**: Generated figures and visual summaries.

# Data format
The 30 competition instances can be found under *instances/fjssp-w*, each of which is a text file containing the description of a FJSSP-W instance. The format of these files is as follows:
### FJSSP-W
#### Header line
1. Number of jobs
2. Number of machines
3. Number of workers

#### Job description lines
1. Number of total operations for the job
2. For each operation:
    1. Number of machines that can process the operation
    2. For each machine:
        1. Machine index
        2. Number of workers that can operate the machine
        3. For each worker:
            1. Worker index
            2. Duration

# Usage
To run the benchmarking suite, use the `benchmarker.py` module:

```bash
python -m src.benchmarker <command> [options]
```

## 1) Run benchmarks
Runs the benchmarks on the specified algorithms and instances, saving results to JSON files under the `results/` directory. Each algorithm-instance pair will produce a two JSON files:
- `[ALG]_[INST].json` containing the best makespan results and other statistics for each run.
- `[ALG]_[INST]_history.json` containing the makespan at each iteration for convergence plotting (for the first run only).
```bash
python -m src.benchmarker run --alg <ALG1> <ALG2> ... [--instances <INST1> <INST2> ...] [--k <K>]
```

- `--alg`: Algorithm(s) to run.
    - Available: `SPEA-II`, `LAHC`, `HYBRID`, `GREEDY`, `ML`.
- `--instances`: Subset of instance filenames to run. If omitted, all instances are used.
- `--k`: Number of runs per instance (default: `10`).

Example:

```bash
python -m src.benchmarker run --alg LAHC HYBRID --instances 0_BehnkeGeiger_42_workers.fjs 2c_Hurink_rdata_28_workers.fjs --k 20
```

## 2) Merge per-instance results
Since each algorithm-instance pair produces a separate JSON file, this command merges them into a single JSON file per algorithm for easier analysis and plotting.

```bash
python -m src.benchmarker merge --alg <ALGORITHM_NAME>
```

- `--alg`: Algorithm name whose split JSON results should be merged into a single file.
    - Available: `SPEA-II`, `LAHC`, `HYBRID`, `GREEDY`, `ML`.

Example:

```bash
python -m src.benchmarker merge --alg LAHC
```

## 3) Plot convergence curves
Create convergence plots showing how the makespan evolves over iterations for a given instance across different algorithms and save the plots under `results/plots/`.

```bash
python -m src.benchmarker convergence --files <HIST_FILE1> <HIST_FILE2> ... --instance <INSTANCE_NAME>
```

- `--files`: History JSON files.
    - Default: `results/HYBRID_history.json`, `results/SPEA-II_history.json`, `results/LAHC_history.json`.
- `--instance`: Instance to visualize.
    - Default: `2c_Hurink_rdata_28_workers.fjs`.

Example:

```bash
python -m src.benchmarker convergence --files results/HYBRID_history.json results/LAHC_history.json --instance 2c_Hurink_rdata_28_workers.fjs
```

## 4) Plot bar comparison
Create bar plots comparing the best makespan results across algorithms for a given instance and save the plots under `results/plots/`.

```bash
python -m src.benchmarker plot --files <RESULT_FILE1> <RESULT_FILE2> ...
```

- `--files`: Result JSON files.
    - Default: `results/LAHC.json`, `results/SPEA-II.json`, `results/OtherResearcher.json`, `results/GREEDY.json`.

Example:

```bash
python -m src.benchmarker plot --files results/LAHC.json results/SPEA-II.json results/GREEDY.json
```

## 5) Perform weighted ranking
Ranks the algorithms based on the Mann-Whitney U test results across all instances, giving more weight to better-performing algorithms and shows the final ranking in a tabular format.

```bash
python -m src.benchmarker rank --files <RESULT_FILE1> <RESULT_FILE2> ...
```

- `--files`: Result JSON files.
    - Default: `results/LAHC.json`, `results/SPEA-II.json`, `results/OtherResearcher.json`, `results/GREEDY.json`.

Example:

```bash
python -m src.benchmarker rank --files results/LAHC.json results/SPEA-II.json results/OtherResearcher.json results/GREEDY.json
```