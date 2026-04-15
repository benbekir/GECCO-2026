# Competition
This repository contains a meta-heuristic solution for the [Competition on Flexible Job Shop Scheduling Problems with Worker Flexibility under Uncertainty](https://gecco-2026.sigevo.org/Competition?itemId=8261) held by the Genetic and Evolutionary Computation Conference ([GECCO](https://gecco-2026.sigevo.org/HomePage)).

The link to the original competition repository can be found [HERE](https://github.com/jrc-rodec/FJSSP-W-Competition).

The goal is to find a schedule that minimizes the makespan while respecting the constraints of machine and worker availability.

## Data format
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


## TODOS
- add wilcoxon ranking to measure algorithm performance
- add local search for last part of the algorithm (as soon as there are X generations without improvement)
    - apply local search for a few iterations and continue with the original algorithm
- visualize solution with Gantt chart
- visualize convergence, comparison of algorithms regarding makespan and worker balance