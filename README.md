# Competition
This repository contains a meta-heuristic solution for the [Competition on Flexible Job Shop Scheduling Problems with Worker Flexibility under Uncertainty](https://gecco-2026.sigevo.org/Competition?itemId=8261) held by the Genetic and Evolutionary Computation Conference ([GECCO](https://gecco-2026.sigevo.org/HomePage)).

The link to the original competition repository can be found [HERE](https://github.com/jrc-rodec/FJSSP-W-Competition).

The goal is to find a schedule that minimizes the makespan while respecting the constraints of machine and worker availability.

## Data format
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

## Questions
1. Is the FJSSP instance folder of any relevance to us? We currently believe that it does not contain instance that need to be solved for the competition. Is that correct?