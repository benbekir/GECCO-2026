from src.util.greedy_solver import GreedyFJSSPSolver, GreedyFJSSPWSolver
from src.util.benchmark_parser import WorkerBenchmarkParser
parser = WorkerBenchmarkParser()
encoding = parser.parse_benchmark(r'instances/fjssp-w/3_DPpaulli_1_workers.fjs')

solver = GreedyFJSSPWSolver(encoding.durations(), encoding.job_sequence())
sequence, machines, workers = solver.solve()

import src.util.evaluation as evaluation
start_times, machines, workers = evaluation.translate(sequence, machines, workers, encoding.durations()) # necessary because the solver does not provide the start times of the schedule
c = evaluation.makespan(start_times, machines, workers, encoding.durations())