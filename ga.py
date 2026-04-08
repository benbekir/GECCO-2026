from util.benchmark_parser import WorkerBenchmarkParser
from util.encoding import WorkerEncoding
from enum import Enum
from dataclasses import dataclass
import random
import util.evaluation as evaluation

class Strategy(Enum):
    PLUS = 1
    COMMA = 2

@dataclass
class Operation:
    machine_index: int
    worker_index: int
    job_index: int
    operation_index: int
    duration: int
    offset: int

class Candidate:
    def __init__(self, schedule: list[list[Operation]], ordered_ops: list[Operation], encoding: WorkerEncoding) -> None:
        self.schedule = schedule
        self.ordered_ops = ordered_ops

        seq, mach, work = self.get_sequences()
        start_times, m_fixed, w_fixed = evaluation.translate(seq, mach, work, encoding.durations())
        self.time = evaluation.makespan(start_times, m_fixed, w_fixed, encoding.durations())

    def __repr__(self) -> str:
        # {[[(task.job_index, task.task_index, task.duration, task.offset) for task in machine] for machine in self.schedule]}
        return f"{self.time}"

    def get_sequences(self) -> tuple[list,list,list]:
        sequence = [op.job_index for op in self.ordered_ops]
        
        total_ops = len(self.ordered_ops)
        machines = [0] * total_ops
        workers = [0] * total_ops
    
        for op in self.ordered_ops:
            machines[op.operation_index] = op.machine_index
            workers[op.operation_index] = op.worker_index

        return sequence, machines, workers

def create_candidate(schedule, machine_ready_times, operation_ready_times, worker_ready_times, ordered_ops, next_op_by_job, last_operation_by_job, active_jobs, encoding: WorkerEncoding) -> Candidate:
    # populate schedule for each machine
    while active_jobs:
        selected_job = random.choice(active_jobs)
        selected_op = next_op_by_job[selected_job]

        # select random viable machine and worker
        usable_machines = encoding.get_machines_for_operation(selected_op)
        selected_machine = random.choice(usable_machines)
        usable_workers = encoding.get_workers_for_operation_on_machine(selected_op, selected_machine)
        selected_worker = random.choice(usable_workers)

        offset = max(operation_ready_times[selected_job], machine_ready_times[selected_machine], worker_ready_times[selected_worker])
        duration = encoding.durations()[selected_op][selected_machine][selected_worker]
        op_info = Operation(selected_machine, selected_worker, selected_job, selected_op, duration, offset)
        schedule[selected_machine].append(op_info)
        ordered_ops.append(op_info)

        operation_ready_times[selected_job] = duration + offset
        machine_ready_times[selected_machine] = duration + offset
        worker_ready_times[selected_worker] = duration + offset

        # check if this was the last task for this job
        next_op_by_job[selected_job] += 1
        if next_op_by_job[selected_job] > last_operation_by_job[selected_job]:
            active_jobs.remove(selected_job)
    
    return Candidate(schedule, ordered_ops, encoding)

def get_initial_candidates(m, last_operation_by_job, encoding: WorkerEncoding):
    num_jobs = encoding.n_jobs()
    num_machines = encoding.n_machines()
    num_workers = encoding.n_workers()
    candidates = list[Candidate]()
    for _ in range(m):
        # create a schedule for each machine
        schedule = [list[Operation]() for _ in range(num_machines)]

        # for each machine, store when it becomes available again
        machine_ready_times = [0] * num_machines
        operation_ready_times = [0] * num_jobs
        worker_ready_times = [0] * num_workers

        ordered_ops = []

        # for each job, store the index of the next operation to schedule
        next_operation_by_job = [0] * num_jobs
        operation_index = 0
        for i in range(num_jobs):
            ops_for_current_job = encoding.get_operations_for_job(i)
            next_operation_by_job[i] = operation_index
            operation_index += ops_for_current_job

        # jobs that still have tasks left
        active_jobs = [index for index in range(num_jobs)]
        candidate = create_candidate(schedule, machine_ready_times, operation_ready_times, worker_ready_times, ordered_ops, next_operation_by_job, last_operation_by_job, active_jobs, encoding)
        candidates.append(candidate)
    return candidates

def mutate(parents: list[Candidate], l, last_operation_by_job, encoding: WorkerEncoding) -> list[Candidate]:
    num_jobs = encoding.n_jobs()
    num_machines = encoding.n_machines()
    num_workers = encoding.n_workers()
    children_per_parent = l / len(parents)
    children = list[Candidate]()
    for parent in parents:
        for _ in range(int(children_per_parent)):
            # select random machine and operation to split at
            operations_for_machine = 0
            random_machine_idx = 0
            while operations_for_machine == 0:
                random_machine_idx = random.randint(0, num_machines - 1)
                operations_for_machine = len(parent.schedule[random_machine_idx])
            random_operation_idx = random.randint(0, operations_for_machine - 1)
            chosen_op = parent.schedule[random_machine_idx][random_operation_idx]

            # reconstruct both the schedule and ordered operations from the ordered sequence
            schedule = [list[Operation]() for _ in range(num_machines)]
            ordered_ops = list[Operation]()
            for op in parent.ordered_ops:
                if op.offset <= chosen_op.offset:
                    ordered_ops.append(op)
                    schedule[op.machine_index].append(op)
                    
                if op == chosen_op:
                    break
            
            # reconstruct state from the partial schedule
            next_operation_by_job = [0] * num_jobs
            op_counter = 0
            for i in range(num_jobs):
                next_operation_by_job[i] = op_counter
                op_counter += encoding.get_operations_for_job(i)
            machine_ready_times = [0] * num_machines
            operation_ready_times = [0] * num_jobs
            worker_ready_times = [0] * num_workers
            
            for op in ordered_ops:
                end_time = op.offset + op.duration
                machine_ready_times[op.machine_index] = end_time
                worker_ready_times[op.worker_index] = end_time
                operation_ready_times[op.job_index] = end_time
                next_operation_by_job[op.job_index] = op.operation_index + 1
            
            # only jobs with remaining tasks are active
            active_jobs = [job_index for job_index in range(num_jobs) if next_operation_by_job[job_index] <= last_operation_by_job[job_index]]
            
            candidate = create_candidate(schedule, machine_ready_times, operation_ready_times, worker_ready_times, ordered_ops, next_operation_by_job, last_operation_by_job, active_jobs, encoding)
            children.append(candidate)
    return children

def solve(strategy: Strategy, m, l, max_generations, encoding: WorkerEncoding) -> tuple[Candidate, list]:
    num_jobs = encoding.n_jobs()
    last_operation_by_job = [0] * num_jobs
    operation_index = 0
    for i in range(num_jobs):
        ops_for_current_job = encoding.get_operations_for_job(i)
        operation_index += ops_for_current_job
        last_operation_by_job[i] = operation_index - 1

    parents = get_initial_candidates(m, last_operation_by_job, encoding)
    offsprings = list[Candidate]()
    history = []
    for gen in range(max_generations):
        offsprings = mutate(parents, l, last_operation_by_job, encoding)

        if strategy == Strategy.PLUS:
            offsprings.extend(parents)
            offsprings.sort(key=lambda x: x.time) 
            parents = []
            for i in range(m):
                parents.append(offsprings[i])
            print(parents)

        elif strategy == Strategy.COMMA:
            offsprings.sort(key=lambda x: x.time) 
            parents = []
            for i in range(m):
                parents.append(offsprings[i])
            print(parents)

        current_best = parents[0].time
        history.append((gen, current_best))

    parents.sort(key=lambda x: x.time) 
    return parents[0], history

def run(instance_file: str, strategy: Strategy) -> tuple[Candidate, list]:
    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(instance_file)
    best, hist = solve(strategy, 10, 50, 50, encoding)
    print("Final best:", best.time)
    return best, hist