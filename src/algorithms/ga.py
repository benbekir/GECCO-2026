from enum import Enum
from src.core.candidate import Candidate, Operation, WorkerEncoding
from src.core.fjssp_algorithm import FJSSPAlgorithm
import random

class Strategy(Enum):
    PLUS = 1
    COMMA = 2

class GASolver(FJSSPAlgorithm):
    def __init__(self, **kwargs) -> None:
        self.strategy = kwargs.get('strategy', Strategy.PLUS)
        self.M = kwargs.get('M', 10)
        self.L = kwargs.get('L', 50)
        self.max_generations = kwargs.get('max_generations', 50)

    def __create_candidate(self, encoding: WorkerEncoding, schedule, machine_ready_times, operation_ready_times, worker_ready_times, ordered_ops, next_op_by_job, last_operation_by_job, active_jobs) -> Candidate:
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

    def __get_initial_candidates(self, encoding: WorkerEncoding, last_operation_by_job):
        num_jobs = encoding.n_jobs()
        num_machines = encoding.n_machines()
        num_workers = encoding.n_workers()
        candidates = list[Candidate]()
        for _ in range(self.M):
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
            candidate = self.__create_candidate(encoding, schedule, machine_ready_times, operation_ready_times, worker_ready_times, ordered_ops, next_operation_by_job, last_operation_by_job, active_jobs)
            candidates.append(candidate)
        return candidates

    def __mutate(self, encoding: WorkerEncoding, parents: list[Candidate], last_operation_by_job) -> list[Candidate]:
        num_jobs = encoding.n_jobs()
        num_machines = encoding.n_machines()
        num_workers = encoding.n_workers()
        children_per_parent = self.L / len(parents)
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
                
                candidate = self.__create_candidate(encoding, schedule, machine_ready_times, operation_ready_times, worker_ready_times, ordered_ops, next_operation_by_job, last_operation_by_job, active_jobs)
                children.append(candidate)
        return children

    def solve(self, encoding: WorkerEncoding) -> tuple[Candidate, list]:
        num_jobs = encoding.n_jobs()
        last_operation_by_job = [0] * num_jobs
        operation_index = 0
        for i in range(num_jobs):
            ops_for_current_job = encoding.get_operations_for_job(i)
            operation_index += ops_for_current_job
            last_operation_by_job[i] = operation_index - 1

        parents = self.__get_initial_candidates(encoding, last_operation_by_job)
        offsprings = list[Candidate]()
        history = []
        for gen in range(self.max_generations):
            offsprings = self.__mutate(encoding, parents, last_operation_by_job)

            if self.strategy == Strategy.PLUS:
                offsprings.extend(parents)
                offsprings.sort(key=lambda x: x.makespan) 
                parents = []
                for i in range(self.M):
                    parents.append(offsprings[i])
                print(parents)

            elif self.strategy == Strategy.COMMA:
                offsprings.sort(key=lambda x: x.makespan) 
                parents = []
                for i in range(self.M):
                    parents.append(offsprings[i])
                print(parents)

            current_best = parents[0].makespan
            history.append((gen, current_best))

        parents.sort(key=lambda x: x.makespan) 
        return parents[0], history