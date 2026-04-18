import copy
import random
from src.core.candidate import Candidate, Operation, WorkerEncoding
from src.core.fjssp_algorithm import FJSSPAlgorithm

class LAHCSolver(FJSSPAlgorithm):
    """Late-acceptance Hill Climber"""
    def __init__(self, **kwargs):
        self.L = kwargs.get('L', 50)
        self.max_iters = kwargs.get('max_iters', 10000)

    def __get_initial_candidate(self, encoding: WorkerEncoding, machines_for_ops, last_operation_by_job):
        num_jobs = encoding.n_jobs()
        num_machines = encoding.n_machines()
        num_workers = encoding.n_workers()
        
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
        # populate schedule for each machine
        while active_jobs:
            selected_job = random.choice(active_jobs)
            selected_op = next_operation_by_job[selected_job]

            # select random viable machine and worker
            usable_machines = machines_for_ops[selected_op]
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
            next_operation_by_job[selected_job] += 1
            if next_operation_by_job[selected_job] > last_operation_by_job[selected_job]:
                active_jobs.remove(selected_job)
    
        return Candidate(schedule, ordered_ops, encoding)

    def __get_neighbor(self, encoding: WorkerEncoding, machines_for_ops, candidate: Candidate):
        new_ordered_ops = copy.deepcopy(candidate.ordered_ops)
        idx1 = random.randrange(len(new_ordered_ops))
        
        mutation_type = random.random()

        # change machine and worker assignment
        if mutation_type < 0.4:
            # we do not have to check if new machine and worker are free
            # since in that case, the translate function will simply delay the start time
            op = new_ordered_ops[idx1]
            usable_machines = machines_for_ops[op.operation_index]
            new_m = random.choice(usable_machines)
            usable_workers = encoding.get_workers_for_operation_on_machine(op.operation_index, new_m)
            new_w = random.choice(usable_workers)
            
            op.machine_index = new_m
            op.worker_index = new_w
            op.duration = encoding.durations()[op.operation_index][new_m][new_w]

        # swap the job order of two operations
        elif mutation_type < 0.8:
            # making a change from job ids [0,2,*2*] to [*2*,2,0] wouldn't break anything 
            # because machine and worker assignments are operation specific and operations are always executed in order.
            # therefore, the resulting change would be [job 0 op 0, job 2 op 0, job 2 op 1] to [job 2 op 0, job 2 op 1, job 0 op 0].
            idx2 = random.randrange(len(new_ordered_ops))
            tmp_job = new_ordered_ops[idx1].job_index
            new_ordered_ops[idx1].job_index = new_ordered_ops[idx2].job_index
            new_ordered_ops[idx2].job_index = tmp_job

        # shift operation to a new position
        else:
            target_idx = random.randint(0, len(new_ordered_ops) - 1)
            job_ids = [op.job_index for op in new_ordered_ops]
            moving_job = job_ids.pop(idx1)
            job_ids.insert(target_idx, moving_job)
            # re-assign the shifted job ids back to the fixed operation slots
            for i in range(len(new_ordered_ops)):
                new_ordered_ops[i].job_index = job_ids[i]

        return Candidate([], new_ordered_ops, encoding)

    def solve(self, encoding: WorkerEncoding) -> tuple[Candidate, list]:
        num_jobs = encoding.n_jobs()
        last_operation_by_job = [0] * num_jobs
        operation_index = 0
        for i in range(num_jobs):
            ops_for_current_job = encoding.get_operations_for_job(i)
            operation_index += ops_for_current_job
            last_operation_by_job[i] = operation_index - 1

        machines_for_ops = encoding.get_all_machines_for_all_operations()

        candidate = self.__get_initial_candidate(encoding, machines_for_ops, last_operation_by_job)
        current = candidate
        best = candidate
        
        history = [current.makespan] * self.L
        progression = []

        for i in range(self.max_iters):
            candidate = self.__get_neighbor(encoding, machines_for_ops, current)
            v = i % self.L

            if candidate.makespan < current.makespan or candidate.makespan < history[v]:
                current = candidate
            elif candidate.makespan == current.makespan and candidate.get_balance() < current.get_balance():
                current = candidate

            if current.makespan < best.makespan:
                best = current
            elif current.makespan == best.makespan and current.get_balance() < best.get_balance():
                best = current
            
            history[v] = current.makespan
            
            if i % 1000 == 0:
                progression.append((i, best.makespan))

        return best, progression

if __name__ == "__main__":
    from src.util.benchmark_parser import WorkerBenchmarkParser
    import os
    from src.util.evaluation import workload_balance
    files = [f for f in os.listdir("instances/fjssp-w") if f.endswith('.fjs')]
    scores = dict()
    for file in files:
        print(file)
        encoding = WorkerBenchmarkParser().parse_benchmark(f"instances/fjssp-w/{file}")
        c, h = LAHCSolver(L=50, max_iters=5000).solve(encoding)
        _, machines, workers = c.get_sequences()
        balance = workload_balance(machines, workers, encoding.durations())
        print(c.makespan, balance)
        scores[file] = c.makespan
    print(scores)