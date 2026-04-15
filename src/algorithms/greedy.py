import random
from src.core.fjssp_algorithm import FJSSPAlgorithm
from src.core.candidate import Candidate, Operation, WorkerEncoding

class GreedyFJSSPWSolver(FJSSPAlgorithm):
    def __to_index(self, job, operation, job_sequence):
        counter = -1
        index = 0
        for i in job_sequence:
            if i == job:
                counter += 1
            if counter == operation:
                return index
            index += 1
        return None

    def determine_next(self, next_operation, durations, job_sequence, counts):
        next_durations = [0] * len(next_operation)
        machine = [float('inf')] * len(next_operation)
        worker = [float('inf')] * len(next_operation)
        min_index = float('inf')
        min_duration = float('inf')

        for i in range(len(next_operation)):
            if next_operation[i] >= counts[i]:
                continue
            index = self.__to_index(i, next_operation[i], job_sequence)
            operation_durations = durations[index]

            next_durations[i] = float('inf')
            for j in range(len(operation_durations)):
                for k in range(len(operation_durations[j])):
                    if operation_durations[j][k] > 0 and operation_durations[j][k] < next_durations[i]:
                        next_durations[i] = operation_durations[j][k]
                        machine[i] = j
                        worker[i] = k
                    elif operation_durations[j][k] > 0 and operation_durations[j][k] == next_durations[i] and random.random() < 0.5:
                        next_durations[i] = operation_durations[j][k]
                        machine[i] = j
                        worker[i] = k
        for i in range(len(next_durations)):
            if next_durations[i] > 0:
                if next_durations[i] < min_duration:
                    min_duration = next_durations[i]
                    min_index = i
                elif next_durations[i] == min_duration and random.random() < 0.5:
                    min_duration = next_durations[i]
                    min_index = i
        return min_index, min_duration, machine[min_index], worker[min_index]
    
    def solve(self, encoding: WorkerEncoding) -> tuple[Candidate, list]:
        durations = encoding.durations()
        job_sequence = encoding.job_sequence()
        jobs = sorted(list(set(job_sequence))) # Ensure consistent job ordering
        counts = [job_sequence.count(job) for job in jobs]

        next_operation = [0 for _ in jobs]
        
        # We need to store the specific machine/worker picked for 
        # EVERY operation index in the flat durations list
        assigned_machines = [0] * len(durations)
        assigned_workers = [0] * len(durations)
        
        # This is the priority sequence for the evaluator
        scheduled_job_sequence = []

        for _ in range(len(job_sequence)):
            job_idx, _, m, w = self.determine_next(next_operation, durations, job_sequence, counts)
            
            # Find where this specific operation lives in the global encoding
            abs_idx = self.__to_index(job_idx, next_operation[job_idx], job_sequence)
            
            assigned_machines[abs_idx] = int(m)
            assigned_workers[abs_idx] = int(w)
            scheduled_job_sequence.append(job_idx)
            
            next_operation[job_idx] += 1

        # Now reconstruct the Candidate. 
        # The Candidate needs ordered_ops to follow the 'scheduled_job_sequence'
        ordered_ops = []
        op_counters = [0 for _ in jobs]
        
        for job_id in scheduled_job_sequence:
            op_num = op_counters[job_id]
            abs_idx = self.__to_index(job_id, op_num, job_sequence)
            
            # Create the Operation with the specific data saved during determine_next
            op_info = Operation(
                machine_index=assigned_machines[abs_idx],
                worker_index=assigned_workers[abs_idx],
                job_index=job_id,
                operation_index=abs_idx, # The absolute index in the durations list
                duration=0, # Candidate.__init__ (via translate) will fix this
                offset=0    # Candidate.__init__ (via translate) will fix this
            )
            ordered_ops.append(op_info)
            op_counters[job_id] += 1

        # History is empty for a constructive solver
        candidate = Candidate(None, ordered_ops, encoding)
        return candidate, [(0, float(candidate.makespan))]
    
if __name__ == "__main__":
    solver = GreedyFJSSPWSolver()
    from src.util.benchmark_parser import WorkerBenchmarkParser
    encoding = WorkerBenchmarkParser().parse_benchmark("instances/fjssp-w/0_BehnkeGeiger_42_workers.fjs")
    print(solver.solve(encoding))