from .encoding import Encoding, WorkerEncoding
import numpy as np

class BenchmarkParser: 
    def __init__(self):
        pass

    def parse_benchmark(self, path: str):
        file_content = []

        try:
            file = open(path, 'r')
            file_content = file.readlines()
        except Exception as exception: print(exception) 

        info = file_content[0].split(' ')
        n_machines = int(info[1])
        n_overall_operations = 0
        lines = [line.split() for line in file_content[1:]]
        for i in range(1, len(file_content)):
            line = file_content[i].split(' ')
            lines[i - 1] = line
            n_overall_operations += int(line[0])
        
        durations = np.zeros((n_overall_operations, n_machines), dtype=int)
        operation_index = 0
        job_sequence = [None] * n_overall_operations

        for i in range(1, len(lines)):
            line = lines[i-1]
            n_operations = int(line[0])
            index = 1
            for j in range(0, n_operations):
                job_sequence[operation_index] = i-1
                n_options = int(line[index])
                index +=1
                for k in range(0, n_options):
                    machine = int(line[index])
                    index += 1
                    duration = int(line[index])
                    index += 1
                    durations[operation_index, machine - 1] = duration
                operation_index += 1
                
        return Encoding(durations, job_sequence)

class WorkerBenchmarkParser: 
    def __init__(self):
        pass

    def parse_benchmark(self, path: str, output = False):
        _print = print if output else lambda *a, **k: None
        file_content = []

        try:
            file = open(path, 'r')
            file_content = file.readlines()
        except Exception as exception: print(exception) 

        info = file_content[0].split(' ')
        n_jobs = int(info[0])
        n_machines = int(info[1])
        n_workers = int(round(float(info[2])))
        n_overall_operations = 0
        lines = [line.split() for line in file_content[1:]]
        
        # START: CUSTOM CODE
        operations_by_job = [0] * n_jobs
        # END

        for i in range(1, len(file_content)):
            line = file_content[i].split(' ')
            lines[i - 1] = line
            n_overall_operations += int(line[0])
            
            # START: CUSTOM CODE
            operations_count = int(line[0])
            operations_by_job[i-1] = operations_count
            # END
        
        durations = np.zeros((n_overall_operations, n_machines, n_workers), dtype=int)
        operation_index = 0
        job_sequence = [None] * n_overall_operations

        for i in range(1, len(lines)+1):
            line = lines[i-1]
            n_operations = int(line[0])
            _print(f"Total operations for job {i}: {n_operations}")
            index = 1
            for j in range(0, n_operations):
                job_sequence[operation_index] = i-1
                n_machine_options = int(line[index])
                _print(f"\tOptions for operation {j+1} (index {operation_index}): {n_machine_options}")
                index +=1
                for k in range(0, n_machine_options):
                    machine = int(line[index])
                    _print(f"\t\tOption {k+1}: Machine {machine}")
                    index += 1
                    n_worker_options = int(line[index])
                    _print("\t\t\tWorker options:", n_worker_options)
                    index += 1

                    for l in range(0, n_worker_options):
                        worker = int(line[index])
                        index += 1
                        duration = int(line[index])
                        _print(f"\t\t\t\t(Worker {worker}, duration {duration})")
                        index += 1
                        durations[operation_index, machine - 1, worker - 1] = duration
                
                operation_index += 1
        
        # START: CUSTOM CODE
        return WorkerEncoding(durations, job_sequence, operations_by_job, n_machines, n_workers)
        # END
