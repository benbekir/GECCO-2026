from dataclasses import dataclass
from src.util.encoding import WorkerEncoding
import src.util.evaluation as evaluation

@dataclass
class Operation:
    machine_index: int
    worker_index: int
    job_index: int
    operation_index: int
    duration: int
    offset: int

class Candidate:
    def __init__(self, schedule: list[list[Operation]] | None, ordered_ops: list[Operation], encoding: WorkerEncoding) -> None:
        self.schedule = schedule
        self.ordered_ops = ordered_ops

        seq, mach, work = self.get_sequences()
        start_times, m_fixed, w_fixed = evaluation.translate(seq, mach, work, encoding.durations())
        self.makespan = evaluation.makespan(start_times, m_fixed, w_fixed, encoding.durations())

    def __repr__(self) -> str:
        # {[[(task.job_index, task.task_index, task.duration, task.offset) for task in machine] for machine in self.schedule]}
        return f"{self.makespan}"

    def get_sequences(self) -> tuple[list,list,list]:
        sequence = [op.job_index for op in self.ordered_ops]
        
        total_ops = len(self.ordered_ops)
        machines = [0] * total_ops
        workers = [0] * total_ops
    
        for op in self.ordered_ops:
            machines[op.operation_index] = op.machine_index
            workers[op.operation_index] = op.worker_index

        return sequence, machines, workers