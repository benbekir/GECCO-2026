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
    def __init__(self, schedule: list[list[Operation]], ordered_ops: list[Operation], encoding: WorkerEncoding) -> None:
        self.schedule = schedule
        self.ordered_ops = ordered_ops
        self._balance = None
        self._encoding = encoding

        seq, mach, work = self.get_sequences()
        start_times, m_fixed, w_fixed = evaluation.translate(seq, mach, work, encoding.durations())
        self.makespan = evaluation.makespan(start_times, m_fixed, w_fixed, encoding.durations())
    
    @classmethod
    def from_sequences(cls, job_seq: list[int], machine_worker_pairs: list[tuple], encoding: WorkerEncoding):
        """Creates a Candidate directly from SPEA2-style sequences."""
        ops = []
        for i in range(len(job_seq)):
            m, w = machine_worker_pairs[i]
            ops.append(Operation(m, w, job_seq[i], i, 0, 0))
        return cls([], ops, encoding)

    def get_balance(self):
        if self._balance is None:
            _, mach, work = self.get_sequences()
            self._balance = evaluation.workload_balance(mach, work, self._encoding.durations())
        return self._balance

    def get_sequences(self) -> tuple[list, list, list]:
        sequence = [op.job_index for op in self.ordered_ops]
        total_ops = len(self.ordered_ops)
        machines = [0] * total_ops
        workers = [0] * total_ops
        for op in self.ordered_ops:
            machines[op.operation_index] = op.machine_index
            workers[op.operation_index] = op.worker_index
        return sequence, machines, workers