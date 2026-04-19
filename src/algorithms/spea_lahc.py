from src.core.fjssp_algorithm import FJSSPAlgorithm
from src.core.candidate import Candidate, WorkerEncoding
from src.core.instance import Instance
from src.algorithms.aspea import density_function, environmental_selection, binary_tournament
import random
import copy

class HybridSPEALAHC(FJSSPAlgorithm):
    def __init__(self, pop_size=40, archive_size=20, max_generations=50, lahc_iters=25, lahc_l=10):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.lahc_iters = lahc_iters  
        self.lahc_l = lahc_l
        self.mutation_rate = 0.03

    def __get_neighbor(self, encoding: WorkerEncoding, machines_for_ops, candidate: Candidate):
        # We only deepcopy the list of operations, not the whole solver/instance
        new_ordered_ops = copy.deepcopy(candidate.ordered_ops)
        idx1 = random.randrange(len(new_ordered_ops))
        mutation_type = random.random()

        if mutation_type < 0.4:
            op = new_ordered_ops[idx1]
            usable_machines = machines_for_ops[op.operation_index]
            new_m = random.choice(usable_machines)
            usable_workers = encoding.get_workers_for_operation_on_machine(op.operation_index, new_m)
            new_w = random.choice(usable_workers)
            
            op.machine_index = new_m
            op.worker_index = new_w
            op.duration = encoding.durations()[op.operation_index][new_m][new_w]

        elif mutation_type < 0.8:
            idx2 = random.randrange(len(new_ordered_ops))
            tmp_job = new_ordered_ops[idx1].job_index
            new_ordered_ops[idx1].job_index = new_ordered_ops[idx2].job_index
            new_ordered_ops[idx2].job_index = tmp_job

        else:
            target_idx = random.randint(0, len(new_ordered_ops) - 1)
            job_ids = [op.job_index for op in new_ordered_ops]
            moving_job = job_ids.pop(idx1)
            job_ids.insert(target_idx, moving_job)
            for i in range(len(new_ordered_ops)):
                new_ordered_ops[i].job_index = job_ids[i]

        return Candidate([], new_ordered_ops, encoding)

    def _local_search_lahc(self, instance: Instance, encoding: WorkerEncoding, machines_for_ops):
        current_cand = instance.to_candidate()
        history = [current_cand.makespan] * self.lahc_l
        curr_m = current_cand.makespan
        
        for i in range(self.lahc_iters):
            neighbor = self.__get_neighbor(encoding, machines_for_ops, current_cand)
            v = i % self.lahc_l
            
            if neighbor.makespan < curr_m or neighbor.makespan < history[v]:
                current_cand = neighbor
                curr_m = neighbor.makespan
            
            history[v] = curr_m

        job_seq, m_seq, w_seq = current_cand.get_sequences()
        instance.operation_sequence = job_seq
        instance.worker_machine_sequence = list(zip(m_seq, w_seq))
        instance.makespan = current_cand.makespan
        instance.worker_balance_fitness = current_cand.get_balance()

    def solve(self, encoding: WorkerEncoding) -> tuple[Candidate, list]:
        all_options = Instance.create_options(encoding)
        machines_for_ops = encoding.get_all_machines_for_all_operations()
        population = [Instance(encoding, all_options) for _ in range(self.pop_size)]
        archive = []
        progression = []

        for ind in population:
            ind.update_fitness()

        for gen in range(1, self.max_generations + 1):
            print(f"Generation: {gen}/{self.max_generations}")
            combined = density_function(population, archive)
            archive = environmental_selection(combined, self.archive_size)
            
            current_best = min(ind.makespan for ind in archive)
            progression.append((gen, current_best))

            population = binary_tournament(archive, self.pop_size, self.mutation_rate, False)

            for offspring in population:
                self._local_search_lahc(offspring, encoding, machines_for_ops)

        # 3. Return best from the final archive
        best_instance = min(archive, key=lambda x: x.makespan)
        return best_instance.to_candidate(), progression