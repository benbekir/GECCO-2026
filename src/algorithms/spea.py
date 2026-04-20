from src.core.fjssp_algorithm import FJSSPAlgorithm
from src.core.candidate import Candidate, Operation,WorkerEncoding
from src.algorithms.aspea import Instance, density_function, environmental_selection, binary_tournament, calculate_fitness
import src.util.evaluation as evaluation

class SPEA2Solver(FJSSPAlgorithm):
    def __init__(self, pop_size=200, archive_size=50, max_generations=500,mutation_rate=0.1,mutation_limit=50,nuke_limit=150):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.base_mutation = mutation_rate
        self.nuke_limit=nuke_limit
        self.mutation_limit=mutation_limit

    def get_evaluations(self) -> int:
        return int(self.pop_size * self.max_generations)

    def solve(self, encoding: WorkerEncoding) -> tuple[Candidate, list]:
        all_options = Instance.create_options(encoding)
        population = [Instance(encoding, all_options) for _ in range(self.pop_size)]
        archive = []
        history_best_makespan = []
        tracker = 0
        global_best_makespan = float('inf')

        for ind in population:
            ind.makespan, ind.worker_balance_fitness = calculate_fitness(ind)

        for gen in range(1, self.max_generations + 1):
            combined = density_function(population, archive)
            archive = environmental_selection(combined, self.archive_size)
            current_best = min(ind.makespan for ind in archive)
         
            history_best_makespan.append((gen, current_best))
            SIGNIFICANCE_THRESHOLD = max(5, global_best_makespan * 0.005)
            if global_best_makespan - current_best > SIGNIFICANCE_THRESHOLD:
                global_best_makespan = current_best
                tracker = 0
                current_mutation = self.base_mutation
            else:
                tracker += 1
            if tracker > self.nuke_limit:
                archive = [min(archive, key=lambda x: x.makespan)]
                population = [Instance(encoding, all_options) for _ in range(self.pop_size)]
                for ind in population:
                    ind.makespan, ind.worker_balance_fitness = calculate_fitness(ind)
                tracker = 0
                current_mutation = self.base_mutation
            elif tracker > self.mutation_limit:
                stuck_duration = tracker - self.mutation_limit
                max_stuck = self.nuke_limit - self.mutation_limit
                ramp = (stuck_duration / max_stuck)
                current_mutation = self.base_mutation + (0.2 - self.base_mutation) * ramp
            else:
                current_mutation = self.base_mutation

            is_stuck = tracker > self.mutation_limit
            population = binary_tournament(archive, self.pop_size, current_mutation, is_stuck)

        best_instance = min(archive, key=lambda x: x.makespan)
        return self.convert_to_candidate(best_instance, encoding), history_best_makespan


    def convert_to_candidate(self, instance, encoding):
        m_assignments = [item[0] for item in instance.worker_machine_sequence]
        w_assignments = [item[1] for item in instance.worker_machine_sequence]
        durations_3d = encoding.durations()
        
        start_times, m_fixed, w_fixed = evaluation.translate(
            instance.operation_sequence, m_assignments, w_assignments, durations_3d
        )
        
        ordered_ops = []
        for i in range(len(instance.operation_sequence)):
            m, w = m_fixed[i], w_fixed[i]
            d = durations_3d[i][m][w]
            
            op_info = Operation(
                machine_index=m,
                worker_index=w,
                job_index=instance.operation_sequence[i],
                operation_index=i,
                duration=d,
                offset=start_times[i]
            )
            ordered_ops.append(op_info)
            
        return Candidate([], ordered_ops, encoding)