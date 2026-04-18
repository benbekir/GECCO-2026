from src.core.fjssp_algorithm import FJSSPAlgorithm
from src.core.candidate import Candidate, WorkerEncoding
from src.core.instance import Instance
from src.algorithms.aspea import density_function, environmental_selection, binary_tournament

class HybridSPEALAHC(FJSSPAlgorithm):
    def __init__(self, pop_size=40, archive_size=20, max_generations=50, lahc_iters=25, lahc_l=10):
        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_generations = max_generations
        self.lahc_iters = lahc_iters  
        self.lahc_l = lahc_l
        self.mutation_rate = 0.03

    def _local_search_lahc(self, instance: Instance):
        """
        Applies a short LAHC burst to an SPEA2 Instance using its genetic 
        representation (swapping and update_fitness).
        """
        history = [instance.makespan] * self.lahc_l
        curr_m = instance.makespan
        
        for i in range(self.lahc_iters):
            neighbor = instance.copy()
            neighbor.swapping(mutation_rate=0.1)
            neighbor.update_fitness()
            
            v = i % self.lahc_l
            
            if neighbor.makespan < curr_m or neighbor.makespan < history[v]:
                instance.operation_sequence = neighbor.operation_sequence
                instance.worker_machine_sequence = neighbor.worker_machine_sequence
                instance.makespan = neighbor.makespan
                instance.worker_balance_fitness = neighbor.worker_balance_fitness
                
                curr_m = neighbor.makespan
            
            history[v] = curr_m

    def solve(self, encoding: WorkerEncoding) -> tuple[Candidate, list]:
        all_options = Instance.create_options(encoding)
        
        population = [Instance(encoding, all_options) for _ in range(self.pop_size)]
        archive = []
        progression = []

        for ind in population:
            ind.update_fitness()

        for gen in range(1, self.max_generations + 1):
            print(f"Gen {gen}/{self.max_generations}")
            combined = density_function(population, archive)
            archive = environmental_selection(combined, self.archive_size)
            
            current_best = min(ind.makespan for ind in archive)
            progression.append((gen, current_best))

            population = binary_tournament(archive, self.pop_size, self.mutation_rate, False)

            for offspring in population:
                self._local_search_lahc(offspring)

        # 3. Return best from the final archive
        best_instance = min(archive, key=lambda x: x.makespan)
        return best_instance.to_candidate(), progression