import random
import src.util.evaluation as evaluation
from src.algorithms.aspea import calculate_fitness
class TabuLocalSearch:
    def __init__(self, tabu_size=20, max_steps=50):
        self.tabu_size = tabu_size
        self.max_steps = max_steps
    def initialize_sequence(num_machines,num_jobs):
        sequence=[]
        for i in range(num_jobs):
            sequence.extend([i]*num_machines)
        random.shuffle(sequence)
        return sequence
    @staticmethod
    def get_neighbour_tabu(candidate):
        move_options = ["Swap_Sequence", "Change_Assignment"]
        move_type=random.choice(move_options)
        neighbour=candidate.copy()
        move_details=None
        if move_type=="Swap_Sequence":
            idx1, idx2 = random.sample(range(len(neighbour.operation_sequence)), 2)
            neighbour.operation_sequence[idx1], neighbour.operation_sequence[idx2] = neighbour.operation_sequence[idx2], neighbour.operation_sequence[idx1]
            move_details = ("seq", idx1, idx2)
        elif move_type=="Change_Assignment":
            op_idx = random.randint(0, len(neighbour.worker_machine_sequence) - 1)
            current_val = neighbour.worker_machine_sequence[op_idx]
            new_possible_values = neighbour.options_map[op_idx]
            if len(new_possible_values) > 1:
                choices = [v for v in new_possible_values if v != current_val]
                neighbour.worker_machine_sequence[op_idx] = random.choice(choices)
            else:
                neighbour.worker_machine_sequence[op_idx] = random.choice(new_possible_values)
                
            move_details = ("assign", op_idx, neighbour.worker_machine_sequence[op_idx])

        return neighbour,move_details

    def tabu_search(best_individual, iterations=50, tabu_size=15):    
        current_individual=best_individual.copy()
        best_individual=current_individual.copy()
        tabu_list=[]
        for i in range(iterations):
            candidates=[]
            for j in range(50):
                neighbor_copy,move=TabuLocalSearch.get_neighbour_tabu(current_individual)
                makespan=calculate_fitness(neighbor_copy)
                candidates.append((neighbor_copy,move))

            candidates.sort(key=lambda x: x[0].makespan)

            for neighbor,move in candidates:
                is_better = neighbor.makespan < best_individual.makespan
                if move not in tabu_list or is_better:
                    current_individual=neighbor
                    tabu_list.append(move)
                    if len(tabu_list)>tabu_size:
                        tabu_list.pop(0)
                    if is_better:
                        best_individual=neighbor.copy()
                    break
        return best_individual