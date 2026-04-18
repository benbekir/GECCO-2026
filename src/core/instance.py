from src.core.candidate import Candidate
from src.util.encoding import WorkerEncoding
import random
import copy

class Instance:
    def __init__(self,encoding:WorkerEncoding,options_map):
        self.encoding=encoding
        self.options_map=options_map
        self.operation_sequence=list(encoding.job_sequence())
        random.shuffle(self.operation_sequence)
        n_total_ops = encoding.n_operations()
        self.worker_machine_sequence = [None] * n_total_ops
        for op_idx in range(n_total_ops):
            choice=random.choice(self.options_map[op_idx])
            self.worker_machine_sequence[op_idx] = choice
        self.objectives = {"makespan": float('inf'), "balance": float('inf')} 
        self.fitness=0.0
        
    @staticmethod
    def create_options(encoding: WorkerEncoding):
        options_map = {}
        n_ops = encoding.n_operations()
        machines=encoding.get_all_machines_for_all_operations()
        for op_idx in range(n_ops):
            valid_pairs = []
            usable_machines = machines[op_idx]
            
            for m_idx in usable_machines:
                usable_workers = encoding.get_workers_for_operation_on_machine(op_idx, m_idx)
                
                for w_idx in usable_workers:
                    valid_pairs.append((m_idx, w_idx))
                    
            options_map[op_idx] = valid_pairs
        return options_map
    def swapping(self,mutation_rate:float):
        #Here we perform the swapping of the tuples
        for op_index in range(len(self.worker_machine_sequence)):
            if random.random()<mutation_rate:
                new_possible_values=self.options_map[op_index]
                assignment_value=random.choice(new_possible_values)
                self.worker_machine_sequence[op_index]=assignment_value
        #Here we just swap the job IDs
        if random.random()<mutation_rate:
            idx1, idx2 = random.sample(range(len(self.operation_sequence)), 2)
            self.operation_sequence[idx1],self.operation_sequence[idx2]=self.operation_sequence[idx2],self.operation_sequence[idx1]

    @staticmethod
    def uniform_crossover(parent_A:'Instance',parent_B:'Instance',mutation_rate:float):
        child1=[]
        child2=[]
        for i in range(len(parent_A.worker_machine_sequence)):
            if random.random()<0.5:
                child1.append(parent_A.worker_machine_sequence[i])
                child2.append(parent_B.worker_machine_sequence[i])
            else:
                child1.append(parent_B.worker_machine_sequence[i])
                child2.append(parent_A.worker_machine_sequence[i])

        return child1,child2
    def copy(self):
       
        return copy.deepcopy(self)
    
    @staticmethod
    def jox_crossover(parent_A:'Instance',parent_B:'Instance'):
        unique_jobs_set=parent_A.operation_sequence+parent_B.operation_sequence
        unique_jobs_set=set(unique_jobs_set)
        unique_jobs=list(unique_jobs_set)
        
        job_selection=random.sample(unique_jobs,len(unique_jobs)//2)
        child=[None]*len(parent_A.operation_sequence)
        for i in range(len(parent_A.operation_sequence)):
            if parent_A.operation_sequence[i] in job_selection:
                child[i]=parent_A.operation_sequence[i]
        pointer=0
        for i in range(len(child)):
            if child[i]==None:
                while parent_B.operation_sequence[pointer] in job_selection:
                    pointer+=1
               
                child[i]=parent_B.operation_sequence[pointer]
                pointer+=1
        return child
    
    @staticmethod
    def breeding(parent_A:'Instance',parent_B:'Instance', mutation_rate:float):
        child_operation_sequence=Instance.jox_crossover(parent_A,parent_B)
        child1,child2=Instance.uniform_crossover(parent_A,parent_B,mutation_rate)

        child_worker_operation=child1 if random.random()<0.5 else child2
        
        child=Instance(parent_A.encoding,parent_A.options_map)
        child.operation_sequence=child_operation_sequence
        child.worker_machine_sequence=child_worker_operation
        child.swapping(mutation_rate)

        return child
    
    def update_fitness(self):
        """Outsources fitness calculation to the Candidate class logic."""
        # Create a temporary candidate to calculate values
        temp_cand = Candidate.from_sequences(
            self.operation_sequence, 
            self.worker_machine_sequence, 
            self.encoding
        )
        self.makespan = temp_cand.makespan
        self.worker_balance_fitness = temp_cand.get_balance()

    def to_candidate(self) -> Candidate:
        """Converts this genetic instance into a full Candidate object."""
        return Candidate.from_sequences(
            self.operation_sequence, 
            self.worker_machine_sequence, 
            self.encoding
        )