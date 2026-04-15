# Generated from: Algorithms.ipynb
# Converted at: 2026-04-13T14:44:26.449Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from math import sqrt as sqrt
import matplotlib.pyplot as plt
import random
import numpy as np 
import sys
import os

sys.path.append(os.path.abspath(os.path.join('..')))

# # Helper Functions


from util.encoding import WorkerEncoding
from util.benchmark_parser import WorkerBenchmarkParser
import random as random
from pathlib import Path

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
    def uniform_crossover(parent_A:Instance,parent_B:Instance,mutation_rate:float):
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
    
    @staticmethod
    def jox_crossover(parent_A:Instance,parent_B:Instance):
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
    def breeding(parent_A:Instance,parent_B:Instance, mutation_rate:float):
        child_operation_sequence=Instance.jox_crossover(parent_A,parent_B)
        child1,child2=Instance.uniform_crossover(parent_A,parent_B,mutation_rate)

        child_worker_operation=child1 if random.random()<0.5 else child2
        
        child=Instance(parent_A.encoding,parent_A.options_map)
        child.operation_sequence=child_operation_sequence
        child.worker_machine_sequence=child_worker_operation
        child.swapping(mutation_rate)

        return child

  

# # SPEA Algorithm


# # Constraints


import util.evaluation as evaluation

def calculate_fitness(instance:Instance):

    m_assignments = [item[0] for item in instance.worker_machine_sequence]

    w_assignments = [item[1] for item in instance.worker_machine_sequence]
   

    start_times, m_fixed, w_fixed = evaluation.translate(

    instance.operation_sequence,m_assignments,w_assignments,

    instance.encoding.durations() 

)
    val_makespan = evaluation.makespan(start_times, m_fixed, w_fixed, instance.encoding.durations())

    val_balance = evaluation.workload_balance(m_fixed, w_fixed, instance.encoding.durations())


    return float(val_makespan), float(np.sum(val_balance))

# # Algorithm


def dominance_function_raw_fitness_combined(population,archive):
   combined=population+archive
   for individual_a in combined:
      individual_a.strength=0
      individual_a.raw_score=0

   for individual_a in combined:
      for individual_b in combined:
         if (individual_a.makespan <= individual_b.makespan and individual_a.worker_balance_fitness <= individual_b.worker_balance_fitness) and (individual_a.makespan < individual_b.makespan or individual_a.worker_balance_fitness < individual_b.worker_balance_fitness):
                individual_a.strength += 1

   for individual_a in combined:
        for individual_b in combined:
            if (individual_b.makespan <= individual_a.makespan and individual_b.worker_balance_fitness <= individual_a.worker_balance_fitness) and \
               (individual_b.makespan < individual_a.makespan or individual_b.worker_balance_fitness < individual_a.worker_balance_fitness):
                individual_a.raw_score += individual_b.strength
   return combined
def euclidean_distance(x1, y1, x2, y2):
   return sqrt(((x1-y1)**2)+((x2-y2)**2))
def density_function(population,archive):
   combined=dominance_function_raw_fitness_combined(population,archive)
   
   max_m = max(individual.makespan for individual in combined)
   min_m = min(individual.makespan for individual in combined)
   max_b = max(individual.worker_balance_fitness for individual in combined)
   min_b = min(individual.worker_balance_fitness for individual in combined)

   range_m = (max_m - min_m) if max_m != min_m else 1
   range_b = (max_b - min_b) if max_b != min_b else 1
   
   k=int(sqrt(len(combined)))
   for individual_a in combined:
      distances=[]
      norm_m_a = (individual_a.makespan - min_m) / range_m
      norm_b_a=(individual_a.worker_balance_fitness-min_b)/range_b
      for individual_b in combined:
         if individual_a==individual_b:
            continue
         norm_m_b = (individual_b.makespan - min_m) / range_m
         norm_b_b=(individual_b.worker_balance_fitness-min_b)/range_b

         distance= euclidean_distance(norm_m_a,norm_b_a,norm_m_b,norm_b_b)
         distances.append(distance)
      distances.sort()
      density=1/(distances[k-1]+2)
      individual_a.final_fitness=individual_a.raw_score+density
      
   return combined
def truncate_archive(archive, archive_size):
   
   while len(archive)>archive_size:
      max_m = max(individual.makespan for individual in archive)
      min_m = min(individual.makespan for individual in archive)
      max_b = max(individual.worker_balance_fitness for individual in archive)
      min_b = min(individual.worker_balance_fitness for individual in archive)

      range_m = (max_m - min_m) if max_m != min_m else 1
      range_b = (max_b - min_b) if max_b != min_b else 1

      neighbour_distances=[]
      for individual_a in archive:
         all_distances=[]
         norm_m_a = (individual_a.makespan - min_m) / range_m
         norm_b_a=(individual_a.worker_balance_fitness-min_b)/range_b
         for individual_b in archive:
            if individual_a==individual_b:
               continue
            norm_m_b = (individual_b.makespan - min_m) / range_m
            norm_b_b=(individual_b.worker_balance_fitness-min_b)/range_b
            distance= euclidean_distance(norm_m_a,norm_b_a,norm_m_b,norm_b_b)
            all_distances.append(distance)
         all_distances.sort()
         neighbour_distances.append(all_distances)
      victim_index = neighbour_distances.index(min(neighbour_distances))
      archive.pop(victim_index)
   return archive      
def environmental_selection(final_population, archive_size):
   new_archive=[individual for individual in final_population if individual.final_fitness<1]
   if len(new_archive)==archive_size:
      return new_archive
   elif len(new_archive)<archive_size:
      dominated_individuals=[individual for individual in final_population if individual.final_fitness>=1]
      dominated_individuals.sort(key=lambda x:x.final_fitness)
      needed_individuals=archive_size-len(new_archive)
      new_archive.extend(dominated_individuals[:needed_individuals])
      return new_archive
   elif len(new_archive)>archive_size:
      return truncate_archive(new_archive,archive_size)
def binary_tournament(archive, population_size,mutation_rate,daredevil_mode):
   population=[]
   while(len(population)<population_size):
      individual_a=random.choice(archive)
      individual_b=random.choice(archive)
      if daredevil_mode and random.random() < 0.3:
            parent_a = random.choice([individual_a, individual_b])
      elif individual_a.final_fitness<individual_b.final_fitness:
         parent_a=individual_a
      else:
         parent_a=individual_b
      individual_c=random.choice(archive)
      individual_d=random.choice(archive)
      if daredevil_mode and random.random() < 0.3:
         parent_b = random.choice([individual_c, individual_d])
      if individual_c.final_fitness<individual_d.final_fitness:
         parent_b=individual_c
      else:
         parent_b=individual_d
      #Breeding where we have uniform crossover,jux crossover and swapping
      child_genes=Instance.breeding(parent_a,parent_b,mutation_rate)
      child_genes.makespan,child_genes.worker_balance_fitness=calculate_fitness(child_genes)
      population.append(child_genes)
   return population

if __name__ == "__main__":
   
    root_path = Path('..').resolve().parent 
    instances_dir = root_path / "instances" / "fjssp-w"
    parser = WorkerBenchmarkParser()
    encoding = parser.parse_benchmark(instances_dir / "0_BehnkeGeiger_42_workers.fjs")

    # Call the static method
    all_options = Instance.create_options(encoding)
    inst=Instance(encoding,all_options)
    print(len(inst.operation_sequence))
    print(len(inst.worker_machine_sequence))
    print(inst.operation_sequence.count(0))
    POP_SIZE = 200
    ARCHIVE_SIZE = 50
    MAX_GENERATIONS = 500
    BASE_MUTATION= 0.1
    TRACKER_LIMIT_MUTATION=50
    TRACKER_LIMIT_NUKE=150
    DAREDEVIL_FLAG=False

    history_best_makespan = []
    history_avg_makespan = []
    print("Initializing population...")
    all_options = Instance.create_options(encoding)
    population = [Instance(encoding, all_options) for _ in range(POP_SIZE)]
    archive = []
    tracker=0
    global_best_makespan = float('inf')

    for ind in population:
        ind.makespan, ind.worker_balance_fitness = calculate_fitness(ind)

    for gen in range(1, MAX_GENERATIONS + 1):
        combined = density_function(population, archive)
        archive = environmental_selection(combined, ARCHIVE_SIZE)
        
        # Track statistics for this generation
        current_best = min(ind.makespan for ind in archive)
        current_avg = sum(ind.makespan for ind in archive) / len(archive)
        
        history_best_makespan.append(current_best)
        history_avg_makespan.append(current_avg)

        SIGNIFICANCE_THRESHOLD = max(5, global_best_makespan * 0.005)
        print(f"Gen {gen:02d} | Best Makespan: {current_best:.1f} | Archive Size: {len(archive)}")

        if global_best_makespan-current_best>SIGNIFICANCE_THRESHOLD:
            global_best_makespan=current_best
            tracker=0
            current_mutation=BASE_MUTATION
        else:
            tracker+=1

        if tracker>TRACKER_LIMIT_NUKE:
            print(f"---Nuking population at generation {gen}---")
            archive.sort(key=lambda x: x.makespan)
            archive = [archive[0]]
            population = [Instance(encoding, all_options) for _ in range(POP_SIZE)]
            for ind in population:
                ind.makespan, ind.worker_balance_fitness = calculate_fitness(ind)
            tracker=0
            current_mutation = BASE_MUTATION
        elif tracker>TRACKER_LIMIT_MUTATION:
            print("---Changing mutation rate---")
            current_mutation = 0.6
        else:
            current_mutation=BASE_MUTATION
        is_stuck=tracker>TRACKER_LIMIT_MUTATION
        population = binary_tournament(archive, POP_SIZE,current_mutation,daredevil_mode=is_stuck)
