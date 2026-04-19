from __future__ import annotations
from math import sqrt as sqrt
import random
from src.core.instance import Instance
import src.util.evaluation as evaluation

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
      child_genes.makespan, child_genes.worker_balance_fitness = calculate_fitness(child_genes)
      population.append(child_genes)
   return population

def calculate_fitness(instance:Instance):
    m_assignments = [item[0] for item in instance.worker_machine_sequence]
    w_assignments = [item[1] for item in instance.worker_machine_sequence]
    start_times, m_fixed, w_fixed = evaluation.translate(instance.operation_sequence,m_assignments,w_assignments, instance.encoding.durations())
    val_makespan = evaluation.makespan(start_times, m_fixed, w_fixed, instance.encoding.durations())
    val_balance = evaluation.workload_balance(m_fixed, w_fixed, instance.encoding.durations())
    return float(val_makespan), val_balance