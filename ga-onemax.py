import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import time
import numpy as np
import random
import imageio
import sys

class OneMaxIndividual():
    def __init__(self, length=None, chromosome=None, crossover_operator='two-point', mutation_operator='bit-flip'):
        if chromosome is None and not length is None:
            self.chromosome = self.create_random_onemax_chromosome(length)
        elif not chromosome is None:
            self.chromosome = chromosome
        else:
            print('Error: Individual cannot be created without lengh or chromosome to be copied.')
            sys.exit(-1)
        self.fitness = None

        self.crossover = self.get_crossover_operator(crossover_operator)
        self.mutate = self.get_mutation_operator(mutation_operator)

    @staticmethod
    def create_random_onemax_chromosome(length):
        return np.random.randint(2, size=length).tolist()

    def get_crossover_operator(self, crossover_operator='two-point'):
        if crossover_operator == 'two-point':
            return self.two_point_crossover
        if crossover_operator == 'one-point':
            return self.one_point_crossover

    def get_mutation_operator(self, mutation_operator='bit-flip'):
        if mutation_operator == 'bit-flip':
            return self.bit_flip_mutation
        if mutation_operator == 'swap':
            return self.swap_mutation

    def evaluate_fitness(self):
        self.fitness = sum(self.chromosome)

    def show_information(self):
        return 'Fitness: {},\t Chromosome: {}\n'.format(self.fitness, self.chromosome)

    def bit_flip_mutation(self, mutation_probability):
        for i in range(len(self.chromosome)):
            if np.random.rand() < mutation_probability:
                self.chromosome[i] = 0 if self.chromosome[i] == 1 else 1
        self.fitness = None

    def swap_mutation(self, mutation_probability):
        for i in range(len(self.chromosome)):
            if np.random.rand() < mutation_probability:
                j = np.random.randint(len(self.chromosome))
                aux = self.chromosome[i]
                self.chromosome[i] = self.chromosome[j]
                self.chromosome[j] = aux
        self.fitness = None

    def one_point_crossover(self, parent2):
        chromosome_length = len(self.chromosome)
        idx = np.random.randint(chromosome_length)
        offspring1, offspring2 = self.chromosome[:idx]+parent2.chromosome[idx:], parent2.chromosome[:idx]+self.chromosome[idx:]
        return OneMaxIndividual(chromosome_length, offspring1), OneMaxIndividual(chromosome_length, offspring2)


    def two_point_crossover(self, parent2):
        chromosome_length = len(self.chromosome)
        idx1, idx2 = np.random.randint(chromosome_length-1), np.random.randint(chromosome_length)
        if idx1 > idx2:
            aux = idx1
            idx1 = idx2
            idx2 = aux
        offspring1, offspring2 = self.chromosome[:idx1]+parent2.chromosome[idx1:idx2]+self.chromosome[idx2:], \
                                 parent2.chromosome[:idx1]+self.chromosome[idx1:idx2]+parent2.chromosome[idx2:]

        return OneMaxIndividual(chromosome_length, offspring1), OneMaxIndividual(chromosome_length, offspring2)


class OneMaxEA():
    def __init__(self, length, population_size, max_generations, crossover_probability,
                 mutation_probability, crossover_operator='two-point-cx', mutation_operator='bit-flip',
                 selection_operator='tournament-selection', number_of_ancestors=10, tournament_size=5):
        self.individual_length = length
        self.population_size = population_size
        self.population = []
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.selection_operator = self.get_selection_operator(selection_operator)
        self.number_of_ancestors = number_of_ancestors
        self.tournament_size = tournament_size

    def create_population(self):
        pop = []
        for i in range(self.population_size):
            pop.append(OneMaxIndividual(self.individual_length, None, self.crossover_operator, self.mutation_operator))
        self.population = pop

    def create_sub_population(self, length):
        pop = []
        for i in range(length):
            pop.append(OneMaxIndividual(self.individual_length, None, self.crossover_operator, self.mutation_operator))
        return pop

    def get_selection_operator(self, selection_operator='tournament-selection'):
        if selection_operator == 'tournament-selection':
            return self.tournament_selection
        if selection_operator == 'roulette-wheel':
            return self.roulette_wheel_selection

    def get_population_stats(self):
        fitness_list = []
        for ind in self.population:
            fitness_list.append(ind.fitness)
        fitness_list = np.array(fitness_list)
        return fitness_list.max(), fitness_list.mean(), fitness_list.std(), fitness_list.min()

    def show_population_stats(self, generation=None):
        maxx, meann, stdd, minn = self.get_population_stats()
        generation_info = 'Generation={}\t'.format(generation) if not generation is None else ''
        return '{}Fitness information: Max={}, Mean={}, Std={}, Min={}'.format(generation_info, maxx, meann, stdd, minn)

    def show_population(self):
        output = 'Population information:\n'
        for ind in self.population:
            output += ind.show_information()
        return output

    def show_evolution(self, best, mean, worst):
        x = list(range(len(best)))
        plt.figure()
        plt.plot(x, best, 'k', linewidth=2, label='Best fitness')
        plt.plot(x, mean, 'b', linewidth=2, label='Mean fitness')
        plt.plot(x, worst, 'r', linewidth=2, label='Worst fitness')
        plt.title('Fitness evolution')
        plt.ylabel('Fitness')
        plt.xlabel('Number of generations')
        plt.legend()
        plt.ylim(self.individual_length/2, self.individual_length)
        plt.savefig('onemax-{}-{}.png'.format(self.crossover_probability, self.mutation_probability))
        plt.show()

    def evaluate_population(self):
        for ind in self.population:
            if ind.fitness is None: ind.evaluate_fitness()

    def tournament_selection(self, number_of_parents, tournament_size):
        parents = []
        opponents_index = list(range(self.population_size))
        for parent in range(number_of_parents):
            opponents = random.sample(opponents_index, k=tournament_size)
            best_fitness = -1
            for opponent in opponents:
                if self.population[opponent].fitness > best_fitness:
                    best_fitness = self.population[opponent].fitness
                    best_opponent = opponent
            parents.append(OneMaxIndividual(None, self.population[best_opponent].chromosome))
            opponents_index.remove(best_opponent)
        return parents

    def roulette_wheel_selection(self, number_of_parents, _):
        parents = []
        opponents_index = list(range(self.population_size))
        for parent in range(number_of_parents):
            sum_fitness = sum([self.population[idx].fitness for idx in opponents_index])
            pick = random.uniform(0, sum_fitness)
            current_fitness = 0
            for idx in opponents_index:
                current_fitness += self.population[idx].fitness
                if current_fitness > pick:
                    selected_individual = idx
                    break
            parents.append(OneMaxIndividual(None, self.population[selected_individual].chromosome))
            opponents_index.remove(selected_individual)
        return parents

    def evolution(self, show_stats):
        best_fitnesses = []
        mean_fitnesses = []
        worst_fitnesses = []

        self.create_population()
        self.evaluate_population()
        if show_stats:
            print(self.show_population())
            print(self.show_population_stats())

        for generation in range(self.max_generations):
            parents = self.selection_operator(self.number_of_ancestors, self.tournament_size)
            offspring = []

            for i in range(0, self.number_of_ancestors, 2):
                if random.random() < self.crossover_probability:
                    offspring1, offspring2 = parents[i].crossover(parents[i+1])
                    offspring.append(offspring1)
                    offspring.append(offspring2)

            for offspring1 in offspring:
                offspring1.mutate(self.mutation_probability)

            self.population += offspring
            self.evaluate_population()
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            self.population = sorted_pop[:self.population_size]
            if show_stats: print(self.show_population_stats(generation))
            best_fitness, mean_fitness, std, worst_fitness = self.get_population_stats()
            best_fitnesses.append(best_fitness)
            mean_fitnesses.append(mean_fitness)
            worst_fitnesses.append(worst_fitness)

        if show_stats:
            print(self.show_population())
            print(self.show_population_stats())
        self.show_evolution(best_fitnesses, mean_fitnesses, worst_fitnesses)

    def random_search(self, show_stats):
        best_fitnesses = []
        mean_fitnesses = []
        worst_fitnesses = []

        self.create_population()
        self.evaluate_population()
        if show_stats:
            print(self.show_population())
            print(self.show_population_stats())

        for generation in range(self.max_generations):
            offspring = self.create_sub_population(self.number_of_ancestors) # Create random solutions
            self.population += offspring
            self.evaluate_population()
            #sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            #self.population = sorted_pop[:self.population_size]
            if show_stats: print(self.show_population_stats(generation))
            best_fitness, mean_fitness, std, worst_fitness = self.get_population_stats()
            best_fitnesses.append(best_fitness)
            mean_fitnesses.append(mean_fitness)
            worst_fitnesses.append(worst_fitness)

        if show_stats:
            #print(self.show_population())
            print(self.show_population_stats())
        self.show_evolution(best_fitnesses, mean_fitnesses, worst_fitnesses)

individual_length = 100
population_size = 100
number_of_generations = 1000
cx_probability = 0.5
mut_probability = 0.01
crossover_operator='two-point-cx'
mutation_operator='bit-flip'
selection_operator='tournament-selection'
number_of_ancestors=10
tournament_size=5

ea = OneMaxEA(individual_length, population_size, number_of_generations, cx_probability, mut_probability)
#ea.evolution(True)
ea.random_search(True)

