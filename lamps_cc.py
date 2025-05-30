# Ce fichier utilise la méthode de coopération co-évolutive en considérant une population de lampes et pas une population de liste de lampes.

import math
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.patches import Circle, Rectangle

import inspyred
import random
import copy

#Constantes
lamps_number   = 5
radius         = 0.3
mutated        = True
seed           = 100
square         = [1, 1]
discretization = 100  # TODO lower discretization here to speed up computation, increase for increased precision
nb_selected_individuals = 2  # on évaluera leur fitness
print("Figures/lampscc_{}_radius_{}_seed_{}_mutation_{}".format(
    lamps_number, radius, seed, mutated))


def global_fitness(lamps, square):
  globalFitness     = 0.0
  individualFitness = [0] * len(lamps)

  # this is a very rough discretization of the space
  discretizationStep = square[0] / discretization
  totalArea = square[0] * discretization * square[1] * discretization

  # compute coverage of the square, going step by step
  coverage = 0.0
  overlap  = 0.0

  for x in np.arange(0.0, square[0], discretizationStep):
    for y in np.arange(0.0, square[1], discretizationStep):
      coveredByLamps = 0
      for l in range(len(lamps)):

        lamp = lamps[l]

        # if the distance between the point and the center of any lamp is less than
        # the radius of the lamps, then the point is lightened up!
        distance = math.sqrt(
            math.pow(lamp[0] - x, 2) + math.pow(lamp[1] - y, 2))
        if distance <= radius:
          coveredByLamps += 1
          individualFitness[l] += 1

      # now, if the point is covered by at least one lamp, the global fitness increases
      if coveredByLamps > 0: coverage += 1
      # but if it is covered by two or more, there's a 'waste' of light here, an overlap
      if coveredByLamps > 1: overlap += 1

  # the global fitness can be computed in different ways
  globalFitness = coverage / totalArea  # just as total coverage by all lamps
  #globalFitness = (coverage - overlap) / totalArea  # or maybe you'd like to also minimize overlap!

  # On ne prend en compte que la fitness globale
  return globalFitness


def mutation(random, candidate, args):
  mut_rate = args.setdefault('mutation_rate', 0.1)
  mean     = args.setdefault('gaussian_mean', 0.0)
  stdev    = args.setdefault('gaussian_stdev', 1.0)
  bounder  = args['_ec'].bounder
  mutant   = copy.copy(candidate)
  for i, m in enumerate(mutant):
    if random.random() < mut_rate:
      mutant[i][0] += random.gauss(mean, stdev)
      mutant[i][1] += random.gauss(mean, stdev)
  mutant = bounder(mutant, args)
  return mutant


def visualize(lamps, radius, square, globalFitness):
  figure = plt.figure()
  ax = figure.add_subplot(111, aspect='equal')

  # matplotlib needs a list of "patches", polygons that it is going to render
  for l in lamps:
    ax.add_patch(Circle((l[0], l[1]), radius=radius, color='b', alpha=0.4))
  ax.add_patch(Rectangle((0, 0), square[0], square[1], color='w', alpha=0.4))

  ax.set_title("Lamp coverage of the arena (global fitness %.2f)" %
               globalFitness)
  plt.savefig("Figures/lampscc_{}_radius_{}_seed_{}_mutation_{}.png".format(
      lamps_number, radius, seed, mutated))


# Inspyred functions
# Cette fonction est une quasi-copie de la fonction evaluateLamp, mais on prend en compte la fitness globale et locale
def evaluateLamps_cc(lamps, radius, square, visualize=False):

  globalFitness     = 0.0
  individualFitness = [0] * len(lamps)
  contribution      = [0] * len(lamps)

  # compute coverage of the square, going step by step
  coverage = 0.0
  overlap  = 0.0

  # this is a very rough discretization of the space
  discretizationStep = square[0] / discretization
  totalArea          = square[0] * discretization * square[1] * discretization

  for x in np.arange(0.0, square[0], discretizationStep):
    for y in np.arange(0.0, square[1], discretizationStep):
      coveredByLamps = 0
      index_lamp_covers_point = 0
      for l in range(len(lamps)):
        lamp = lamps[l]

        # if the distance between the point and the center of any lamp is less than
        # the radius of the lamp, then the point is lightened up!
        distance = math.sqrt(
            math.pow(lamp[0] - x, 2) + math.pow(lamp[1] - y, 2))
        if distance <= radius:
          coveredByLamps += 1
          individualFitness[l] += 1
          index_lamp_covers_point = l

      # now, if the point is covered by at least one lamp, the global fitness increases
      if coveredByLamps > 0: coverage += 1
      # but if it is covered by two or more, there's a 'waste' of light here, an overlap
      if coveredByLamps > 1:
        overlap += 1
      if coveredByLamps == 1:
        # if the point is covered by one lamp, this lamp contributes well to covering the arena
        contribution[index_lamp_covers_point] += 1

  # the global fitness can be computed in different ways
  # globalFitness = coverage / totalArea  # just as total coverage by all lamps
  # globalFitness = (coverage - overlap) / totalArea  # or maybe you'd like to also minimize overlap!

  # if the flag "visualize" is true, let's plot the situation
  if visualize:
    figure = plt.figure()
    ax = figure.add_subplot(111, aspect='equal')

    # matplotlib needs a list of "patches", polygons that it is going to render
    for l in lamps:
      print(l)
      ax.add_patch(Circle((l[0], l[1]), radius=radius, color='b', alpha=0.4))
    ax.add_patch(Rectangle((0, 0), square[0], square[1], color='w', alpha=0.4))

    ax.set_title("Lamp coverage of the arena (fitness %.2f)" % globalFitness)
    plt.savefig("TestTest")
    plt.close(figure)

  # On prend en compte la fitness globale et locale
  fitness = [0] * len(lamps)
  for l in range(len(lamps)):
    #fitness[l] = (individualFitness[l] / totalArea) + (contribution[l] / coverage) * globalFitness  #I modified this'''
    fitness[l] = contribution[l] / totalArea  # la contribution prend en compte le fitness individuel et aussi son interaction avec les autres individus
  return fitness


def evaluateCandidates(candidates, args):
  radius    = args["radius"]
  visualize = args["visualize"]
  square    = args["square"]

  # iterate over all the candidates, run the Weierstrass function, append result to list
  #candidates représente une configuration de lampes
  list_of_fitness_values = evaluateLamps_cc(candidates, radius, square,
                                            visualize)

  return list_of_fitness_values


def generatorLamps(random, args):
  number_of_dimensions = args["number_of_dimensions"]  # the number of dimensions of the problem will be specified later, and put it in "args"
  minimum              = args["minimum"]  # also, the minimum value of each dimension will be specified later in "args"
  maximum              = args["maximum"]  # same goes for the maximum value
  # the individual will be a series of "number_of_dimensions" random values, generated between "minimum" and "maximum"
  individual = [
      random.uniform(minimum, maximum),
      random.uniform(minimum, maximum)
  ]

  return individual


# this main is just here to try the function, and give you an idea of how it works
def main():
  # sides of the square, [width, height]
  #square = [1, 1]
  # radius of the lamps
  #radius = 0.3
  # coordinates of the lamps [ [x1,y1], [x2,y2], [x3,y4], ... ]
  lamps = [[0.3, 0.3], [0.7, 0.7], [0.3, 0.7], [0.7, 0.3]]
  # calling the function; the argument "visualize=True" makes it plot the current situation
  fitness = evaluateLamps_cc(lamps, radius, square, visualize=True)
  print(fitness)

  random_number_generator = random.Random()
  random_number_generator.seed(seed)  # always obtain the same sequence of numbers at every run

  # instantiate the evolutionary algorithm object
  evolutionary_algorithm = inspyred.ec.EvolutionaryComputation(random_number_generator)

  # and now, we specify every part of the evolutionary algorithm
  evolutionary_algorithm.selector = inspyred.ec.selectors.tournament_selection  # by default, tournament selection has tau=2 (two individuals), but it can be modified (see below)

  if mutated:
    evolutionary_algorithm.variator = [
        inspyred.ec.variators.uniform_crossover,
        inspyred.ec.variators.gaussian_mutation
    ]
    # the genetic operators are put in a list, and executed one after the other
  else:
    evolutionary_algorithm.variator = [inspyred.ec.variators.uniform_crossover]

  # Replacer
  evolutionary_algorithm.replacer = inspyred.ec.replacers.truncation_replacement  # "plus" -> "mu+lambda"
  #evolutionary_algorithm.replacer = inspyred.ec.replacers.random_replacement # pour comparer à une remplacement aléatoire

  # Terminator
  evolutionary_algorithm.terminator = inspyred.ec.terminators.evaluation_termination  # the algorithm terminates when a given number of evaluations (see below) is reached

  # Observer
  #evolutionary_algorithm.observer = inspyred.ec.observers.best_observer  # prints best individual to screen
  evolutionary_algorithm.observer = inspyred.ec.observers.stats_observer  # print out population statistics
  #evolutionary_algorithm.observer = inspyred.ec.observers.plot_observer  # plots evolution

  final_population = evolutionary_algorithm.evolve(
      generator          = generatorLamps,  # of course, we need to specify the evaluator
      evaluator          = evaluateCandidates,  # and the corresponding evaluator
      pop_size           = lamps_number,  # size of the population
      num_selected       = 200,  # size of the offspring (children individuals)
      maximize           = True,  # this is a maximization problem
      max_evaluations    = 5000,  # maximum number of evaluations before stopping, used by the terminator

      # all arguments specified below, THAT ARE NOT part of the "evolve" method, will be automatically placed in "args"
      lamps_number         = lamps_number,  # nombre de lampes dans le carré
      number_of_dimensions = 2,  # number of dimensions of the problem, used by "generator_weierstrass"
      minimum              = 0,  # minimum value of each dimension, used by "generator_weierstrass"
      maximum              = 1,  # maximum value of each dimension, used by "generator_weierstrass"
      radius               = radius,  # rayon de la lampe
      square               = [1, 1],  # définit le carré
      visualize            = False,  # Permet d'afficher le carré
  )

  # after the evolution is over, the resulting population is stored in "final_population"; the best individual is on the top
  best_individual = final_population[0]
  print(best_individual.candidate)
  print(final_population[0:5])
  print("The best individual has fitness %.2f" % best_individual.fitness)
  print([elt.candidate for elt in final_population])
  visualize([elt.candidate for elt in final_population], 0.3, [1, 1],
            global_fitness([elt.candidate for elt in final_population],
                           [1, 1]))

  return


if __name__ == "__main__":
  sys.exit(main())
