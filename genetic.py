import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import brewer2mpl
from deap import base, creator, tools, algorithms
from sklearn.externals import joblib
matplotlib.use('Agg')
bmap = brewer2mpl.get_map('Set2', 'qualitative', 5)
colors = bmap.mpl_colors

data_in = joblib.load('Pickled\\data_pca_reduced.pkl')
# data_in = np.array([[4, 10], [7, 10], [10, 5], [5, 2], [4, 8], [9, 3], [11, 4], [2, 2], [3, 4], [12, 3], [6, 8], [12, 6]])

# Minimum and maximum values found in the dataset.
# These are used as the limits within which
# numbers will be generated.
MIN_VAL = np.amin(data_in)
MAX_VAL = np.amax(data_in)

# Specify the number of clusters desired.
# Find the number of attributes (dimensionality).
NO_CLUSTERS = 16
DIMENSIONS = data_in.shape[1]

# Seed random with this constant to evaluate tests
# with different parameters
# random.seed(1470)

# Probability to sample an individual randomly or from the data
RANDOM_SAMPLE_PB = 1

# Number of individuals to be in any given generation is POP_SIZE
# Number of iterations the evolutionary algorithm must run is N_GEN
POP_SIZE = [20, 50]
N_GEN = [200]

# Evolutionary Algorithm Probability Parameters
# for Crossover and Mutation
CX_PB = 0.8
MUT_PB = 0.07
IND_PB = 0.1

# Number of individuals to retain in Elitist Strategy
BEST_K = {
    20: [1, 2, 4, 4, 4],
    50: [4, 4, 4, 4, 4]
    }

# Generation Frequency at which generation data is saved to disk
FREQ = 10


def createPoint(dimensions, min_val, max_val):
    '''
    Creates a point in [dimensions]-dimensional space,
    which is a potential cluster centroid.
    The point's coordinates are randomly generated
    in the bounded [dimensions]-dimensional cube
    '''
    point = []
    for i in range(dimensions):
        point.append(random.uniform(min_val, max_val))
        # point.append(random.randint(min_val, max_val))
    return point


def choosePoint(data):
    '''
    Chooses a point from the dataset,
    which is a potential cluster centroid
    '''
    dice = random.randint(0, data.shape[0]-1)
    point = list(data[dice])
    return point


def randomPoint(data, dimensions, min_val, max_val):
    '''
    With a probability of RANDOM_SAMPLE_PB,
    returns randomly created points.
    Otherwise, returns a point in the dataset.
    '''
    if random.random() < RANDOM_SAMPLE_PB:
        return createPoint(dimensions, min_val, max_val)
    else:
        return choosePoint(data)


# Fitness Funtion Definition
def findEuclideanDistance(individual, data):
    '''
    Returns the sum of the intra-cluster distances
    for all clusters
    '''
    distanceSum = 0.0
    distance_list = []

    for row in data:
        min = float('inf')
        for point in individual:
            distance = np.linalg.norm(row-point)
            if (distance < min):
                min = distance

        distance_list.append(min)

    distanceSum = sum(distance_list)

    # return 100000000/(1+distanceSum),
    return 1000000/(1+distanceSum),


def mutateIndividual(individual, indpb, min_val, max_val):
    '''
    Base function to mutate an individual
    with attribute's independent probability as indpb
    '''
    for point in individual:
        prob = random.random()
        if (prob <= indpb):
            for index, element in enumerate(point):
                point[index] = random.uniform(min_val, max_val)
    return individual,


def evaluateInvalidIndividuals(pop):
    '''
    Evaluates the individuals with an invalid fitness
    in the population.
    Returns the number of such individuals found.
    '''
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return len(invalid_ind)


def saveEvolutionData(popsize, pop, gen, hof, log, timelist):
    '''
    Helper function that saves the evolution data
    which includes: Population, Generation, Best individuals found
    Logbook, Time spent till this generation, Current state of randomness
    The data is available as:
    Pickled\checkpoint_[pop_size]_[gen].pkl
    '''
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(
        population=pop,
        generation=gen,
        halloffame=hof,
        logbook=log,
        times=timelist,
        rndstate=random.getstate()
        )
    cp_name = "Pickled\\checkpoint_" + str(pop_size) + "_" + str(gen) + ".pkl"
    joblib.dump(cp, cp_name)
    print("Pickled at gen: %s" % (gen))


def plotEvol(ax, best_cluster, data=data_in):
    '''
    Function to test GA against 2-D point set
    Plots the data and best individual at each generation
    '''
    data_x, data_y = zip(*data)
    best_x, best_y = zip(*best_cluster)
    ax.axis([min(data_x)-1, max(data_x)+1, min(data_y)-1, max(data_y)+1])
    ax.plot(data_x, data_y, 'ro', label='data')
    ax.plot(best_x, best_y, 'g^', label='centroid')


def testGA():
    '''
    A test method for two-dimensional data
    Performs Elitist GA
    '''
    pop = toolbox.population(n=POP_SIZE)
    start_gen = 0
    hof = tools.HallOfFame(1)
    logbook = tools.Logbook()

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # fig, ax_array = plt.subplots(2, int(NGEN/2), sharex=True, sharey=True)

    # Performs evolution for (NGEN - start_gen) generations
    for gen in range(start_gen, N_GEN):

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        best_ind = tools.selBest(pop, 1)[0]
        print("\tBest individual and fitness:\n", end="")
        print("\t\t%s: %s" % (best_ind, best_ind.fitness.values[0]))

        # plot the best individual against the data
        # ax = np.ravel(ax_array)[gen]
        # plotEvol(ax, best_ind)

        offspring = toolbox.select(pop, len(pop) - 1)
        offspring = [toolbox.clone(ind) for ind in offspring]
        offspring.append(toolbox.clone(best_ind))
        random.shuffle(offspring)

        # Apply crossover on the offspring
        # print("\tApplying crossover:")
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CX_PB:
                # Replace the old individuals, only
                # when the new_generation is healthier than the old one
                orig_max = max(ind1.fitness.values[0], ind2.fitness.values[0])

                child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
                toolbox.mate(child1, child2)

                child_list = [child1, child2]
                child_fit = toolbox.map(toolbox.evaluate, child_list)
                for child, fit in zip(child_list, child_fit):
                    child.fitness.values = fit
                child_max = max([child.fitness.values[0] for child in child_list])

                if (child_max > orig_max):
                    # print("\t\t\tParents %s, %s " % (ind1, ind2), end="")
                    # print("replaced with Children %s, %s " % (child1, child2), end="")
                    # print("with fitness: %s" % (child_max))
                    ind1 = child1
                    ind2 = child2

        # Apply mutation on the offspring
        # print("\tApplying mutation:")
        for ind in offspring:
            if random.random() < MUT_PB:
                mutant = toolbox.clone(ind)
                toolbox.mutate(mutant)
                mutant_fit = toolbox.evaluate(mutant)
                mutant.fitness.values = mutant_fit

                if mutant_fit[0] > ind.fitness.values[0]:
                    # print("\t\t\tParent replaced with Mutant with fitness: %s" % (mutant_fit[0]))
                    ind = mutant

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Update Hall of Fame, Stats, and Log for the current generation
        hof.update(pop)
        record = stats.compile(pop)
        print("\t%s" % (record))
        logbook.record(gen=gen, evals=len(invalid_ind), **record)

    # plt.show()
    return pop, logbook, hof


def GA(pop_size, n_gen, best_k, mode=0, checkpoint=None):
    '''
    Performs Evolutionary algorithm on either:
    1. A random population, if checkpoint is not specified,
    2. From the population and generation as specified in
       checkpoint

    The algorithm can work in 3 modes -
    Mode 0: Simple GA (Evaluate, Select, Reproduce/Mutate) without constraints
    Mode 1: Replacement GA, where the children replace the parents
            in the new generation only if they are fitter.
    Mode 2: Elitist Replacement GA, where in addition to the constraints
            of Replacement GA, the best k individuals of the previous
            generation is retained in the current generation

    A checkpoint is created at every 5th generation,
    to increase robustness.
    '''
    if checkpoint:
        # A file name has been given, then load the data from the file
        cp = joblib.load(checkpoint)
        pop = cp["population"]
        start_gen = cp["generation"] + 1
        hof = cp["halloffame"]
        logbook = cp["logbook"]
        time_list = cp["times"]
        random.setstate(cp["rndstate"])
        print('Evolving from specified checkpoint: %s' % (checkpoint))
    else:
        # Start a new population
        pop = toolbox.population(n=pop_size)
        start_gen = 0
        hof = tools.HallOfFame(1)
        logbook = tools.Logbook()
        time_list = []

    print("For pop size: %s, and no of gen: %s" % (pop_size, n_gen))

    # Create the statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    print('gen\tmax\t\tavg\t\tmin')

    # Performs evolution for (n_gen - start_gen) generations
    for gen in range(start_gen, n_gen):
        start_time = time.process_time()

        # Evaluate the individuals with an invalid fitness
        n_evals = evaluateInvalidIndividuals(pop)

        # Mode: 0 implies Simple GA
        if mode == 0:
            # Select individuals from the old population
            # based on Roulette Wheel Selection
            pop = toolbox.select(pop, len(pop))

            # Perform Crossover and Mutation
            pop = algorithms.varAnd(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB)

            # Update individuals' fitness values in the new population
            n_evals = evaluateInvalidIndividuals(pop)

        # Mode: 1 implies GA with replacement of parents
        elif mode == 1:
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            random.shuffle(offspring)

            # Apply crossover on the offspring
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CX_PB:
                    # Replace the old individuals, only
                    # when the new_generation is healthier than the old one
                    orig_max = max(ind1.fitness.values[0], ind2.fitness.values[0])

                    child1, child2 = [toolbox.clone(ind) for ind in (ind1, ind2)]
                    toolbox.mate(child1, child2)

                    child_list = [child1, child2]
                    child_fit = toolbox.map(toolbox.evaluate, child_list)
                    for child, fit in zip(child_list, child_fit):
                        child.fitness.values = fit
                    child_max = max([child.fitness.values[0] for child in child_list])

                    if (child_max > orig_max):
                        # print("\t\t\tParents %s, %s " % (ind1, ind2)),
                        # print("replaced with Children %s, %s " % (child1, child2)),
                        # print("with fitness: %s" % (child_max))
                        ind1 = child1
                        ind2 = child2

            # Apply mutation on the offspring
            for ind in offspring:
                if random.random() < MUT_PB:
                    mutant = toolbox.clone(ind)
                    toolbox.mutate(mutant)
                    mutant_fit = toolbox.evaluate(mutant)
                    mutant.fitness.values = mutant_fit
                    # print("\t\tParent's fitness: %s, " % (ind.fitness.values)),
                    # print(Mutant's fitness: %s" % (mutant.fitness.values))
                    if mutant_fit[0] > ind.fitness.values[0]:
                        # print("\t\t\tParents replaced with Mutant "),
                        # print("with fitness: %s" % (temp_fit[0]))
                        ind = mutant

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        # Mode: 2 implies GA with replacement of parents and an elitist strategy
        elif mode == 2:
            # Elitist approach retains the best K candidate
            best_ind = tools.selBest(pop, best_k)
            best_ind = list(map(toolbox.clone, best_ind))
            offspring = toolbox.select(pop, len(pop) - best_k)
            offspring = list(map(toolbox.clone, offspring))
            for ind in best_ind:
                offspring.append(ind)

            random.shuffle(offspring)
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=CX_PB, mutpb=MUT_PB)
            n_evals = evaluateInvalidIndividuals(offspring)

            pop[:] = offspring

        time_diff = time.process_time() - start_time
        time_list.append(time_diff)

        # Update Hall of Fame, Stats, and Log for the current generation
        hof.update(pop)
        record = stats.compile(pop)
        print("%s\t%s\t%s\t%s" % (gen, record['max'], record['avg'], record['min']))
        logbook.record(gen=gen, evals=n_evals, **record)

        # Pickle the generation data at every FREQth generation
        if gen % FREQ == 0:
            saveEvolutionData(pop_size, pop, gen, hof, logbook, time_list)

    saveEvolutionData(pop_size, pop, gen, hof, logbook, time_list)
    return pop, logbook, hof, time_list


# Define Individual's type and Fitness type and Objective specification
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Register associated functions as aliases in the toolbox
# These are used to create, select, reproduce and mutate individuals
# Also, register the fitness function associated with the individual
toolbox = base.Toolbox()
toolbox.register('create_point', createPoint, dimensions=DIMENSIONS, min_val=MIN_VAL, max_val=MAX_VAL)
toolbox.register('choose_point', choosePoint, data=data_in)
toolbox.register('random_point', randomPoint, data=data_in, dimensions=DIMENSIONS, min_val=MIN_VAL, max_val=MAX_VAL)
#Individuals are created by randomPoint function NO_CLUSTERS time
toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.random_point, n=NO_CLUSTERS)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', findEuclideanDistance, data=data_in)
# Single point crossover is registered as the crossover operator
toolbox.register('mate', tools.cxOnePoint)
toolbox.register("mutate", mutateIndividual, indpb=IND_PB, min_val=MIN_VAL, max_val=MAX_VAL)
# Standard Roulette Selection is registered as the selection operator
toolbox.register('select', tools.selRoulette)

if __name__ == "__main__":
    for pop_size in POP_SIZE:
        if pop_size == 20:
            continue
        for i_gen, n_gen in enumerate(N_GEN):

            if i_gen == 0:
                last_cp = None
            else:
                last_cp = N_GEN[i_gen-1] - 1
                last_cp = "Pickled\\checkpoint_" + str(pop_size) + "_" + str(last_cp) + ".pkl"

            best_k = BEST_K[pop_size][i_gen]

            pop, log, hof, time_gen = GA(pop_size, n_gen, best_k, mode=2, checkpoint=last_cp)
            cum_time = np.cumsum(time_gen)
            # pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.75, mutpb=0.05, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)
            print("\nBest individual has fitness: %s" % (hof[0].fitness.values[0]))
            gen, avg, min_, max_ = log.select("gen", "avg", "min", "max")

            # Plotting 2-D test points
            # x, y = zip(*hof[0])
            # plt.plot(data_in[:, 0], data_in[:, 1], 'ro')
            # plt.plot(x, y, 'g^')
            # plt.axis([min(data_in[:, 0])-3, max(data_in[:, 0])+3, min(data_in[:, 1])-3, max(data_in[:, 1])+3])
            # plt.show()
            plt.clf()
            plt.figure(figsize=(16, 8))
            plt.tick_params(axis='both', which='major', labelsize=8)
            plt.tick_params(axis='both', which='minor', labelsize=6)

            plt.subplot(121)
            plt.plot(gen, min_, linewidth=2, label="minimum", linestyle=':', color=colors[2])
            plt.plot(gen, avg, linewidth=2, label="average", linestyle='--', color=colors[1])
            plt.plot(gen, max_, linewidth=2, label="maximum", linestyle='-', color=colors[0])
            plt.xlabel("Generation", fontsize=8)
            plt.ylabel("Fitness", fontsize=8)
            plt.legend(loc="lower right", framealpha=0.5, prop={'size': 8})

            plt.subplot(122)
            plt.plot(gen, time_gen, linewidth=2, label="Per Generation", color=colors[3])
            plt.plot(gen, cum_time, linewidth=2, label="Cumulative", color=colors[4])
            plt.xlabel("Generation", fontsize=8)
            plt.ylabel("Time spent (s)", fontsize=8)
            plt.legend(loc="lower right", framealpha=0.5, prop={'size': 8})

            plt.tight_layout()
            plt.savefig("Plots\\ga_" + str(pop_size) + "_" + str(n_gen) + ".png")
            plt.close()
