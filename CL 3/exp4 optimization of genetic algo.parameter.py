import random
import numpy as np # type: ignore

# Simulated neural network accuracy based on weights
def evaluate_fitness(weights):
    """
    This is a mock fitness evaluation function for simulating the performance of the neural network.
    The goal is to find the weights that minimize the error from the ideal weight of 0.5.
    """
    return 1 / (1 + np.sum(np.square(weights - 0.5)))  # Best fitness when weights ~0.5

# GA parameters
POP_SIZE = 10  # Population size
CHROMO_LENGTH = 5  # Length of the chromosome (number of parameters/weights)
MUTATION_RATE = 0.1  # Probability of mutation
CROSSOVER_RATE = 0.8  # Probability of crossover
GENERATIONS = 15  # Number of generations to run

# Generate initial population
def init_population():
    """
    Initialize the population with random individuals.
    Each individual is a set of random weights.
    """
    return [np.random.rand(CHROMO_LENGTH) for _ in range(POP_SIZE)]

# Crossover two parents
def crossover(p1, p2):
    """
    Perform a two-point crossover between two parents to produce two children.
    """
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, CHROMO_LENGTH - 1)
        child1 = np.concatenate((p1[:point], p2[point:]))
        child2 = np.concatenate((p2[:point], p1[point:]))
        return child1, child2
    return p1, p2

# Mutate a chromosome
def mutate(chromo):
    """
    Mutate an individual's chromosome by changing some of its genes with a certain probability.
    """
    for i in range(CHROMO_LENGTH):
        if random.random() < MUTATION_RATE:
            chromo[i] = random.random()  # Random mutation of the gene
    return chromo

# GA loop
def genetic_algorithm():
    """
    Run the genetic algorithm to optimize the weights for the neural network.
    """
    population = init_population()
    best_fitness = 0
    best_solution = None

    for gen in range(GENERATIONS):
        # Evaluate the fitness of each individual in the population
        fitness_scores = [evaluate_fitness(ind) for ind in population]
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Sort individuals by fitness (descending)
        population = [population[i] for i in sorted_indices]  # Sort population by fitness

        # Track the best solution
        if fitness_scores[sorted_indices[0]] > best_fitness:
            best_fitness = fitness_scores[sorted_indices[0]]
            best_solution = population[0]

        print(f"Generation {gen+1} | Best Fitness: {best_fitness:.4f}")

        # Elitism: Keep the best 2 individuals
        new_population = population[:2]

        # Create new individuals through crossover and mutation
        while len(new_population) < POP_SIZE:
            p1, p2 = random.sample(population[:5], 2)  # Select 2 parents from the top 5 individuals
            c1, c2 = crossover(p1, p2)  # Perform crossover to create two children
            new_population.append(mutate(c1))  # Apply mutation to child 1
            if len(new_population) < POP_SIZE:
                new_population.append(mutate(c2))  # Apply mutation to child 2

        population = new_population  # Set the new population for the next generation

    # Print the best result after all generations
    print("\nOptimized Weights:", np.round(best_solution, 3))
    print("Simulated Coconut Milk Spray Drying Accuracy:", round(best_fitness * 100, 2), "%")

# Run the genetic algorithm
genetic_algorithm()
