import random

# Objective function
def objective_function(x):
    return x ** 2

# Parameters
POP_SIZE = 100
GENES = 16  # Number of bits to represent x
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
GENERATIONS = 100

# Convert binary string to float in [0, 1]
def decode(individual):
    decimal = int(individual, 2)
    return decimal / (2**GENES - 1)

# Create initial population
def create_population():
    return [''.join(random.choice('01') for _ in range(GENES)) for _ in range(POP_SIZE)]

# Evaluate fitness
def evaluate(population):
    return [objective_function(decode(ind)) for ind in population]

# Selection (roulette wheel)
def select(population, fitness):
    total_fit = sum(fitness)
    probs = [f / total_fit for f in fitness]
    return random.choices(population, weights=probs, k=POP_SIZE)

# Crossover
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        point = random.randint(1, GENES - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2

# Mutation
def mutate(individual):
    return ''.join(
        bit if random.random() > MUTATION_RATE else str(1 - int(bit))
        for bit in individual
    )

# Genetic Algorithm
def genetic_algorithm():
    population = create_population()
    best_solution = None
    best_fitness = float('-inf')

    for gen in range(GENERATIONS):
        fitness = evaluate(population)
        max_fit = max(fitness)
        if max_fit > best_fitness:
            best_fitness = max_fit
            best_solution = population[fitness.index(max_fit)]

        selected = select(population, fitness)
        next_gen = []

        for i in range(0, POP_SIZE, 2):
            p1, p2 = selected[i], selected[i+1]
            c1, c2 = crossover(p1, p2)
            next_gen.extend([mutate(c1), mutate(c2)])

        population = next_gen

        if gen % 10 == 0:
            print(f"Generation {gen}: Best Fitness = {best_fitness:.5f}")

    x_best = decode(best_solution)
    print(f"\nBest solution: x = {x_best:.5f}, f(x) = {objective_function(x_best):.5f}")

genetic_algorithm()
