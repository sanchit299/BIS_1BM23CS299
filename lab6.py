import numpy as np

# --- Objective Function (Rastrigin) ---
def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# --- Parameters ---
grid_size = (10, 10)          # 10x10 grid
dim = 2                       # 2D optimization problem
num_iterations = 100
alpha = 0.5                   # influence of best neighbor
mutation_rate = 0.1
search_range = (-5.12, 5.12)

# --- Initialization ---
population = np.random.uniform(search_range[0], search_range[1], (grid_size[0], grid_size[1], dim))
fitness = np.zeros(grid_size)

def evaluate_population(pop):
    fit = np.zeros(grid_size)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            fit[i, j] = rastrigin(pop[i, j])
    return fit

def get_neighbors(i, j):
    # Moore neighborhood (8 neighbors)
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni, nj = (i + di) % grid_size[0], (j + dj) % grid_size[1]  # wrap-around (toroidal grid)
            neighbors.append((ni, nj))
    return neighbors

# --- Evaluate initial population ---
fitness = evaluate_population(population)
best_global = population[np.unravel_index(np.argmin(fitness), fitness.shape)]
best_global_value = rastrigin(best_global)

# --- Main Loop ---
for iteration in range(num_iterations):
    new_population = np.copy(population)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            neighbors = get_neighbors(i, j)
            neighbor_solutions = [population[n] for n in neighbors]
            neighbor_fitness = [fitness[n] for n in neighbors]
            
            # Best neighbor
            best_neighbor = neighbor_solutions[np.argmin(neighbor_fitness)]
            
            # Update rule
            new_solution = population[i, j] + alpha * (best_neighbor - population[i, j])
            new_solution += mutation_rate * np.random.uniform(-1, 1, dim)
            
            # Clip to search range
            new_solution = np.clip(new_solution, search_range[0], search_range[1])
            new_population[i, j] = new_solution

    # Evaluate new population
    new_fitness = evaluate_population(new_population)
    
    # Replace if improved
    improved = new_fitness < fitness
    population[improved] = new_population[improved]
    fitness[improved] = new_fitness[improved]
    
    # Track global best
    current_best = population[np.unravel_index(np.argmin(fitness), fitness.shape)]
    current_best_value = rastrigin(current_best)
    if current_best_value < best_global_value:
        best_global_value = current_best_value
        best_global = current_best
    
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Best Value = {best_global_value:.6f}")

# --- Output Best Solution ---
print("Sarim 1BM23CS304")
print("\nBest Solution Found:")
print("x =", best_global)
print("f(x) =", best_global_value)
