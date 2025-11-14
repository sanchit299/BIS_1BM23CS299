import numpy as np
import random
import math


grid = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

start = (0, 0)
goal = (5, 4)
num_waypoints = 5   

# -------------------------------------------------
# Function to initialize population (wolves)
# -------------------------------------------------
def initialize_population(pop_size, num_waypoints):
    population = []
    for _ in range(pop_size):
        # Each wolf is a list of randomly placed waypoints
        path = np.random.randint(0, 6, size=(num_waypoints, 2))
        population.append(path)
    return population


# -------------------------------------------------
# Fitness Function (shorter path + avoid obstacles)
# -------------------------------------------------
def evaluate_fitness(path):
    points = [start] + list(path) + [goal]
    total_cost = 0

    for i in range(len(points)-1):
        (x1, y1) = points[i]
        (x2, y2) = points[i+1]

        # Distance cost
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_cost += dist

        # Obstacle penalty
        # Ensure indices are within bounds of the grid
        x2 = min(x2, grid.shape[1] - 1)
        y2 = min(y2, grid.shape[0] - 1)

        if grid[y2][x2] == 1:  # Check for obstacle
            total_cost += 50  # large penalty

    return total_cost


# -------------------------------------------------
# Update wolf position using GWO equations
# -------------------------------------------------
def update_position(wolf, alpha, beta, delta, a):
    new_wolf = np.copy(wolf)

    for i in range(len(wolf)):  # each waypoint
        for j in range(2):  # x,y coordinate
            r1, r2 = random.random(), random.random()

            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = abs(C1 * alpha[i][j] - wolf[i][j])
            X1 = alpha[i][j] - A1 * D_alpha

            r1, r2 = random.random(), random.random()
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = abs(C2 * beta[i][j] - wolf[i][j])
            X2 = beta[i][j] - A2 * D_beta

            r1, r2 = random.random(), random.random()
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = abs(C3 * delta[i][j] - wolf[i][j])
            X3 = delta[i][j] - A3 * D_delta

            new_wolf[i][j] = (X1 + X2 + X3) / 3

        # keep inside grid
        new_wolf[i] = np.clip(new_wolf[i], 0, 5)

    return new_wolf


# -----------------------------
# Grey Wolf Optimization (GWO)
# -----------------------------
pop_size = 20
iterations = 50

wolves = initialize_population(pop_size, num_waypoints)

alpha = beta = delta = None
alpha_f = beta_f = delta_f = float("inf")

for t in range(iterations):
    a = 2 * (1 - t / iterations)  # decreasing coefficient

    for wolf in wolves:
        fitness = evaluate_fitness(wolf)

        if fitness < alpha_f:
            delta, delta_f = beta, beta_f
            beta, beta_f = alpha, alpha_f
            alpha, alpha_f = wolf, fitness

        elif fitness < beta_f:
            delta, delta_f = beta, beta_f
            beta, beta_f = wolf, fitness

        elif fitness < delta_f:
            delta, delta_f = wolf, fitness

    # update all wolves
    new_wolves = []
    for wolf in wolves:
        new_wolves.append(update_position(wolf, alpha, beta, delta, a))
    wolves = new_wolves

# Output the final optimal path and fitness
print("Optimal Path Waypoints:", alpha)
print("Fitness:", alpha_f)
