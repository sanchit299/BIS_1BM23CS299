import numpy as np

# -----------------------------------------------------------
# Step 1: Define the objective function to minimize
# -----------------------------------------------------------
# Example: Sphere function (you can replace this with your own)
def objective_function(x):
    return np.sum(x ** 2)  # minimize sum of squares


# -----------------------------------------------------------
# Step 2: Define the GWO Algorithm
# -----------------------------------------------------------
def grey_wolf_optimizer(obj_func, dim, bounds, num_wolves=20, max_iter=100):
    # Initialize the population of grey wolves Xi
    lower_bound, upper_bound = bounds
    wolves = np.random.uniform(lower_bound, upper_bound, (num_wolves, dim))

    # Initialize alpha, beta, delta wolves
    alpha_pos = np.zeros(dim)
    beta_pos = np.zeros(dim)
    delta_pos = np.zeros(dim)

    alpha_score = float("inf")
    beta_score = float("inf")
    delta_score = float("inf")

    # Main loop
    for t in range(max_iter):
        # Evaluate fitness of each wolf
        for i in range(num_wolves):
            # Boundary check
            wolves[i] = np.clip(wolves[i], lower_bound, upper_bound)

            # Fitness calculation
            fitness = obj_func(wolves[i])

            # Identify alpha, beta, delta
            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = alpha_score
                beta_pos = alpha_pos.copy()

                alpha_score = fitness
                alpha_pos = wolves[i].copy()

            elif fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()

                beta_score = fitness
                beta_pos = wolves[i].copy()

            elif fitness < delta_score:
                delta_score = fitness
                delta_pos = wolves[i].copy()

        # Update control parameter 'a'
        a = 2 - 2 * (t / max_iter)

        # Update the position of each wolf
        for i in range(num_wolves):
            for j in range(dim):
                r1, r2 = np.random.rand(), np.random.rand()

                # Compute for alpha wolf
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i][j])
                X1 = alpha_pos[j] - A1 * D_alpha

                # Compute for beta wolf
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i][j])
                X2 = beta_pos[j] - A2 * D_beta

                # Compute for delta wolf
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i][j])
                X3 = delta_pos[j] - A3 * D_delta

                # Update position
                wolves[i][j] = (X1 + X2 + X3) / 3

        # Display progress
        if t % 10 == 0 or t == max_iter - 1:
            print(f"Iteration {t+1}/{max_iter} | Best Fitness: {alpha_score:.6f}")

    # Return best solution
    return alpha_pos, alpha_score


# -----------------------------------------------------------
# Step 3: Run GWO
# -----------------------------------------------------------
dim = 5                      # number of variables
bounds = (-10, 10)           # lower and upper bounds
num_wolves = 30              # population size
max_iter = 100               # iterations

best_pos, best_score = grey_wolf_optimizer(objective_function, dim, bounds, num_wolves, max_iter)

print("\n✅ Best Position (α wolf):", best_pos)
print("💰 Best Fitness Value:", best_score)
