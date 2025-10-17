import numpy as np
import math

# -----------------------------
# Step 1: Define Generator Data
# -----------------------------
# [a, b, c, Pmin, Pmax]
generators = np.array([
    [0.00375, 2.0, 0.0, 50, 200],
    [0.0175, 1.75, 0.0, 20, 100],
    [0.0625, 1.0, 0.0, 30, 120]
])

Pd = 250  # total demand (MW)
Ng = len(generators)

# -----------------------------
# Step 2: Objective and Constraint Functions
# -----------------------------
def total_cost(Pg):
    cost = 0
    for i in range(Ng):
        a, b, c, _, _ = generators[i]
        cost += a * Pg[i] ** 2 + b * Pg[i] + c
    return cost

def check_constraints(Pg):
    penalty = 0
    # Power balance
    P_total = np.sum(Pg)
    penalty += abs(P_total - Pd) * 100
    
    # Generator limits
    for i in range(Ng):
        Pmin, Pmax = generators[i][3], generators[i][4]
        if Pg[i] < Pmin:
            penalty += (Pmin - Pg[i]) ** 2 * 100
        elif Pg[i] > Pmax:
            penalty += (Pg[i] - Pmax) ** 2 * 100
    return penalty

def fitness(Pg):
    return total_cost(Pg) + check_constraints(Pg)

# -----------------------------
# Step 3: Lévy Flight Function
# -----------------------------
def levy_flight(Lambda):
    sigma = (math.gamma(1 + Lambda) * math.sin(math.pi * Lambda / 2) /
             (math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    u = np.random.normal(0, sigma, Ng)
    v = np.random.normal(0, 1, Ng)
    step = u / (np.abs(v) ** (1 / Lambda))
    return step

# -----------------------------
# Step 4: Cuckoo Search Algorithm
# -----------------------------
def cuckoo_search_OPF(n_nests=20, pa=0.25, n_iter=100):
    # Initialize nests randomly within limits
    nests = np.array([np.random.uniform(g[3], g[4], Ng) for g in generators for _ in range(n_nests)])
    nests = nests[:n_nests]
    
    fitness_values = np.array([fitness(n) for n in nests])
    best_idx = np.argmin(fitness_values)
    best = nests[best_idx].copy()
    
    for t in range(n_iter):
        for i in range(n_nests):
            new_nest = nests[i] + 0.01 * levy_flight(1.5) * (nests[i] - best)
            
            # Keep within limits
            for j in range(Ng):
                Pmin, Pmax = generators[j][3], generators[j][4]
                new_nest[j] = np.clip(new_nest[j], Pmin, Pmax)
            
            new_fitness = fitness(new_nest)
            if new_fitness < fitness_values[i]:
                nests[i] = new_nest
                fitness_values[i] = new_fitness
        
        # Abandon fraction of worst nests
        worst = int(pa * n_nests)
        worst_idx = np.argsort(fitness_values)[-worst:]
        for i in worst_idx:
            nests[i] = np.array([np.random.uniform(g[3], g[4]) for g in generators])
            fitness_values[i] = fitness(nests[i])
        
        # Update best
        current_best_idx = np.argmin(fitness_values)
        if fitness_values[current_best_idx] < fitness_values[best_idx]:
            best_idx = current_best_idx
            best = nests[best_idx].copy()
        
        if t % 10 == 0:
            print(f"Iteration {t} | Best Cost: {fitness_values[best_idx]:.4f}")
    
    return best, fitness_values[best_idx]

# -----------------------------
# Step 5: Run the Optimization
# -----------------------------
best_Pg, best_cost = cuckoo_search_OPF()

print("\n✅ Optimal Generation (MW):", best_Pg)
print("💰 Minimum Total Cost: %.4f $" % best_cost)
print("⚖️ Power Balance Check: %.2f MW" % np.sum(best_Pg))
