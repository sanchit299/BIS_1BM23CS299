import random

# ------------------------------
# 1. Define Problem Parameters
# ------------------------------

# Warehouse supply capacities
supply = [300, 400]  # W1, W2

# Store demands
demand = [250, 200, 150]  # S1, S2, S3

# Transportation cost matrix (rows = warehouses, columns = stores)
transport_cost = [
    [4, 6, 8],  # From W1
    [5, 4, 3]   # From W2
]

# Inventory cost (per unit stored)
inventory_cost = [0.5, 0.7]  # per unit left in W1, W2

# Shortage penalty (per unmet unit of demand)
shortage_penalty = 10

num_warehouses = len(supply)
num_stores = len(demand)
num_variables = num_warehouses * num_stores  # Decision variables

# ------------------------------
# 2. Define Objective Function
# ------------------------------
def total_cost(shipments):
    """
    shipments: list of length num_warehouses * num_stores
               representing amount shipped from each warehouse to each store
    """
    # Convert 1D vector into 2D matrix [warehouse][store]
    allocation = [
        shipments[i * num_stores:(i + 1) * num_stores]
        for i in range(num_warehouses)
    ]

    # Calculate transport cost
    cost = sum(
        allocation[i][j] * transport_cost[i][j]
        for i in range(num_warehouses)
        for j in range(num_stores)
    )

    # Calculate inventory cost
    for i in range(num_warehouses):
        remaining = supply[i] - sum(allocation[i])
        if remaining > 0:
            cost += remaining * inventory_cost[i]
        else:
            # Penalty if warehouse overships
            cost += abs(remaining) * 20

    # Calculate shortage penalties
    for j in range(num_stores):
        delivered = sum(allocation[i][j] for i in range(num_warehouses))
        shortage = demand[j] - delivered
        if shortage > 0:
            cost += shortage * shortage_penalty

    return cost

# ------------------------------
# 3. PSO Parameters
# ------------------------------
num_particles = 40
max_iterations = 100
w = 0.7
c1 = 1.5
c2 = 1.5

min_val = 0   # No negative shipments
max_val = max(supply)  # Max possible shipment from one warehouse

# ------------------------------
# 4. Initialize Particles
# ------------------------------
particles = [
    [random.uniform(min_val, max_val) for _ in range(num_variables)]
    for _ in range(num_particles)
]
velocities = [
    [random.uniform(-1, 1) for _ in range(num_variables)]
    for _ in range(num_particles)
]
pbest_positions = [p[:] for p in particles]
pbest_values = [total_cost(p) for p in particles]

gbest_value = min(pbest_values)
gbest_position = pbest_positions[pbest_values.index(gbest_value)]

# ------------------------------
# 5. Main PSO Loop
# ------------------------------
for iteration in range(max_iterations):
    for i in range(num_particles):
        fitness = total_cost(particles[i])

        # Update personal and global best
        if fitness < pbest_values[i]:
            pbest_values[i] = fitness
            pbest_positions[i] = particles[i][:]

        if fitness < gbest_value:
            gbest_value = fitness
            gbest_position = particles[i][:]

    # Update velocity and position
    for i in range(num_particles):
        for d in range(num_variables):
            r1 = random.random()
            r2 = random.random()
            velocities[i][d] = (
                w * velocities[i][d]
                + c1 * r1 * (pbest_positions[i][d] - particles[i][d])
                + c2 * r2 * (gbest_position[d] - particles[i][d])
            )
            particles[i][d] += velocities[i][d]
            # Keep within bounds
            particles[i][d] = max(min(particles[i][d], max_val), min_val)

    print(f"Iteration {iteration+1}/{max_iterations} - Best Cost: {gbest_value:.2f}")

# ------------------------------
# 6. Output the Optimal Strategy
# ------------------------------
print("\nOptimization Complete!")
print(f"Minimum Total Cost: {gbest_value:.2f}")

# Convert best position into matrix
best_allocation = [
    gbest_position[i * num_stores:(i + 1) * num_stores]
    for i in range(num_warehouses)
]

print("\nOptimal Shipment Plan (Warehouses → Stores):")
for i in range(num_warehouses):
    print(f"Warehouse {i+1}: {['%.1f' % x for x in best_allocation[i]]}")
