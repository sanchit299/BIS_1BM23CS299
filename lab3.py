import numpy as np

class ACO_TSP:
    def __init__(self, coords, n_ants=10, n_iterations=100, alpha=1.0, beta=5.0, rho=0.5, initial_pheromone=1.0):
        self.coords = coords
        self.n_cities = len(coords)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha          # pheromone importance
        self.beta = beta            # heuristic importance
        self.rho = rho              # evaporation rate
        self.initial_pheromone = initial_pheromone
        
        self.distances = self._calculate_distances()
        self.pheromone = np.ones((self.n_cities, self.n_cities)) * self.initial_pheromone
        self.heuristic = 1 / (self.distances + 1e-10)  # avoid division by zero
        
        self.best_tour = None
        self.best_length = np.inf

    def _calculate_distances(self):
        dist = np.zeros((self.n_cities, self.n_cities))
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                dist[i, j] = np.linalg.norm(np.array(self.coords[i]) - np.array(self.coords[j]))
        return dist
    
    def _select_next_city(self, current_city, visited):
        pheromone = self.pheromone[current_city]
        heuristic = self.heuristic[current_city]
        
        prob_numerators = []
        for city in range(self.n_cities):
            if city not in visited:
                val = (pheromone[city] ** self.alpha) * (heuristic[city] ** self.beta)
                prob_numerators.append(val)
            else:
                prob_numerators.append(0)
        
        prob_numerators = np.array(prob_numerators)
        if prob_numerators.sum() == 0:
            # If all probabilities are zero (shouldn't happen), pick a random unvisited city
            candidates = [c for c in range(self.n_cities) if c not in visited]
            return np.random.choice(candidates)
        
        probabilities = prob_numerators / prob_numerators.sum()
        next_city = np.random.choice(range(self.n_cities), p=probabilities)
        return next_city
    
    def _construct_solution(self):
        solutions = []
        lengths = []
        for _ in range(self.n_ants):
            visited = []
            current_city = np.random.randint(0, self.n_cities)
            visited.append(current_city)
            
            while len(visited) < self.n_cities:
                next_city = self._select_next_city(current_city, visited)
                visited.append(next_city)
                current_city = next_city
                
            # Return to start city
            tour_length = self._calculate_tour_length(visited)
            solutions.append(visited)
            lengths.append(tour_length)
        
        return solutions, lengths
    
    def _calculate_tour_length(self, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += self.distances[tour[i], tour[i+1]]
        length += self.distances[tour[-1], tour[0]]  # Return to start
        return length
    
    def _update_pheromones(self, solutions, lengths):
        # Evaporate pheromone
        self.pheromone *= (1 - self.rho)
        
        # Add new pheromone based on solutions quality
        for solution, length in zip(solutions, lengths):
            deposit = 1.0 / length
            for i in range(len(solution) - 1):
                self.pheromone[solution[i], solution[i+1]] += deposit
                self.pheromone[solution[i+1], solution[i]] += deposit
            # Also for return edge
            self.pheromone[solution[-1], solution[0]] += deposit
            self.pheromone[solution[0], solution[-1]] += deposit
    
    def run(self):
        for iteration in range(self.n_iterations):
            solutions, lengths = self._construct_solution()
            self._update_pheromones(solutions, lengths)
            
            min_length_idx = np.argmin(lengths)
            if lengths[min_length_idx] < self.best_length:
                self.best_length = lengths[min_length_idx]
                self.best_tour = solutions[min_length_idx]
            
            print(f"Iteration {iteration+1}/{self.n_iterations}, Best Length: {self.best_length:.4f}")
        
        return self.best_tour, self.best_length

if __name__ == "__main__":
    # Example: Define cities as (x,y) coordinates
    cities = [(0, 0), (1, 5), (5, 2), (6, 6), (8, 3), (7, 7)]
    
    aco = ACO_TSP(cities, n_ants=20, n_iterations=10, alpha=1, beta=5, rho=0.5, initial_pheromone=1)
    best_tour, best_length = aco.run()
    
    print("\nBest tour found:", best_tour)
    print("Best tour length:", best_length)
