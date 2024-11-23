import time
import numpy as np


def ant_colony_optimization(
    distance_matrix,
    nants,
    alpha,
    beta,
    rho,
    max_iter,
):
    n = len(distance_matrix)
    pheromone = np.ones((n, n))

    def calculate_probabilities(path, unvisited):
        """
        Calculate the probabilities of each possible next node given the current path
        and unvisited nodes.
        """
        pheromone_weights = np.array([pheromone[path[-1], i] for i in unvisited])
        distance_weights = np.array(
            [1 / distance_matrix[path[-1], i] for i in unvisited]
        )
        probabilities = (pheromone_weights**alpha) * (distance_weights**beta)
        return probabilities / probabilities.sum()

    def construct_solution():
        """
        Construct a solution by randomly selecting the next city at each step
        according to the probabilities given by the pheromone and distance weights.
        """
        path = [np.random.randint(0, n)]
        unvisited = set(range(n)) - {path[0]}
        while unvisited:
            probabilities = calculate_probabilities(path, unvisited)
            next_city = np.random.choice(list(unvisited), p=probabilities)
            path.append(next_city)
            unvisited.remove(next_city)
        return path

    def update_pheromone(paths, costs):
        """
        Update the pheromone levels on the paths based on the quality of the
        solutions found. The pheromone decay is applied to all paths, and additional
        pheromone is deposited proportionally to the inverse of the path cost.
        """
        nonlocal pheromone
        pheromone *= 1 - rho
        for path, cost in zip(paths, costs):
            for i in range(len(path)):
                pheromone[path[i - 1], path[i]] += 1 / cost

    best_path = None
    best_cost = float("inf")

    convergence_history = []
    iterations_history = []
    start_time = time.time()

    for iteration in range(max_iter):
        paths = [construct_solution() for _ in range(nants)]
        costs = [
            sum(distance_matrix[paths[k][i], paths[k][i + 1]] for i in range(n - 1))
            + distance_matrix[paths[k][-1], paths[k][0]]
            for k in range(nants)
        ]

        best_iter_cost = min(costs)
        best_iter_path = paths[np.argmin(costs)]

        if best_iter_cost < best_cost:
            best_path, best_cost = best_iter_path, best_iter_cost

        convergence_history.append(best_cost)
        iterations_history.append(iteration + 1)

        update_pheromone(paths, costs)

    elapsed_time = time.time() - start_time

    return (best_path, best_cost, convergence_history, iterations_history, elapsed_time)
