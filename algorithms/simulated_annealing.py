import time
import random
import numpy as np


def calculate_distance_matrix(coordinates):
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = np.sqrt(
                (coordinates[i][0] - coordinates[j][0]) ** 2
                + (coordinates[i][1] - coordinates[j][1]) ** 2
            )

    return dist_matrix


def simulated_annealing(
    distance_matrix,
    T,
    TF,
    alpha,
    max_iter,
):
    def calculate_cost(path):
        """
        Calculate the total distance (cost) of a given path.
        """
        return (
            sum(distance_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))
            + distance_matrix[path[-1], path[0]]
        )

    def initial_solution():
        """
        Generate an initial solution using a greedy nearest-neighbor approach.
        """
        n = len(distance_matrix)
        free_nodes = set(range(n))
        current_node = random.choice(list(free_nodes))
        solution = [current_node]
        free_nodes.remove(current_node)

        while free_nodes:
            next_node = min(free_nodes, key=lambda x: distance_matrix[current_node, x])
            solution.append(next_node)
            free_nodes.remove(next_node)
            current_node = next_node
        return solution

    def segment_reversal_move(path):
        """
        Generate a new solution by reversing a segment of the path.
        """
        n = len(path)
        l = random.randint(2, n - 1)
        i = random.randint(0, n - l)
        new_path = path[:i] + path[i : (i + l)][::-1] + path[i + l :]
        return new_path

    n = len(distance_matrix)
    current_path = initial_solution()
    current_cost = calculate_cost(current_path)
    best_path = current_path.copy()
    best_cost = current_cost

    convergence_history = []
    iterations_history = []
    start_time = time.time()

    iterations = 0
    while T >= TF and iterations < max_iter:
        new_path = segment_reversal_move(current_path)
        new_cost = calculate_cost(new_path)

        if new_cost < current_cost or np.random.rand() < np.exp(
            (current_cost - new_cost) / T
        ):
            current_path, current_cost = new_path, new_cost
            if current_cost < best_cost:
                best_path, best_cost = current_path, current_cost

        convergence_history.append(best_cost)
        iterations_history.append(iterations)

        iterations += 1
        T *= alpha

    elapsed_time = time.time() - start_time

    return (best_path, best_cost, convergence_history, iterations_history, elapsed_time)
