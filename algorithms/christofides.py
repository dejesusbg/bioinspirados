import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def plot_solution(coordinates, tour):
    coordinates = np.array(coordinates)

    plt.figure(figsize=(10, 6))
    plt.plot(coordinates[:, 0], coordinates[:, 1], "ro")
    for i in range(len(tour) - 1):
        plt.plot(
            [coordinates[tour[i]][0], coordinates[tour[i + 1]][0]],
            [coordinates[tour[i]][1], coordinates[tour[i + 1]][1]],
            "b-",
        )
    plt.plot(
        [coordinates[tour[-1]][0], coordinates[tour[0]][0]],
        [coordinates[tour[-1]][1], coordinates[tour[0]][1]],
        "b-",
    )
    plt.title("Path - Christofides")
    plt.grid()
    plt.tight_layout()
    plt.savefig("christofides.png")
    plt.show()


def christofides_algorithm(coordinates, distance_matrix):
    def create_graph(distance_matrix):
        G = nx.Graph()
        num_nodes = len(distance_matrix)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                G.add_edge(i, j, weight=distance_matrix[i][j])
        return G

    def minimum_spanning_tree(graph):
        mst = nx.minimum_spanning_tree(graph)
        return mst

    def find_odd_degree_nodes(mst):
        odd_nodes = [node for node in mst.nodes() if mst.degree(node) % 2 == 1]
        return odd_nodes

    def minimum_weight_perfect_matching(graph, odd_nodes):
        matching_graph = graph.subgraph(odd_nodes)
        perfect_matching = nx.algorithms.matching.min_weight_matching(matching_graph)
        return perfect_matching

    def eulerian_circuit(graph):
        return list(nx.eulerian_circuit(graph))

    def create_tour(eulerian_circuit, start_node):
        tour = []
        visited = set()
        for node in eulerian_circuit:
            if node[0] not in visited:
                tour.append(node[0])
                visited.add(node[0])
        tour.append(start_node)
        return tour

    def calculate_tour_length(coordinates, tour):
        length = 0
        for i in range(len(tour) - 1):
            length += np.linalg.norm(
                np.array(coordinates[tour[i]]) - np.array(coordinates[tour[i + 1]])
            )
        length += np.linalg.norm(
            np.array(coordinates[tour[-1]]) - np.array(coordinates[tour[0]])
        )
        return length

    start_time = time.time()

    graph = create_graph(distance_matrix)
    mst = minimum_spanning_tree(graph)
    odd_nodes = find_odd_degree_nodes(mst)
    matching = minimum_weight_perfect_matching(graph, odd_nodes)

    multigraph = nx.MultiGraph(mst)
    for u, v in matching:
        multigraph.add_edge(u, v)

    circuit = eulerian_circuit(multigraph)

    start_node = circuit[0][0]
    tsp_path = create_tour(circuit, start_node)
    tsp_cost = float(calculate_tour_length(coordinates, tsp_path))
    elapsed_time = time.time() - start_time

    return tsp_path, tsp_cost, elapsed_time
