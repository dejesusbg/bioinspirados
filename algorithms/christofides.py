import time
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance


def plot_tsp_path(coordinates, tsp_path):
    G = nx.Graph()

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist = distance.euclidean(coordinates[i], coordinates[j])
            G.add_edge(i, j, weight=dist)

    pos = {i: coordinates[i] for i in range(len(coordinates))}

    plt.figure(figsize=(8, 6))

    tsp_edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)] + [
        (tsp_path[-1], tsp_path[0])
    ]

    nx.draw_networkx_edges(G, pos, edgelist=tsp_edges, edge_color="black", width=2)

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=tsp_path,
        node_size=50,
        node_color="red",
        alpha=0.7,
    )

    plt.title("Path - Christofides")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("christofides.png")


def christofides_algorithm(coordinates):
    start_time = time.time()

    G = nx.Graph()

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            dist = distance.euclidean(coordinates[i], coordinates[j])
            G.add_edge(i, j, weight=dist)

    MST = nx.minimum_spanning_tree(G)

    odd_vertices = [v for v in MST.nodes() if MST.degree[v] % 2 == 1]

    subgraph = G.subgraph(odd_vertices)

    matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True)
    for u, v in matching:
        MST.add_edge(u, v, weight=G[u][v]["weight"])

    eulerian_circuit = list(nx.eulerian_circuit(MST))

    tsp_path = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            tsp_path.append(u)
            visited.add(u)

    tsp_cost = sum(
        G[tsp_path[i - 1]][tsp_path[i]]["weight"] for i in range(len(tsp_path))
    )

    elapsed_time = time.time() - start_time

    return (tsp_path, tsp_cost, elapsed_time)
