import numpy as np
import heapq
from collections import defaultdict

def maximize_entropy_with_alpha(graph, probabilities, iterations, alpha):
    """
    Maximizes the entropy of a scalar probability field on a graph using an alpha-weighted redistribution rule.

    Parameters:
    - graph: List of edges (tuples of vertices)
    - probabilities: Dictionary of vertex probabilities {vertex: probability}
    - iterations: Number of iterations to perform
    - alpha: Control factor for probability redistribution

    Returns:
    - List of probability distributions at each iteration
    - List of selected edges at each iteration
    - List of entropies at each iteration
    """
    # Store the results
    probability_distributions = [probabilities.copy()]
    selected_edges_list = []
    entropy_list = [-np.log(sum([p**2 for p in probabilities.values()]))]  # Initial entropy (second Rényi entropy)

    for _ in range(iterations):
        # Compute pairwise differences and create a max-heap
        edge_diffs = []
        for u, v in graph:
            diff = abs(probabilities[u] - probabilities[v])
            heapq.heappush(edge_diffs, (-diff, u, v))  # Negative for max-heap

        # Select a set of mutually disjoint edges
        selected_edges = []
        used_vertices = set()

        while edge_diffs:
            dff, u, v = heapq.heappop(edge_diffs)
            if dff==0.:
                continue
            if u not in used_vertices and v not in used_vertices:
                selected_edges.append((u, v))
                used_vertices.add(u)
                used_vertices.add(v)

        selected_edges_list.append(selected_edges)

        # Redistribute probabilities for the selected edges using alpha-weighted rule
        new_probabilities = probabilities.copy()
        for u, v in selected_edges:
            p_u = probabilities[u]
            p_v = probabilities[v]
            new_probabilities[u] = (1 + alpha) / 2 * p_u + (1 - alpha) / 2 * p_v
            new_probabilities[v] = (1 + alpha) / 2 * p_v + (1 - alpha) / 2 * p_u

        # Update probabilities and compute entropy
        probabilities = new_probabilities
        probability_distributions.append(probabilities.copy())
        entropy_list.append(-np.log(sum([p**2 for p in probabilities.values()])))

    return probability_distributions, selected_edges_list, entropy_list



