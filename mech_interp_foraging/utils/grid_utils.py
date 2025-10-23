import random
import string
import networkx as nx

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_random_names(count):
    """Generate unique random 2-letter names."""
    names = set()
    while len(names) < count:
        names.add(''.join(random.choices(string.ascii_lowercase, k=2)))
    return list(names)

# def get_grid_graph(nodes, size=4):
#     """Create a grid graph with given node names."""
#     if len(nodes) != size * size:
#         raise ValueError(f"Expected {size*size} nodes, got {len(nodes)}")
#     random.shuffle(nodes)
#     G = nx.DiGraph()
#     for r in range(size):
#         for c in range(size):
#             idx = r * size + c
#             u = nodes[idx]
#             if c < size - 1: G.add_edge(u, nodes[idx + 1], direction='EAST')
#             if c > 0: G.add_edge(u, nodes[idx - 1], direction='WEST')
#             if r < size - 1: G.add_edge(u, nodes[idx + size], direction='SOUTH')
#             if r > 0: G.add_edge(u, nodes[idx - size], direction='NORTH')
#     return G

# fixed coordinate assignment for tracking nodes correctly
def get_grid_graph(nodes, size=4):
    """Create a grid graph with given node names. [CORRECTED VERSION]"""
    if len(nodes) != size * size:
        raise ValueError(f"Expected {size*size} nodes, got {len(nodes)}")
    
    random.shuffle(nodes)
    G = nx.DiGraph()

    for r in range(size):
        for c in range(size):
            idx = r * size + c
            G.add_node(nodes[idx])
    
    for r in range(size):
        for c in range(size):
            idx = r * size + c
            u = nodes[idx]
            if c < size - 1: G.add_edge(u, nodes[idx + 1], direction='EAST')
            if c > 0: G.add_edge(u, nodes[idx - 1], direction='WEST')
            if r < size - 1: G.add_edge(u, nodes[idx + size], direction='SOUTH')
            if r > 0: G.add_edge(u, nodes[idx - size], direction='NORTH')
            
    return G

def generate_random_walk(G, start_node, length):
    """Generate a random walk of specified length."""
    path = [start_node]
    current_node = start_node
    for _ in range(length - 1):
        neighbors = list(G.successors(current_node))
        if not neighbors: break
        current_node = random.choice(neighbors)
        path.append(current_node)
    return path

def walk_to_string(walk, G):
    """Convert walk to string format."""
    if not walk or len(walk) < 2:
        return ""
    parts = [f"{walk[i]} {G.edges[walk[i], walk[i+1]]['direction']}" for i in range(len(walk) - 1)]
    parts.append(walk[-1])
    return " ".join(parts)
