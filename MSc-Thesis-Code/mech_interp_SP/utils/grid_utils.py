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

def find_random_hamiltonian_path(G):
    """Finds a random Hamiltonian path in the graph G using backtracking."""
    start_node = random.choice(list(G.nodes()))
    path, visited = [start_node], {start_node}
    def search(current_node):
        if len(path) == len(G.nodes()): return True
        neighbors = list(G.successors(current_node)); random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor); path.append(neighbor)
                if search(neighbor): return True
                path.pop(); visited.remove(neighbor)
        return False
    if search(start_node): return path
    else: return find_random_hamiltonian_path(G)

# def generate_sp_prompt_components(G, context_type=None, min_walk_len=50, max_walk_len=50):
#     """
#     Generates components for a single shortest-path example with more control.

#     Args:
#         context_type (str): "H", "RW", or "Both".
#     """
#     nodes = list(G.nodes())
    
#     # 1. Generate context path based on the specified type
#     if context_type == "H":
#         actual_context_type = "Hamiltonian"
#         context_path_nodes = find_random_hamiltonian_path(G)
#     elif context_type == "RW":
#         actual_context_type = "Random Walk"
#         walk_len = random.randint(min_walk_len, max_walk_len)
#         context_path_nodes = generate_random_walk(G, random.choice(nodes), walk_len)
#     # else: # "Both" - probabilistic choice
#     #     if random.random() < 0.5:
#     #         actual_context_type = "Hamiltonian"
#     #         context_path_nodes = find_random_hamiltonian_path(G)
#     #     else:
#     #         actual_context_type = "Random Walk"
#     #         walk_len = random.randint(min_walk_len, max_walk_len)
#     #         context_path_nodes = generate_random_walk(G, random.choice(nodes), walk_len)

#     context_str = walk_to_string(context_path_nodes, G)

#     # 2. Select start/goal nodes
#     solvable_nodes = list(set(context_path_nodes))
#     if len(solvable_nodes) < 2: return None
#     start_node, goal_node = random.sample(solvable_nodes, 2)
    
#     # 3. Find target path and create its string representation
#     try:
#         target_path_nodes = next(nx.all_shortest_paths(G, source=start_node, target=goal_node))
#     except (nx.NetworkXNoPath, StopIteration): return None
#     target_str = walk_to_string(target_path_nodes, G)
        
#     # 4. Format prompt
#     task_instruction = f"[SHORTEST] [START_NODE] {start_node} [GOAL] {goal_node}"
#     prompt = f"[SOS] {context_str} [SEP] {task_instruction} [PLAN]"

#     return {
#         "prompt": prompt,
#         "context_str": context_str,
#         "context_path_nodes": context_path_nodes,
#         "context_type": actual_context_type,
#         "target_str": target_str,
#         "target_path_nodes": target_path_nodes,
#         "start_node": start_node,
#         "goal_node": goal_node,
#     }




# REPLACE your old generate_sp_prompt_components function with this one.

def generate_sp_prompt_components(G, context_type=None, min_walk_len=50, max_walk_len=50, max_context_attempts=1000):
    """
    Generates components for a single, GUARANTEED SOLVABLE shortest-path example
    using the correct "Geometer" logic (Task first, then Context). [FINAL CORRECTED VERSION]

    Args:
        G (nx.DiGraph): The ground-truth graph.
        context_type (str): "H" (Hamiltonian) or "RW" (Random Walk).
        min_walk_len (int): Minimum length for Random Walk contexts.
        max_walk_len (int): Maximum length for Random Walk contexts.
        max_context_attempts (int): How many times to try generating a valid context before giving up.
    """
    nodes = list(G.nodes())

    # 1. Unbiased Task Selection: Pick the start and goal nodes FIRST from the entire graph.
    start_node, goal_node = random.sample(nodes, 2)
    
    try:
        # 2. Find ALL valid solutions to know what information is required.
        all_solution_paths = list(nx.all_shortest_paths(G, source=start_node, target=goal_node))
        if not all_solution_paths:
            return None # Should not happen on a connected grid, but good practice.
        
        # Determine the set of nodes required to solve the task for ANY of the possible solutions.
        list_of_required_nodes = [set(p) for p in all_solution_paths]
        
    except (nx.NetworkXNoPath, StopIteration):
        return None

    # 3. Context Generation Loop: Find a context that makes the task solvable.
    for _ in range(max_context_attempts):
        # Generate a candidate context path based on the specified type
        if context_type == "H":
            actual_context_type = "Hamiltonian"
            context_path_nodes = find_random_hamiltonian_path(G)
        elif context_type == "RW":
            actual_context_type = "Random Walk"
            walk_len = random.randint(min_walk_len, max_walk_len)
            context_path_nodes = generate_random_walk(G, random.choice(nodes), walk_len)
        else:
            return None # Invalid context_type

        nodes_in_context = set(context_path_nodes)
        
        # 4. The Crucial Filter: Check if the context contains the necessary information.
        is_solvable = any(req_nodes.issubset(nodes_in_context) for req_nodes in list_of_required_nodes)
        
        if is_solvable:
            # 5. Success! We found a valid context. Format and return.
            context_str = walk_to_string(context_path_nodes, G)
            
            # Choose one of the valid paths as the ground truth target.
            # It's important to pick a path that is actually solvable with the given context.
            solvable_paths = [p for p in all_solution_paths if set(p).issubset(nodes_in_context)]
            target_path_nodes = random.choice(solvable_paths)
            target_str = walk_to_string(target_path_nodes, G)
            
            task_instruction = f"[SHORTEST] [START_NODE] {start_node} [GOAL] {goal_node}"
            prompt = f"[SOS] {context_str} [SEP] {task_instruction} [PLAN]"

            return {
                "prompt": prompt,
                "context_str": context_str,
                "context_path_nodes": context_path_nodes,
                "context_type": actual_context_type,
                "target_str": target_str,
                "target_path_nodes": target_path_nodes,
                "start_node": start_node,
                "goal_node": goal_node,
            }

    # If the loop finishes without finding a valid context, return None.
    return None