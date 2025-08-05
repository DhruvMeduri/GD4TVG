import pandas as pd
import random
import networkx as nx
import pickle

ensemblesize=1000

def generate_evolve_graph(num_nodes=10, num_edges=2, steps=4, p_add_edge=0.5, p_remove_edge=0.4, p_add_node=0.2, p_remove_node=0.4):
    G = nx.barabasi_albert_graph(num_nodes, num_edges)
    graphs = [G.copy()]  # Store all graph snapshots
    new_node_id = num_nodes
    for _ in range(steps):
        new_G = graphs[-1].copy()  # Start from the last graph state
        
        if random.random() < p_add_node:
            new_node = new_node_id
            new_node_id+=1  # Ensure unique node IDs
            new_G.add_node(new_node)
            # Connect the new node to an existing random node
            if len(new_G.nodes) > 1:
                existing_node = random.choice(list(new_G.nodes - {new_node}))
                new_G.add_edge(new_node, existing_node)

        if random.random() < p_remove_node and len(new_G.nodes) > 1:
            node_to_remove = random.choice(list(new_G.nodes))
            temp2 = new_G.copy()
            temp2.remove_node(node_to_remove)

            if(nx.is_connected(temp2)):
                new_G.remove_node(node_to_remove)

        if random.random() < p_add_edge and len(new_G.nodes) > 1:
            u, v = random.sample(list(new_G.nodes), 2)
            new_G.add_edge(u, v)

        if random.random() < p_remove_edge and new_G.number_of_edges() > 1:
            u, v = random.choice(list(new_G.edges))
            temp = new_G.copy()
            temp.remove_edge(u, v)

            if(nx.is_connected(temp)):
                new_G.remove_edge(u, v)

        if not nx.is_connected(new_G):
            break
        graphs.append(new_G.copy())  # Store the updated graph
    
    return graphs  # Return the list of graphs at each time step





def generate_training_data(outputpath="../graphs/", i=0):
    num_nodes = random.randint(5,15)
    # num_nodes=10
    num_edges = random.randint(1,3)
    steps = 10
    graphlist = generate_evolve_graph(num_nodes, num_edges, steps)
    filename = "graph"+str(i)
    filepath = outputpath+filename
    with open(filepath, 'wb') as f:
        pickle.dump(graphlist, f)


if __name__ == "__main__":
    for i in range(ensemblesize):
        generate_training_data(i=i)