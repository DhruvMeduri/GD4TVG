
import networkx as nx
import pickle
# Gs = generate_evolve_graph(num_nodes=4, p_remove_node=0.5, p_add_node=0.5)
# draw_graph_ensemble(Gs)

ensemblesize = 1000



if __name__ == "__main__":
    for k in range(ensemblesize):
        # print(i)
        with open('../graphs/graph'+str(k), 'rb') as f:
            Gs = pickle.load(f)
        max_len = len(Gs[0])
        G_0 = nx.relabel_nodes(Gs[0], lambda x: f"0_{x}")
        G_temp = G_0.copy()
        G_combined =G_0.copy()
        for i in range(1,len(Gs)):
            G_new = nx.relabel_nodes(Gs[i], lambda x: f"{i}_{x}")
            max_len = max(len(Gs[i]), max_len)
            for j in range(max_len):
                G_combined = nx.compose(G_combined, G_new)
                if f"{i}_{j}" in G_new and f"{i-1}_{j}" in G_combined:
                    G_combined.add_edge(f"{i}_{j}",f"{i-1}_{j}")
            
            G_temp = G_new.copy()

        numnodes = G_combined.number_of_nodes()
        if numnodes >= 250 :
            print(f"oopsie\t{k}\t{numnodes}")


        filename=f"../graphdad/graph_{k}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(G_combined, f)
        print(f"saved Graph: \t\t{k}")





# Gs.append(G_combined)

# draw_graph_ensemble(Gs)



