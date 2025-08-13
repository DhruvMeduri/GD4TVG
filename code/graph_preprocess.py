#This expects a nx graph_daddy with appropriately weighted edges(based on matching or optimal transport)
#Each node must have 3 input features, id, random feature (as in the paper) and node-timestep

from torch_geometric.utils import from_networkx
import networkx as nx
import torch
import pickle 
import random
from torch_geometric.data import Data   

def generate_input(G_daddy, num_real_nodes):

    data = from_networkx(G_daddy)
    node_features = data.x[:, :-1].float() #Removes the node-timestep
    edge_index = data.edge_index

    #Now apply floyd-warshall
    dist_dic = nx.floyd_warshall(G_daddy,weight='weight')
    nodes = list(G_daddy.nodes)
    num_nodes = len(nodes)

    #This is the graph-distance matrix
    DM = torch.zeros((num_nodes,num_nodes),dtype=torch.float)
    for i,u in enumerate(nodes):
        for j,v in enumerate(nodes):
            DM[i,j] = dist_dic[u][v]

    #Need to have a time_step matrix
    time_matrix = torch.zeros((num_nodes,num_nodes),dtype=torch.float)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if data.x[i][2] == data.x[j][2]: #If both the nodes are in the same time step
                time_matrix[i][j] = 1

    
    return Data(x = node_features, edge_index = edge_index, metric = DM, time = time_matrix, num_real_nodes = torch.tensor([num_real_nodes]))



def parse_daddy(graphname,max_size):
    with open(graphname, 'rb') as f:
        data = pickle.load(f)

    for u, v in data.edges():
        if(u.split('_')[0] != v.split('_')[0]):
            data[u][v]['weight'] = 0

        else:
            data[u][v]['weight']=1

    is_connected = nx.is_connected(data)

    count = 0
    for u in data.nodes():
        data.nodes[u]['x'] = torch.tensor([count,random.randint(1, 1000),float(u.split('_')[0])])
        count = count + 1
    
    for i in range(count,max_size):
        feature = torch.tensor([i, random.randint(1, 1000), -1])
        data.add_node(i, x=feature)
    
    return data, is_connected, count