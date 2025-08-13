from graph_preprocess import generate_input
from graph_preprocess import parse_daddy
from model import TGDModel
import torch
import networkx as nx
from matplotlib import pyplot as plt
import os
import shutil


#First get a color map
def get_color_map(n=50):
    # Use a combination of several qualitative colormaps
    cmap_list = [plt.cm.tab20, plt.cm.tab20b, plt.cm.tab20c, plt.cm.Paired]
    
    # Generate enough distinct colors
    colors = []
    for cmap in cmap_list:
        for i in range(cmap.N):
            colors.append(cmap(i))
            if len(colors) == n:
                break
        if len(colors) == n:
            break
    
    # Map keys (0 to n-1) to colors
    color_map = {i: colors[i] for i in range(n)}
    return color_map

color_dict = get_color_map(50)

model = TGDModel(2,2,4)
model.load_state_dict(torch.load('../saved_models_0.1_0.1_0.1/model_48.pth'))
model.eval()

test_folder = '../test_0.1_0.1_0.1'
if(os.path.exists(test_folder)):
        shutil.rmtree(test_folder)
os.mkdir(test_folder)

for test_num in range(100):
#test_num = 10
    print(test_num)
    #Create folder 
    test_path = test_folder + '/graph_daddy_'+str(test_num)

    if(os.path.exists(test_path)):
        shutil.rmtree(test_path)
    os.mkdir(test_path)

    G, is_connected, num_real_nodes = parse_daddy('../graphdad/graph_' + str(test_num) + '.pkl',300)
    data = generate_input(G,num_real_nodes)

    pred = model(data.x, data.edge_index)
    #Remove the masked nodes
    nodes = list(G.nodes)
    nodes_to_remove = nodes[num_real_nodes:]
    G.remove_nodes_from(nodes_to_remove)
    pred = pred[:num_real_nodes]
    # print(pred.shape)
    # print(G.nodes())

    #Get num_time_steps
    num_time = max(int(i.split('_')[0]) for i in G.nodes())+1
    graphs = []
    for i in range(num_time):
        g = nx.Graph()
        graphs.append(g)

    for c,node in enumerate(G.nodes()):
        time_step = int(node.split('_')[0])
        graphs[time_step].add_node(node, pos = pred[c])


    for edge in G.edges():
        time_step1 = int(edge[0].split('_')[0])
        time_step2 = int(edge[1].split('_')[0])
        
        if(time_step1==time_step2):
            graphs[time_step1].add_edge(edge[0],edge[1])


    #Render graphs
    for i in range(num_time):
        pos_dic = {}
        colors = []
        for node in graphs[i].nodes():
            pos_dic[node] = tuple(graphs[i].nodes[node]['pos'].detach().numpy())
            colors.append(color_dict[int(node.split('_')[1])])
        # Draw graph using custom positions
        nx.draw(graphs[i], pos=pos_dic, with_labels=True, node_color=colors, edge_color='gray')
        plt.savefig(test_path + '/time_'+str(i)+'.png')
        plt.clf()