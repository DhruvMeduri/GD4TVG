import torch
import torch.optim as optim
from graph_preprocess import generate_input
from graph_preprocess import parse_daddy
from model import TGDModel
from loss import tsnet
import networkx as nx
import pickle

#First we create the training dataset
ensemblesize=1000
#Setup the input
data = [] #This must be the list of graph-daddies
for i in range(ensemblesize):
    G = parse_daddy('../graphdad/graph_'+str(i)+'.pkl')
    data.append(G)


#Foor the model
model = TGDModel(2,2,4)
model.train()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

#Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)

#Now coming to the training loop
training_loss = []
for epoch in range(100):
    for graph in data[:900]:
        p_input = generate_input(graph)
        p_input['features'] = p_input['features'].to(device)
        p_input['edge_index'] = p_input['edge_index'].to(device)
        optimizer.zero_grad()
        pred = model(p_input['features'], p_input['edge_index'])
        loss = tsnet(p_input,pred)
        loss.backward()
        optimizer.step()
        #print("LOSS: ",loss)

    #For computing the loss after every epoch on the validation dataset
    plot_loss = 0
    for graph in data[900:1000]:
        p_input = generate_input(graph)
        p_input['features'] = p_input['features'].to(device)
        p_input['edge_index'] = p_input['edge_index'].to(device)
        pred = model(p_input['features'], p_input['edge_index'])
        plot_loss = plot_loss + tsnet(p_input,pred)
    print("EPOCH: ",epoch," LOSS: ",plot_loss/100)
    training_loss.append(plot_loss/100)

    if epoch%5==0:
        torch.save(model.state_dict(), '../saved_models/model_' + str(epoch) + '.pth')

# Save list to a .pkl file
with open("../saved_models/training_loss.pkl", "wb") as f:
    pickle.dump(training_loss, f)




