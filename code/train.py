import torch
import torch.optim as optim
from graph_preprocess import generate_input
from model import TGDModel
from loss import tsnet
#Setup the input
data = [] #This must be the list of graph-daddies

#Foor the model
model = TGDModel(2,2,4)
model.train()
optimizer = optim.Adam(model.paramets(),lr=1e-4)

#Check for GPU
device = torch.device("cuda" if torch.cuda_is_available() else 'cpu')
model.to(device)

#Now coming to the training loop
for epoch in range(100):
    for graph in data:
        p_input = generate_input(graph)
        p_input['features'] = p_input['features'].to(device)
        p_input['edge_index'] = p_input['edge_index'].to(device)
        optimizer.zero_grad()
        pred = model(p_input.features, p_input.edge_index)
        loss = tsnet(p_input,pred)
        loss.backward()
        optimizer.step()
        print(loss)




