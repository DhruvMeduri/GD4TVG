import torch
import torch.optim as optim
from graph_preprocess import generate_input
from graph_preprocess import parse_daddy
from model import TGDModel
from loss import tsnet
import networkx as nx
import pickle
from torch_geometric.loader import DataLoader
import os
import shutil

#Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'
#First we create the training dataset
ensemblesize = 1000
batch_size = 32
max_size = 300
#Setup the input
print("PROCESSING DATA")
input_data = [] #This must be the list of graph-daddies
for i in range(ensemblesize):
    print(i)
    G, is_connected, num_real_nodes = parse_daddy('../graphdad/graph_'+str(i)+'.pkl', max_size)
    if is_connected:
        temp = generate_input(G, num_real_nodes)
        input_data.append(temp)

ensemblesize = len(input_data)
train_loader = DataLoader(input_data, batch_size=batch_size, shuffle=True)

#Foor the model
model = TGDModel(2,2,4)
model.train()
optimizer = optim.Adam(model.parameters(),lr=1e-4)
model.to(device)

save_dir = '../saved_models_1_0_1/'
if(os.path.exists(save_dir)):
    shutil.rmtree(save_dir)
os.mkdir(save_dir)


#Now coming to the training loop
training_loss = []
for epoch in range(50):
    batch_count = 0
    plot_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index)
        loss = tsnet(batch, pred)
        plot_loss = plot_loss + loss.detach()
        print(loss.item())
        loss.backward()
        optimizer.step()
        print("BATCH: ",batch_count, " EPOCH: ",epoch)
        batch_count = batch_count + 1
        batch = batch.to("cpu")

    print("EPOCH: ",epoch," LOSS: ",plot_loss/batch_count)
    training_loss.append(plot_loss/batch_count)
    if epoch%2==0:
        torch.save(model.state_dict(), save_dir + 'model_' + str(epoch) + '.pth')

# Save list to a .pkl file
with open(save_dir + "training_loss.pkl", "wb") as f:
    pickle.dump(training_loss, f)




