import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

class TGDModel(torch.nn.Module):
    def __init__(self,input_dim, output_dim, K):
        '''
        K: chebyshev polynomial order
        '''
        super().__init__()
        self.res_blocks = torch.nn.ModuleList() # residual blocks - each block has multiple chebyshev convolutions within it
        #Now I build the architecture exactly as in the paper

        #First the blocks

        block1 = self.res_block_build(input_dim,16,32)
        self.res_blocks.append(block1)

        block2 = self.res_block_build(32,16,32)
        self.res_blocks.append(block2)

        block3 = self.res_block_build(32,16,32)
        self.res_blocks.append(block3)

        block4 = self.res_block_build(32,32,64)
        self.res_blocks.append(block4)

        block5 = self.res_block_build(64,32,64)
        self.res_blocks.append(block5)

        block6 = self.res_block_build(64,32,64)
        self.res_blocks.append(block6)

        block7 = self.res_block_build(64,32,64)
        self.res_blocks.append(block7)

        block8 = self.res_block_build(64,64,128)
        self.res_blocks.append(block8)

        block9 = self.res_block_build(128,64,128)
        self.res_blocks.append(block9)

        block10 = self.res_block_build(128,64,128)
        self.res_blocks.append(block10)

        block11 = self.res_block_build(128,64,128)
        self.res_blocks.append(block11)

        block12 = self.res_block_build(128,64,128)
        self.res_blocks.append(block12)

        block13 = self.res_block_build(128,64,128)
        self.res_blocks.append(block13)

        block14 = self.res_block_build(128,128,128)
        self.res_blocks.append(block14)

        block15 = self.res_block_build(128,128,128)
        self.res_blocks.append(block15)

        block16 = self.res_block_build(128,128,128)
        self.res_blocks.append(block16)

        #There are some layers in the middle which I call res_layers - the residual layers provide shortcut connections - after blocks 1,4,8,14
        self.res_layer1 = ChebConv(input_dim,32,K)
        self.res_layer2 = ChebConv(32,64,K)
        self.res_layer3 = ChebConv(64,128,K)
        self.res_layer4 = ChebConv(128,128,K)

        #Now to define the regression layers - node -wise fully connected layers
        self.reg1 = torch.nn.Linear(128,256)
        self.reg2 = torch.nn.Linear(256,128)
        self.reg3 = torch.nn.Linear(128,64)
        self.reg4 = torch.nn.Linear(64,output_dim)


    def res_block_build(self, input_dim, hidden_dim, output_dim, num_layers = 3, K = 4):
        res_block = torch.nn.ModuleList()
        res_block.append(ChebConv(input_dim,hidden_dim,K)) # first layer    
        for i in range(num_layers-2):
            res_block.append(ChebConv(hidden_dim,hidden_dim,K)) # middle layers except first and last (num_layers-2 )
        res_block.append(ChebConv(hidden_dim,output_dim,K))   # last layer

        return res_block
    
    def forward_block(self,block,input, edge_index, input_res):
        x = input
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
        for layer in block:
            x = layer(x, edge_index,edge_weight)
            x = F.relu(x)
        
        return F.relu(x+input_res) #Ensure the input_res dim is same as output dim
    
    def forward(self,input,edge_index):
        x = input
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)
        count = 0
        for block in self.res_blocks:
            if count == 0:
                x = self.forward_block(block,x,edge_index,self.res_layer1(x,edge_index,edge_weight))
            elif  count == 3:
                x = self.forward_block(block,x,edge_index,self.res_layer2(x,edge_index,edge_weight))
            elif  count == 7:
                x = self.forward_block(block,x,edge_index,self.res_layer3(x,edge_index,edge_weight))
            elif  count == 13:
                x = self.forward_block(block,x,edge_index,self.res_layer4(x,edge_index,edge_weight))
            else:
                x = self.forward_block(block,x,edge_index,x)
            
            count = count + 1
        
        x = self.reg1(x)
        x = F.relu(x)
        x = self.reg2(x)
        x = F.relu(x)
        x = self.reg3(x)
        x = F.relu(x)
        x = self.reg4(x)

        return x
    
#Just testing if the code works
# test_model = TGDModel(2,2,4)
# x = torch.tensor([[1,2],[2,3],[3,4]],dtype=torch.float)
# edge_index = torch.tensor([[0,1,2],[1,2,0]],dtype=torch.long)
# out = test_model(x,edge_index)
# print(out)






