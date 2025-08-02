import torch
def tsnet(p_input,pred): #takes as input the processed_input and pred (num_nodes x 2)

    num_nodes = pred.shape[0]
    l_c = 1
    l_kl = 1
    l_r = 1
    r = 0.000001

    #First compute the compression term
    L_c = (torch.linalg.norm(pred)**2)/(2*num_nodes)

    #Compute the KL term
    L_kl = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            euc_dis = torch.linalg.norm(pred[i]-pred[j])
            graph_dis = p_input['metric'][i][j]
            L_kl = L_kl + graph_dis*torch.log(graph_dis/euc_dis)
    
    #Compute the repulsion term
    euc_mat = torch.cdist(pred,pred) #num_nodes x num_nodes
    euc_mat = euc_mat +  r
    euc_mat = torch.log(euc_mat)
    euc_mat = torch.matmul(p_input['time_step'],euc_mat)
    L_r = torch.sum(euc_mat)/(2*num_nodes*num_nodes)

    return l_c*L_c + l_r*L_r + l_kl*L_kl





