import torch
def tsnet(batch,pred): #takes as input the processed_input and pred (num_nodes x 2)

    l_kl = 1
    l_c = 0
    l_r = 1
    batch_loss = 0
    eps = 1e-8


    for c in range(batch.num_graphs):#This is the batch_size
        num_real_nodes = int(batch.num_real_nodes[c])
        #print(num_real_nodes)
        mask = (batch.batch == c)
        #print(mask)
        pred_c = pred[mask][:num_real_nodes]
        time_c = batch.time[mask][:num_real_nodes, :num_real_nodes]
        metric_c = batch.metric[mask][:num_real_nodes, :num_real_nodes]
        #print(pred_c.shape, time_c.shape, metric_c.shape)

        #First compute the compression term
        L_c = (torch.linalg.norm(pred_c)**2)/(2*num_real_nodes)

        #Compute the KL term
        L_kl = 0
        #It is not clear to me, how sigma is chosen. For now:
        sigma = 1
        metric_c = metric_c**2
        metric_c = -metric_c
        metric_c = metric_c/(2*sigma*sigma)
        metric_c = torch.exp(metric_c)
        metric_c.fill_diagonal_(0)
        normalize = metric_c.sum(dim=1, keepdim=True) + eps
        #normalize = normalize - metric_c
        metric_c = metric_c/normalize 
        metric_c = metric_c + metric_c.T
        metric_c = metric_c/(2*num_real_nodes)
        

        euc_mat = torch.cdist(pred_c,pred_c)
        euc_mat = 1 + (euc_mat**2)
        euc_mat = 1/euc_mat
        euc_mat.fill_diagonal_(0) 
        normalize = euc_mat.sum() + eps
        #normalize = normalize - euc_mat
        euc_mat = euc_mat/normalize
        


        for i in range(num_real_nodes):
            for j in range(num_real_nodes):
                if i!=j:
                    euc_dis = euc_mat[i][j]
                    graph_dis = metric_c[i][j]
                    log_term = graph_dis/(euc_dis+eps)
                    log_term = torch.clamp(log_term, min=eps)
                    #print(euc_dis, graph_dis, log_term)
                    L_kl = L_kl + graph_dis*torch.log(log_term)
        
        
        #Compute the repulsion term
        euc_mat = torch.cdist(pred_c,pred_c) #num_nodes x num_nodes
        euc_mat = euc_mat +  eps
        euc_mat = torch.log(euc_mat)
        device = time_c.device
        euc_mat = euc_mat.to(device)#Ensuring both are on the same device
        euc_mat = torch.matmul(time_c,euc_mat)
        L_r = torch.sum(euc_mat)/(2*num_real_nodes*num_real_nodes)


        batch_loss = batch_loss + l_c*L_c + l_r*L_r + l_kl*L_kl
    
    return batch_loss





