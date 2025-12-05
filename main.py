# Import all the packages
import argparse
import sys
from tqdm import tqdm
import torch

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

#from blobs import *
from sklearn.decomposition import PCA
# import sparse 
# import stats
import math
from torch_geometric.data import Data, Batch, DataLoader


sys.path.append('./src/')

from spectral_clustering_signed import Spectral_clustering_init

#

from SGAAE import SGAAE_
inner=True
from link_prediction import LP_

parser = argparse.ArgumentParser(description='SGAAE')

parser.add_argument('--epochs', type=int, default=3000, metavar='N',
                    help='number of epochs for training (default: 3K)')


parser.add_argument('--cuda', type=eval, 
                      choices=[True, False],  default=False,
                    help='CUDA training')



parser.add_argument('--LP',type=eval, 
                      choices=[True, False], default=True,
                    help='performs link prediction')



parser.add_argument('--D', type=int, default=8, metavar='N',
                    help='dimensionality of the embeddings (default: 8)')

parser.add_argument('--lr', type=float, default=0.005, metavar='N',
                    help='learning rate for the ADAM optimizer, for large values of delta 0.01 is more stable (default: 0.005)')

parser.add_argument('--sample_percentage', type=float, default=0.10, metavar='N',
                    help='Sample size network percentage, it should be equal or less than 1 (default: 0.3)')



parser.add_argument('--dataset', type=str, default='wiki_elec',
                    help='dataset to apply SGAAE on')



args = parser.parse_args()

if  args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')





plt.style.use('ggplot')


torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    latent_dim=args.D
    dataset=args.dataset
       
    print(dataset)
       
    losses=[]
    data=np.loadtxt("./datasets/undirected/"+dataset+'/edges.txt')
    data[:,0:2].sort(1)
    mask=data[:,0]<data[:,1]
    data=data[mask]
    
    sparse_i=torch.from_numpy(data[:,0]).long().to(device)
    # input data, link column positions with i<j
    sparse_j=torch.from_numpy(data[:,1]).long().to(device)
    
    
     
    
    weights_signed=torch.from_numpy(data[:,2]).long().to(device)
    rand_feat=False
    init_dim=32
    method='Normalized_sym'
    # method='Adjacency'
    # method='Normalized'
    
    sign_L=Spectral_clustering_init(num_of_eig=init_dim,method=method)
    
       
       
    
    # network size
    N=int(sparse_j.max()+1)
    
    
    if rand_feat:
        node_f_plus=torch.randn(N,init_dim).to(device)
        node_f_minus=torch.randn(N,init_dim).to(device)
     
    
    else:
    
        node_f_plus = sign_L.spectral_clustering(sparse_i=sparse_i,sparse_j=sparse_j,weights_signed=weights_signed,input_size=N)
        node_f_minus = sign_L.spectral_clustering(sparse_i=sparse_i,sparse_j=sparse_j,weights_signed=-weights_signed,input_size=N)
    
    
    tot_i=torch.cat((sparse_i,sparse_j)).unsqueeze(0)
    tot_j=torch.cat((sparse_j,sparse_i)).unsqueeze(0)
    tot_w=torch.cat((weights_signed,weights_signed))
    sparse_i=tot_i.view(-1)
    sparse_j=tot_j.view(-1)
    weights_signed=tot_w
    
    
    edge_index=torch.cat((tot_i,tot_j),0)
    mask_pos=torch.where(tot_w>0)[0]
    mask_neg=torch.where(tot_w<0)[0]
    edge_index_pos=edge_index[:,mask_pos]
    edge_index_neg=edge_index[:,mask_neg]
    
       
    
    dataset_pgm=Data(x_plus=node_f_plus,x_minus=node_f_minus,edge_index=edge_index,
              edge_index_pos=edge_index_pos,edge_index_neg=edge_index_neg,num_nodes=N).to(device)
    
    
    
    
    
    # sample size of blocks-> sample_size*(sample_size-1)/2 pairs
    VAE=False
    unique_emb=False
    sample_size=int(args.sample_percentage*N)
    model = SGAAE_(init_dim,dataset_pgm,sparse_i,sparse_j,weights_signed,N,latent_dim=latent_dim,sample_size=sample_size,device=device,VAE=VAE).to(device)         
    model.scaling=0
    
    
    # set-up optimizer
    optimizer = optim.Adam(model.parameters(), 0.005)  
    optimizer.zero_grad() # clear the gradients.   
    
    
      
        
   
    for epoch in tqdm(range(args.epochs),desc="Model is Running…",ascii=False, ncols=75):      
        model.train()
        if epoch == 1000:         
            new_lr = 0.001        
            for param_group in optimizer.param_groups:             
                param_group['lr'] = new_lr    
                
        if epoch == 2000:         
              new_lr = 0.0005         
              for param_group in optimizer.param_groups:             
                  param_group['lr'] = new_lr    
                 
                
        
        
       
        loss=model.LSM_likelihood_bias_sample(epoch=epoch)/(model.sample_size**2)#+0.5*(model.R_tot**2).sum()

            
            
        losses.append(loss.item())
        
        optimizer.zero_grad() # clear the gradients.   

     
        loss.backward() # backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step() # update the weights

        
        
        if epoch%100==0:
            with torch.no_grad():
                model.eval()
                     
                try:
                   
                    if model.scaling:
                         
                         pred=LP_(model.latent_z.detach(),model.latent_w.detach(),model.gamma_.detach(),model.delta_.detach(),dataset,sparse_i,sparse_j,weights_signed,device=device,inner=inner)
                    else:
                         pred=LP_(model.latent_z.detach(),model.latent_w.detach(),model.gamma_.detach(),model.delta_.detach(),dataset,sparse_i,sparse_j,weights_signed,device=device,inner=inner)
    
                    # p@n
                    p_n_roc,p_n_pr=pred.pos_neg()
                    # p@z
                    p_z_roc,p_z_pr=pred.pos_zer()
                    # n@z
                    n_z_roc,n_z_pr=pred.neg_zer()
                except:
                    print('Skipping')
    
        
    if args.LP:
        pred=LP_(model.latent_z.detach(),model.latent_w.detach(),model.gamma_.detach(),model.delta_.detach(),dataset,sparse_i,sparse_j,weights_signed,device=device,inner=inner)
        # p@n
        p_n_roc,p_n_pr=pred.pos_neg()
        # p@z
        p_z_roc,p_z_pr=pred.pos_zer()
        # n@z
        n_z_roc,n_z_pr=pred.neg_zer()
    
    torch.save(model.state_dict(),f'./model_{dataset}.pth')      

