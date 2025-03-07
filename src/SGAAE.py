
# Import all the packages
import torch
import torch.nn as nn
import numpy as np
from spectral_clustering_signed import Spectral_clustering_init

# import stats
import math
from torch_sparse import spspmm

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from numpy.matlib import repmat
from Encoder import GCN

from torch_geometric.nn import global_add_pool, global_mean_pool



  


class SGAAE_(nn.Module,Spectral_clustering_init):
    def __init__(self,init_dim,dataset_pgm,sparse_i,sparse_j,weights_signed, input_size,latent_dim,sample_size,scaling=1,device=None,dist='Dirichlet',VAE=False):
        super(SGAAE_, self).__init__()
        # initialization
        self.input_size=input_size
        self.dataset_pgm=dataset_pgm
        self.device=device
        # self.bias1=nn.Parameter(torch.randn(1,device=device))
        # self.bias2=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
        self.init_dim=init_dim
        
        self.learning_A=False
        self.dist=dist
        
        self.VAE=VAE
        
        # self.gamma1=nn.Parameter(torch.rand(input_size,device=device))
        # self.delta1=nn.Parameter(torch.rand(input_size,device=device))

        # self.gamma1=torch.ones(self.input_size,device=device)
        self.weights_signed=weights_signed
        # self.R_tot=nn.Parameter(torch.randn(self.latent_dim,self.latent_dim,device=device))
        
        self.scaling=scaling
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        
        self.sampling_weights=torch.ones(self.input_size,device=device)
        self.sample_size=sample_size
        self.Softmax=nn.Softmax(1)
        self.learning_emb=False
        
        self.use_GCN=True

        
        if sample_size==input_size:
            self.up_i,self.up_j=torch.triu_indices(input_size,input_size,1)
        else:
            self.up_i_,self.up_j_=torch.triu_indices(sample_size,sample_size,1)
            self.up_i__,self.up_j__=torch.triu_indices(sample_size,sample_size,-1)
            self.up_i=torch.cat((self.up_i_,self.up_i__))
            self.up_j=torch.cat((self.up_j_,self.up_j__))



            
        self.softplus=nn.Softplus()
        self.batch=torch.zeros(self.input_size).long()
        
        self.a=0.1*torch.ones(self.latent_dim,device=device)

        self.dir_=torch.distributions.Dirichlet(self.a)

        self.length_=nn.Parameter(torch.rand(1))
 
        

        self.mu_dir = torch.log(self.a) - torch.mean(torch.log(self.a))

        self.var_dir = (((1.0 / self.a) * (1 - (2.0 / self.latent_dim))) +

                                    (1.0 / (self.latent_dim * self.latent_dim)) * torch.sum(1.0 / self.a))
       
        
        self.elements=0.5*(input_size*(input_size-1))
        self.torch_pi=torch.tensor(math.pi)
        self.R_tot=nn.Parameter(0.1*torch.randn(latent_dim,latent_dim))
        
        
        hidden1=64
        hidden2=32
        n_layers=2
        
        self.dropout=0.
        
        self.fc_z=nn.Linear(hidden1, latent_dim)
        self.fc_w=nn.Linear(hidden1, latent_dim)
        
        self.fc_Gp=nn.Linear(hidden1, latent_dim)
        self.fc_Gn=nn.Linear(hidden1, latent_dim)
        
        self.fc_g=nn.Linear(hidden1, 1)
        self.fc_d=nn.Linear(hidden1, 1)
        
        self.fc_R=nn.Linear(2*hidden1, latent_dim**2)


            

            
            


   
        self.GCN_pos=GCN(input_dim=self.init_dim, hidden_dim=hidden1, latent_dim=hidden1, n_layers=2,dropout=self.dropout)
       

        self.GCN_neg=GCN(input_dim=self.init_dim, hidden_dim=hidden1, latent_dim=hidden1, n_layers=2,dropout=self.dropout)


      
    def forward_polytope(self,features_plus,features_minus,edge_index_pos,edge_index_neg,training=True):
        
        R_plus=self.GCN_A_plus(x=features_plus,edge_index=edge_index_pos,batch=self.batch,pool=True,training=training).view(-1)
        R_neg=self.GCN_A_neg(x=features_minus,edge_index=edge_index_neg,batch=self.batch,pool=True,training=training).view(-1)
        
        
        G_plus=self.GCN_G_plus(x=features_plus,edge_index=edge_index_pos,batch=self.batch,training=training)
        G_neg=self.GCN_G_neg(x=features_minus,edge_index=edge_index_neg,batch=self.batch,training=training)
        
        
        return R_plus.view(self.latent_dim,self.latent_dim),R_neg.view(self.latent_dim,self.latent_dim),G_plus,G_neg
    
  
       
        
        
       
    def forward(self,features_plus,features_minus,edge_index_pos,edge_index_neg):


            
            x=self.GCN_pos(x=features_plus,edge_index=edge_index_pos,batch=self.batch)
            y=self.GCN_neg(x=features_minus,edge_index=edge_index_neg,batch=self.batch)
            
            
            
            
            return x,y
            
            
    def forward_bias(self,features_plus,features_minus,edge_index_pos,edge_index_neg,training=True):
        if self.use_GCN:


            if self.VAE:
                x1_=self.GCN_g(x=features_plus,edge_index=edge_index_pos,batch=self.batch,training=training)
                x2_=self.GCN_d(x=features_minus,edge_index=edge_index_neg,batch=self.batch,training=training)

                log_var_x1=x1_[:,1]#.clamp(np.log(1e-8), -np.log(1e-8))
                log_var_x2=x2_[:,1]#.clamp(np.log(1e-8), -np.log(1e-8))
                x1=x1_[:,0]
                x2=x2_[:,0]

 
 
                return x1.view(-1),log_var_x1.view(-1).clamp(np.log(1e-8), -np.log(1e-8)),x2.view(-1),log_var_x2.view(-1).clamp(np.log(1e-8), -np.log(1e-8))
            else:
                x1=self.GCN_g(x=features_plus,edge_index=edge_index_pos,batch=self.batch,training=training)
                x2=self.GCN_d(x=features_minus,edge_index=edge_index_neg,batch=self.batch,training=training)
                
                return x1.view(-1),x2.view(-1)
            
            
    def KL_normal(self,mu,log_var):

        KL_norm = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
 
        

        return KL_norm.sum()

    def KL_dirichlet(self,mu,log_var):

        var_model=torch.exp(log_var)


        term1=torch.sum(((1/self.var_dir)*var_model),1)

        term2=torch.log(torch.prod(self.var_dir)+1e-10)-torch.log(torch.prod(var_model,1)+1e-10)

        term3=(((mu-self.mu_dir)*(1/self.var_dir))*(mu-self.mu_dir)).sum(-1)


        KL_loss_dir=0.5 * (term1+term2+term3-self.latent_dim )


        return KL_loss_dir.sum()
        
            

    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm
    
        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,self.weights_signed.float(), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
        
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]
     
        
        return sample_idx,sparse_i_sample,sparse_j_sample,valueC
    
    
    

    
    
    def LSM_likelihood_bias_sample(self,epoch):
        '''
        Skellam MAP ignoring constant terms
        
        '''
        self.epoch=epoch
        
  
        
        
        if self.VAE:
            
            max_value=10
            min_value=-10
            
        else:
        
            max_value=150
            min_value=-150

        
        sample_idx,sparse_i_sample,sparse_j_sample,self.weights_sample=self.sample_network()

        if self.scaling:
            
            
            
            
            self.seed_plus,self.seed_neg=self.forward(self.dataset_pgm.x_plus,self.dataset_pgm.x_minus,self.dataset_pgm.edge_index_pos,self.dataset_pgm.edge_index_neg)
            self.gamma=self.fc_g(self.seed_plus).view(-1)
            self.delta=self.fc_d(self.seed_neg).view(-1)
            self.latent_z_=self.fc_z(self.seed_plus)
            self.latent_w_=self.fc_w(self.seed_neg)

            self.gamma_=torch.clamp(self.gamma,min=min_value,max=max_value)
            self.delta_=torch.clamp(self.delta,min=min_value,max=max_value)
            self.latent_z =torch.softmax(self.latent_z_,1) 

            self.latent_w = torch.softmax(self.latent_w_,1) 
                
            temp_gamma=self.gamma_[sample_idx]
            temp_delta=self.delta_[sample_idx]

 
            
            bias_m_g=temp_gamma.unsqueeze(1)+temp_gamma
            bias_m_d=temp_delta.unsqueeze(1)+temp_delta
            mat_gamma=torch.exp(bias_m_g)
            mat_delta=torch.exp(bias_m_d)

            z_pdist1_1=mat_gamma
            z_pdist1_2=mat_delta

            z_pdist1=((z_pdist1_1+z_pdist1_2)[self.up_i,self.up_j]).sum()

            z_pdist2=((self.weights_sample/2)*((self.gamma_[sparse_i_sample]+self.gamma_[sparse_j_sample]-self.delta_[sparse_i_sample]-self.delta_[sparse_j_sample]))).sum()
            rates_non_link=0.5*((bias_m_g+bias_m_d)[self.up_i,self.up_j]).view(-1)
            rates_link=0.5*(self.gamma_[sparse_i_sample]+self.gamma_[sparse_j_sample]+self.delta_[sparse_i_sample]+self.delta_[sparse_j_sample])

            log_bessel_0,log_bessel_1=self.bessel_calc_sample(rates_non_link=rates_non_link,rates_link=rates_link)
            log_likelihood_sparse=-(z_pdist2-z_pdist1+log_bessel_0+log_bessel_1)
            
            
            if self.VAE:
                KL_gamma=self.KL_normal(self.gamma_mean[sample_idx],self.gamma_logvar[sample_idx])
                KL_delta=self.KL_normal(self.delta_mean[sample_idx],self.delta_logvar[sample_idx])
                KL_Z=0
                KL_W=0
                    

    
            if self.epoch==200:
                self.scaling=0
                self.learning_emb=True
   
            
            
            
        else:
            
            self.seed_plus,self.seed_neg=self.forward(self.dataset_pgm.x_plus,self.dataset_pgm.x_minus,self.dataset_pgm.edge_index_pos,self.dataset_pgm.edge_index_neg)
            self.gamma=self.fc_g(self.seed_plus).view(-1)
            self.delta=self.fc_d(self.seed_neg).view(-1)
            self.latent_z_=self.fc_z(self.seed_plus)
            self.latent_w_=self.fc_w(self.seed_neg)
  
            self.gamma_=torch.clamp(self.gamma,min=min_value,max=max_value)
            self.delta_=torch.clamp(self.delta,min=min_value,max=max_value)
            tau=0.1
            self.latent_raa_z =torch.softmax(self.latent_z_/tau,1) 
  
            self.latent_raa_w = torch.softmax(self.latent_w_/tau,1) 
              
            self.G_plus=self.fc_Gp(self.seed_plus)
            self.G_neg=self.fc_Gn(self.seed_neg)
            
            
            # self.R_tot=self.fc_R(torch.cat((self.seed_plus,self.seed_neg),1)).mean(0).view(self.latent_dim,self.latent_dim)
            # self.R_tot=((self.seed_plus+self.seed_neg).mean(0)).view(self.latent_dim,self.latent_dim)

            
            
      #       self.Gate_plus=torch.sigmoid(self.G_plus)
      #       self.C_plus = (self.latent_raa_z * self.Gate_plus) / (self.latent_raa_z * self.Gate_plus).sum(0)
      # #      self.A_plus=(self.softplus(self.R_plus).transpose(0,1)@(self.latent_raa_z.transpose(0,1)@self.C_plus)).transpose(0,1)
      #       self.A_plus=((self.R_tot).transpose(0,1)@(self.latent_raa_z.transpose(0,1)@self.C_plus)).transpose(0,1)

            self.latent_z=self.latent_raa_z@self.R_tot
            
            
     #        self.Gate_neg=torch.sigmoid(self.G_neg)
     #        self.C_neg = (self.latent_raa_w * self.Gate_neg) / (self.latent_raa_w * self.Gate_neg).sum(0)
     # #       self.A_neg=(self.softplus(self.R_neg).transpose(0,1)@(self.latent_raa_w.transpose(0,1)@self.C_neg)).transpose(0,1)
     #        self.A_neg=((self.R_tot).transpose(0,1)@(self.latent_raa_w.transpose(0,1)@self.C_neg)).transpose(0,1)

            self.latent_w=self.latent_raa_w@self.R_tot
            
            
            
            
            
            temp_z=self.latent_z[sample_idx]
            temp_w=self.latent_w[sample_idx]

            mat_z=(temp_z.unsqueeze(1)*temp_z).sum(-1)+1e-06
            mat_w=(temp_w.unsqueeze(1)*temp_w).sum(-1)+1e-06

            temp_gamma=self.gamma_[sample_idx]
            temp_delta=self.delta_[sample_idx]


            
            mat_gamma=temp_gamma.unsqueeze(1)+temp_gamma
            mat_delta=temp_delta.unsqueeze(1)+temp_delta
            
            

            z_pdist1_1=torch.exp(mat_z+mat_gamma)
            z_pdist1_2=torch.exp(mat_w+mat_delta)

            z_pdist1=((z_pdist1_1+z_pdist1_2)[self.up_i,self.up_j]).sum()
            temp_dist_z=-(((((self.latent_z[sparse_i_sample]*self.latent_z[sparse_j_sample]+1e-06)).sum(-1))))
            temp_dist_w=-(((((self.latent_w[sparse_i_sample]*self.latent_w[sparse_j_sample]+1e-06)).sum(-1))))

            z_pdist2=((self.weights_sample/2)*((-temp_dist_z+temp_dist_w)+(self.gamma_[sparse_i_sample]+self.gamma_[sparse_j_sample]-self.delta_[sparse_i_sample]-self.delta_[sparse_j_sample]))).sum()
            rates_non_link=0.5*(((mat_gamma+mat_z)+(mat_delta+mat_w))[self.up_i,self.up_j]).view(-1)
            rates_link=0.5*(self.gamma_[sparse_i_sample]+self.gamma_[sparse_j_sample]-temp_dist_z-temp_dist_w+self.delta_[sparse_i_sample]+self.delta_[sparse_j_sample])

            log_bessel_0,log_bessel_1=self.bessel_calc_sample(rates_non_link=rates_non_link,rates_link=rates_link)
            log_likelihood_sparse=-(z_pdist2-z_pdist1+log_bessel_0+log_bessel_1)
            
            
            if self.VAE:
                KL_gamma=self.KL_normal(self.gamma_mean[sample_idx],self.gamma_logvar[sample_idx])
                KL_delta=self.KL_normal(self.delta_mean[sample_idx],self.delta_logvar[sample_idx])
                if self.dist=='Dirichlet':
                    KL_Z=self.KL_dirichlet(self.latent_z_mean[sample_idx],self.latent_z_logvar[sample_idx])
                    KL_W=self.KL_dirichlet(self.latent_w_mean[sample_idx],self.latent_w_logvar[sample_idx])
                if self.dist=='Normal':
                    KL_Z=self.KL_normal(self.latent_z_mean[sample_idx],self.latent_z_logvar[sample_idx])
                    KL_W=self.KL_normal(self.latent_w_mean[sample_idx],self.latent_w_logvar[sample_idx])
        
        if self.VAE:
            return log_likelihood_sparse,KL_gamma,KL_delta,KL_Z,KL_W
        else:
            return log_likelihood_sparse

    
    def bessel_calc_sample(self,rates_non_link,rates_link):
        
       
        nu_link=self.weights_sample.abs().float()
    
    
    
        sum_el=50
       
        order=torch.arange(sum_el)


        q=-torch.special.gammaln(order+1)-torch.special.gammaln(nu_link.unsqueeze(1)+order+1)+  rates_link.unsqueeze(-1)*(nu_link.unsqueeze(1)+2*order)
        logI=torch.logsumexp(q,1)
        
        
        nu_link_z=torch.zeros(self.up_i.shape[0]).float()
    
    
    
      

        q_z=-torch.special.gammaln(order+1)-torch.special.gammaln(nu_link_z.unsqueeze(1)+order+1)+  rates_non_link.unsqueeze(-1)*(nu_link_z.unsqueeze(1)+2*order)
        logI_z=torch.logsumexp(q_z,1)
        
        nu_link_e=torch.zeros(self.weights_sample.shape[0]).float()
    
    
    
       


        q_e=-torch.special.gammaln(order+1)-torch.special.gammaln(nu_link_e.unsqueeze(1)+order+1)+  rates_link.unsqueeze(-1)*(nu_link_e.unsqueeze(1)+2*order)
        logI_e=torch.logsumexp(q_e,1)

      
        
        log_bessel_0=logI_z.sum()-logI_e.sum()

        log_bessel_1=logI.sum()
        return log_bessel_0,log_bessel_1
    
    
    
    

    
    
    
 
           
