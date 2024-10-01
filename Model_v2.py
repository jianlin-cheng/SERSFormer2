"""
author: Akshata 
timestamp: Wednesday April 17 2024 10:19 PM
"""
import torch.nn as nn
import torch
from torch import  Tensor
import math
class regression_features(nn.Module):
    def __init__(self, dim_model=32,channels=4):
        super(regression_features,self).__init__()
        
        self.cnn = nn.Conv1d(in_channels=channels, out_channels=dim_model, kernel_size=3)
        self.cnn2 = nn.Conv1d(in_channels=dim_model,out_channels=dim_model*2,kernel_size=3)
        self.cnn1_bn = nn.BatchNorm1d(dim_model)
        #self.pos_encoder = PositionalEncoding(dim_model, drop)
        self.max_pool = nn.MaxPool1d(kernel_size=3) 
        self.avg_pool = nn.AvgPool1d(kernel_size=3)
        self.layernorm = nn.LayerNorm(dim_model*2)
        self.droput = nn.Dropout1d(p=0.1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.cnn(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.cnn1_bn(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.droput(x)
        return x

class TransformerClassifyRegress_sep(nn.Module):
    def __init__(self, dim_model=32, attn_head=1, dim_ff=64, drop=0.1, batch_f=True, encoder_layers=1,n_labels=1):
        super(TransformerClassifyRegress_sep,self).__init__()
        print(attn_head)
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=dim_model, kernel_size=3),nn.ReLU(),nn.MaxPool1d(kernel_size=3) ,
                                 nn.BatchNorm1d(dim_model),
                                 nn.Conv1d(in_channels=dim_model,out_channels=dim_model*2,kernel_size=3),nn.ReLU(),nn.MaxPool1d(kernel_size=3) ,
                                nn.Dropout1d(p=0.5))
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=dim_model, kernel_size=3),nn.ReLU(),nn.MaxPool1d(kernel_size=3) ,
                                 nn.BatchNorm1d(dim_model),
                                 nn.Conv1d(in_channels=dim_model,out_channels=dim_model*2,kernel_size=3),nn.ReLU(),nn.MaxPool1d(kernel_size=3) ,
                                nn.Dropout1d(p=0.5))
        self.reg_feat = regression_features(dim_model=dim_model,channels=4)
        self.layernorm = nn.LayerNorm(dim_model*2)
       
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model*2, nhead=attn_head, dim_feedforward=dim_ff, dropout=drop, 
                                                                    batch_first=batch_f)
                                                                  
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,num_layers=encoder_layers)
        self.MLPreg = nn.Sequential(nn.Linear(172*dim_model*2*2,dim_model*4),nn.ReLU(),nn.BatchNorm1d(dim_model*4),
                                    nn.Linear(dim_model*4,dim_model),nn.ReLU(),nn.BatchNorm1d(dim_model),nn.Linear(dim_model,n_labels),nn.ReLU())
        self.MLPclass = nn.Sequential(nn.Linear(172*dim_model*2,dim_model*4),nn.ReLU(),nn.BatchNorm1d(dim_model*4),
                                      nn.Linear(dim_model*4, dim_model),nn.ReLU(),nn.BatchNorm1d(dim_model),nn.Linear(dim_model,n_labels),nn.Sigmoid())
        self.activation_ELU= nn.GELU()
        #Initialization
        #nn.init.xavier_normal_(self.cnn.weight)
        #nn.init.xavier_normal_(self.cnn2.weight)
        

    def forward(self,data):
        da_c = data[0]
         
        emb = self.cnn(da_c)   
        emb = emb.view(emb.size(0), emb.size(2), emb.size(1))
     
        #x = self.pos_encoder(x)
        x = self.layernorm(emb)
        x = self.activation_ELU(x)
        x_c = self.transformer_encoder(x)
        x = self.activation_ELU(x_c)
        
        x = x.view(x.size(0), x.size(2)*x.size(1))
       
        classify = self.MLPclass(x)

        #x,_ = torch.max(x, dim=1)
        
        da_r = data[2]
         
        emb1 = self.cnn2(da_r)   
        emb1 = emb1.view(emb1.size(0), emb1.size(2), emb1.size(1))
        x3 = self.layernorm(emb1)
        x3 = self.activation_ELU(x3)
        x3 = self.transformer_encoder(x3)
        x3 = x3+x_c
        x3 = self.activation_ELU(x3)
        
        
        x1 = data[1]
        x1 = self.reg_feat(x1)  
        x1 = x1.view(x1.size(0), x1.size(2), x1.size(1)) 
        x2 = torch.cat((x1,x3),-1)
        #x = x.squeeze(0) 
        #x,_ = torch.max(x, dim=1) 
        #x = x.squeeze(0)    

        #x2 = x2.squeeze(0)      
        x2 = self.activation_ELU(x2)
        x2 = x2.view(x2.size(0), x2.size(2)*x2.size(1))
        regress = self.MLPreg(x2)
        
        return [classify,regress]

