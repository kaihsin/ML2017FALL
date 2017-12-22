import torch 
import torch.nn as nn
from torch.autograd import Variable
class MFnn(nn.Module):
    def __init__(self, usr_dim, mov_dim, emb_dim):
        super(MFnn, self).__init__()
        self.emb_u = nn.Embedding(usr_dim, emb_dim)
        self.emb_m = nn.Embedding(mov_dim, emb_dim)
        self.dnn_u = nn.Sequential(
                        nn.Linear(emb_dim,128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128,256),
                     )
        self.dnn_m = nn.Sequential(
                        nn.Linear(emb_dim,128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128,256),
                     )

        self.b_u = nn.Embedding(usr_dim,1)
        self.b_m = nn.Embedding(mov_dim,1)
        
        self.dropout = nn.Dropout(0.1)
        self.activ = nn.Sigmoid()

    def forward(self, users, movies):
        embedded_u = self.dnn_u(self.dropout(self.emb_u(users)))
        embedded_m = self.dnn_m(self.dropout(self.emb_m(movies)))
        bias_u = self.b_u(users)
        bias_m = self.b_m(movies)
        
        out = torch.sum(embedded_m * embedded_u, dim=1).view(-1,1) + bias_u + bias_m
        out = self.activ(out)
        return out








