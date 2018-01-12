import torch 
import torch.nn as nn 

class Autoenc(nn.Module):
    def __init__(self,Fdim=8):
        super(Autoenc,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(64,Fdim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(Fdim,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.ReLU(),
        )

    def forward(self,x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode , decode

"""
class CAutoenc(nn.Module):
    def __init__(self,Fdim):
        super(CAutoenc, self).__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) 
            #nn.Conv2d(64,128, 3, stride=4,padding=1),  # b, 8, 3, 3
            #nn.BatchNorm2d(128),
            #nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(7*7*64,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,Fdim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(Fdim,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid()
        )


    def forward(self, x):
        feture = self.encoder_cnn(x).view(len(x),-1)
        #print(feture.size())
        #exit(1)
        feture = self.encoder(feture)
        x = self.decoder(feture)
        return feture, x.view(len(x),28,28)
"""



         


