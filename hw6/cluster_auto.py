import numpy as np 
import torch
import torch.nn as nn 
from autoenc import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from cluster import *
#from pca import *
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import os,sys
#from tsne import *
from parser import *

datapath = sys.argv[3]
savepath = 'model/auto'


pars = Parser()
pars.parse(sys.argv[1])
isresume = int(sys.argv[2])

ID       = pars.val['ID']
batch_sz = int(pars.val['batch_sz'])
epoch    = int(pars.val['epoch'])
learn_rate = float(pars.val['learn_rate'])
valid_rate = float(pars.val['valid_rate'])
regu       = float(pars.val['regu'])
fdim       = int(pars.val['Fdim'])

model_dir = os.path.join(savepath,ID)
####
if not os.path.exists(model_dir):
    os.system("mkdir %s"%(model_dir))


#### Data process & load 
imgs = np.load(datapath)
imgs = imgs/255.
imgs = torch.from_numpy(imgs).float()

Ntot = len(imgs)
#Ntrain = int(Ntot*train_rate)
Nvalid = int(Ntot*(valid_rate))
print ("tot   %d"%(Ntot))
#print ("train %d"%(Ntrain))
print ("valid %d"%(Nvalid))

train_data = imgs
valid_data = imgs[:Nvalid]


#### Data loader 
train_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(train_data,torch.from_numpy(np.arange(0,len(train_data),1)).float()),\
                batch_size=batch_sz,\
                shuffle=True)
del train_data
del imgs


#### model 
model =Autoenc(Fdim = fdim)

if isresume :
    print ("Resume")
    sys.stdout.flush()
    model.load_state_dict(torch.load(os.path.join(model_dir,'autoenc.model')))

optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate,weight_decay=regu)
lossfx    = nn.MSELoss().cuda()
model.cuda()


model.train()
for e in range(epoch):
    
    for i ,(x,y) in enumerate(train_loader):
        xipt = Variable(x,requires_grad=False).cuda()
        context, y_pred = model(xipt)
        loss = lossfx(y_pred,xipt)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ("%d ] loss: %f"%(e,loss.data.cpu().numpy()[0]))


## Save:
model.cpu()
model.eval()
torch.save(model.state_dict(),os.path.join(model_dir,'autoenc.model'))



## validation:
model.cuda()
RMSE = 0
for i in np.arange(0,len(valid_data),batch_sz):
    endl = i + batch_sz
    if endl > Nvalid:
        endl = Nvalid
    x = Variable(valid_data[i:endl],requires_grad=False).cuda()
    context, y_pred = model(x)
    loss = lossfx(y_pred, x)
    RMSE += loss.data.cpu().numpy()[0]*len(x)
    
RMSE = np.sqrt(RMSE / Nvalid )
print ("RMSE : %f"%(RMSE))
print ("done.")


#plt.figure(2)
#fig , ax = plt.subplots(4)
#plt.scatter(pjimgs[:,0],pjimgs[:,1],s=10, c=pred_label)
#plt.show()


