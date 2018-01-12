import numpy as np 
import torch
import torch.nn as nn 
from autoenc import *
from torch.autograd import Variable
from cluster import *
from pca import *
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import os,sys
from parser import *

datapath = sys.argv[2]
savepath = 'model/auto'


pars = Parser()
pars.parse(sys.argv[1])

ID       = pars.val['ID']
batch_sz = int(pars.val['batch_sz'])
epoch    = int(pars.val['epoch'])
learn_rate = float(pars.val['learn_rate'])
valid_rate = float(pars.val['valid_rate'])
regu       = float(pars.val['regu'])
fdim       = int(pars.val['Fdim'])
model_dir = os.path.join(savepath,ID)


#### Data process & load 
imgs = np.load(datapath)
imgs = imgs/255.
imgs = torch.from_numpy(imgs).float()

Ntot = len(imgs)
print ("tot   %d"%(Ntot))
#train_data = imgs

#### Data loader 
#train_loader = torch.utils.data.DataLoader(\
#                torch.utils.data.TensorDataset(train_data,torch.from_numpy(np.arange(0,len(train_data),1)).float()),\
#                batch_size=batch_sz,\
#                shuffle=True)
#del train_data
#del imgs


#### model 
model =Autoenc(Fdim=fdim)
model.load_state_dict(torch.load(os.path.join(model_dir,'autoenc.model')))


lossfx    = nn.MSELoss().cuda()
model.cuda()
model.eval()



## get feature:
Feat = []
for i in np.arange(0,len(imgs),batch_sz):
    print (i)
    endl = i + batch_sz
    if endl > Ntot:
        endl = Ntot
    x = Variable(imgs[i:endl],requires_grad=False).cuda()
    context, y_pred = model(x)
    Feat.append(context)

Feat = torch.cat(Feat).data.cpu().numpy()

print ("PCA for features")
mean,s,eigV = PCA(Feat)

pjimgs = np.dot(Feat-mean, eigV.T)


print ("clustering")
#pred_label , _ = K_means_batch(pjimgs,2,batch_size=2480,verbose=True)
#pred_label , _ = K_means(pjimgs,3,verbose=True)
#pred_label, _ = SpCluster(pjimgs[:20000])
#pred_label , _ = Agg(pjimgs,2)
pred_label , _ = Birch(pjimgs,2)


print ("save pred")
np.save(os.path.join(model_dir,'pred.npy'),pred_label)

#plt.scatter(pjimgs[:,0],pjimgs[:,1],s=0.5,c=pred_label)
#plt.show()


