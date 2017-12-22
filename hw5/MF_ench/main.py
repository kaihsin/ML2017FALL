import numpy as np 
import torch
import torch.utils.data
from torch.autograd import Variable
import os,sys
from model import *
from parser import *
import pandas as pd
from tools import *
from  data_reader import *



def train_fx(ipt):
    usrmov = ipt[:,1:3] - 1 
    rate   = (ipt[:,3]-1.)/5

    return usrmov, rate
    


if (len(sys.argv) < 1):
    print (".py <.cfg> <is resume 0/1>")
    exit(1)


cfg_path = sys.argv[1]
#mov_path = "./data/movies.csv"
#usr_path = "./data/users.csv"
train_path = "./data/train.csv"
USR_DIM    = 6040
MOV_DIM    = 3952
isresume = int(sys.argv[2])

# read .cfg:
pars = Parser()
pars.parse(cfg_path)

ID          = pars.val['ID']
Batch_sz    = int(pars.val['batch_sz'])
epoch       = int(pars.val['epoch'])
embed_dim   = int(pars.val['embed_dim'])
learn_rate  = float(pars.val['learn_rate'])
train_rate  = float(pars.val['train_rate']) # percentage of data to train
regu        = float(pars.val['regu'])

model_dir   = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model'),ID)

pars.print_vals()
sys.stdout.flush()

##checking :
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

## feature:
#users_feature = pd.read_csv(usr_path, header=None, skiprows=1, sep="::", engine='python')
#users_feature = users_feature.as_matrix() # from pandas Frame to numpy array	
#movies_feature = pd.read_csv(mov_path, header=None, skiprows=1, sep="::", engine='python')
#movies_feature = movies_feature.as_matrix() # from pandas Frame to numpy array	
#print (len(users_feature))
#print (len(movies_feature))

## look-up table:
mov2id = {}
for i , mv in enumerate(movies_feature[:,0]):
    mov2id[mv] = i

## train_data :
raw = Data_reader()
raw.Read_csv(train_path,Format="utf-8",func=train_fx)
N_Tot = len(raw.data[0])
N_Train = int(N_Tot*train_rate)
N_Valid = N_Tot-N_Train
usrmov  = torch.LongTensor(raw.data[0])
rate    = torch.from_numpy(raw.data[1]).float()

del raw
train_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(usrmov[:N_Train],rate[:N_Train]),\
                batch_size=Batch_sz,\
                shuffle=True)

valid_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(usrmov[N_Train:],rate[N_Train:]),\
                batch_size=Batch_sz,\
                shuffle=False)

del usrmov,rate


## build model :
model = MFnn(USR_DIM,MOV_DIM,embed_dim)
if isresume:
    print ("[load model]")
    sys.stdout.flush()
    model.load_state_dict(torch.load(os.path.join(model_dir,'mf.model')))

optimizer = torch.optim.Adam(model.parameters(),lr= learn_rate,weight_decay=regu)
loss_fx = nn.MSELoss()


model.cuda()
model.train()
for e in range(epoch):
    rmse = 0
    for i , (x,y) in enumerate(train_loader):
        #print (i)
        x1 = x[:,0].cuda()
        x2 = x[:,1].cuda()
        pred = model.forward(Variable(x1,requires_grad=False).long(),Variable(x2,requires_grad=False))
        loss = loss_fx(pred,Variable(y.cuda().unsqueeze(1),requires_grad=False).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print (pred.data.squeeze().cpu().numpy() - y.numpy())
        #exit(1) 
        rmse += np.sum((pred.data.squeeze().cpu().numpy() - y.numpy())**2)
    rmse = 5*np.sqrt(rmse/N_Train)
    
    print ("%2d] rmse : %f"%(e,rmse))

    
model.cpu()
torch.save(model.state_dict(),os.path.join(model_dir,'mf.model'))

## validate:
model.cuda()
model.eval()
rmse = 0
tot = 0
for i , (x,y) in enumerate(valid_loader):
    x1 = x[:,0].cuda()
    x2 = x[:,1].cuda()
    pred = model.forward(Variable(x1,requires_grad=False).long(),Variable(x2,requires_grad=False))
    rmse += np.sum((pred.data.squeeze().cpu().numpy() - y.numpy())**2)
    tot += len(y.numpy())
    
rmse = 5*np.sqrt(rmse/N_Valid)
print (tot)
print (N_Valid)
print ("valid: %f"%(rmse))


