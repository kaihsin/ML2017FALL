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



def test_fx(ipt):
    usrmov = ipt[:,1:3] - 1 

    return usrmov 
    


if (len(sys.argv) < 4):
    print (".py <.cfg> <test.csv> <output>")
    exit(1)


cfg_path = sys.argv[1]
#mov_path = "./data/movies.csv"
#usr_path = "./data/users.csv"
#test_path = "./data/test.csv"
test_path = sys.argv[2]
sav_path  = sys.argv[3]
USR_DIM    = 6040
MOV_DIM    = 3952

# read .cfg:
pars = Parser()
pars.parse(cfg_path)

ID          = pars.val['ID']
Batch_sz    = int(pars.val['batch_sz'])
embed_dim   = int(pars.val['embed_dim'])

model_dir   = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model'),ID)

pars.print_vals()
sys.stdout.flush()


## feature:
#users_feature = pd.read_csv(usr_path, header=None, skiprows=1, sep="::", engine='python')
#users_feature = users_feature.as_matrix() # from pandas Frame to numpy array	
#movies_feature = pd.read_csv(mov_path, header=None, skiprows=1, sep="::", engine='python')
#movies_feature = movies_feature.as_matrix() # from pandas Frame to numpy array	
#print (len(users_feature))
#print (len(movies_feature))
## look-up table:
#mov2id = {}
#for i , mv in enumerate(movies_feature[:,0]):
#    mov2id[mv] = i

## test_data :
raw = Data_reader()
raw.Read_csv(test_path,Format="utf-8",func=test_fx)
N_Tot = len(raw.data)
usrmov  = torch.LongTensor(raw.data)


## build model :
model = MFnn(USR_DIM,MOV_DIM,embed_dim)
model.load_state_dict(torch.load(os.path.join(model_dir,'mf.model')))


model.cuda()
model.eval()
res = np.array([])
for x in np.arange(0,N_Tot,Batch_sz):
    end = x + Batch_sz
    x1 , x2 = None,None
    if end > N_Tot:
        x1 = usrmov[x:,0].cuda()
        x2 = usrmov[x:,1].cuda()
    else:
        x1 = usrmov[x:end,0].cuda()
        x2 = usrmov[x:end,1].cuda()

    pred = model.forward(Variable(x1,requires_grad=False).long(),Variable(x2,requires_grad=False))
    res= np.hstack((res,pred.data.squeeze().cpu().numpy()))
    #rmse += np.sum((pred.data.squeeze().cpu().numpy() - y.numpy())**2)

res = res*5 + 1
f = open(sav_path,'w')
f.write("TestDataID,Rating\n")
for i in range(len(res)):
    f.write("%d,%1.1f\n"%(i+1,res[i]))
f.close()


