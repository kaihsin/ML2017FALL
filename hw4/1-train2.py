import numpy as np 
import torch 
import torch.utils.data
from torch.autograd import Variable
from parser import *
from read_data import *
from tools import *
from rnn import *	
if (len(sys.argv) < 1):
	print (".py <.cfg> <dict_path> <data_lbl> <is resume 0/1>")
	exit(1)

cfg_path = sys.argv[1]
dict_path = sys.argv[2]
data_path = sys.argv[3]
isresume = int(sys.argv[4])

# read .cfg:
pars = Parser()
pars.parse(cfg_path)

ID          = pars.val['ID']
Batch_sz    = int(pars.val['batch_sz'])
epoch       = int(pars.val['epoch'])
embed_dim   = int(pars.val['embed_dim'])
learn_rate  = float(pars.val['learn_rate'])
train_rate  = float(pars.val['train_rate']) # percentage of data to train
dict_type   = int(pars.val['dict_type'])
regu        = float(pars.val['regu'])
model_dir   = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model'),ID)

pars.print_vals()
sys.stdout.flush()

##checking :
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

## read dict for tokenize:
tk_dict = Read_dict(dict_path)

## prep data:
label , raw = Read_label(data_path,tk_dict)
#exit(1)
## padding :
raw = Padding(raw,tk_dict["PADD"],40)

## slice validation:
Ntotal = len(label)
Ntrain = int(Ntotal*train_rate)
Nvalid = Ntotal - Ntrain
print ("Total: %d"%(Ntotal))
print ("Train: %d"%(Ntrain))
print ("Valid: %d"%(Nvalid))
sys.stdout.flush()
train_x = torch.from_numpy(raw).long()
train_y = torch.from_numpy(label).float()

#print (train_y)
del raw ## release source
del label

train_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(train_x[:Ntrain],train_y[:Ntrain]),\
                batch_size=Batch_sz,\
                shuffle=True)

valid_loader = torch.utils.data.DataLoader(\
                torch.utils.data.TensorDataset(train_x[Ntrain:],train_y[Ntrain:]),\
                batch_size=Batch_sz,\
                shuffle=False)
del train_x
del train_y



print(len(tk_dict))


## model :
model = cnn_rnn_GRU(len(tk_dict),embed_dim,int(tk_dict["PADD"])).float()
optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate,weight_decay=regu)
lossfx = nn.BCELoss()

if isresume :
	print ("Resume")
	sys.stdout.flush()
	model.load_state_dict(torch.load(os.path.join(model_dir,'rnn.model')))
			
model.cuda()
model.train()
for e in range(epoch):
	acc = 0
	for b, (x,y) in enumerate(train_loader):
		pred = model(Variable(x.cuda(),requires_grad=False))
		loss = lossfx(pred.squeeze().double(),Variable(y.cuda(),requires_grad=False))
	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		acc += np.sum(1. - np.abs(np.round(pred.cpu().data.numpy().flatten()) - y.numpy().flatten()))
	
	acc/=Ntrain
	print ("%d] loss: %f acc: %f\n"%(e,loss.cpu().data.numpy(),acc))
	sys.stdout.flush()

model.cpu()
torch.save(model.state_dict(),os.path.join(model_dir,'rnn.model'))


model.cuda()
model.eval()
ac = 0
for b , (x,y) in enumerate(train_loader):
	pred = model(Variable(x.cuda(),requires_grad=False))
	ac += np.sum(1. - np.abs(np.round(pred.cpu().data.numpy().flatten()) - y.numpy().flatten()))

ac /= Ntrain
print ("valid : acc: %f\n"%(ac))





