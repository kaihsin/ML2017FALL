import os,sys
from math import *
from cnn import *
import torch 
import torch.utils.data
from torch.autograd import *
from data_reader import *
from parser import *
from tools import *

## Global:
Fig_y = 48
Fig_x = 48
Classify = 7
mode = 0

## data processing func:
def train_fx(ipt):
	global Classify,Fig_y,Fig_x
	#print (ipt[0,1])
	label = np.array(ipt[:,0],dtype=np.int)
	#print (label)
	#print (np.shape(ipt[:,1].flatten()))
	
	ipt   = np.array([np.array(ipt[t,1].split(' '),dtype=np.float) for t in range(len(ipt))])

	#print(np.shape(ipt))

	#exit(1)
#.reshape(len(ipt),Fig_y,Fig_x)
	#print (ipt)
	#exit(1)
	ipt = [label,ipt] 
	return ipt


if len(sys.argv) < 3:
	print ("train.py <.prop> <data> [mode]") 
	exit(1)

if len(sys.argv) > 3:
	mode = int(sys.argv[3])
	
pars = Parser()
pars.parse(sys.argv[1])

ID          = pars.val['ID']
Batch_sz    = int(pars.val['batch_sz'])
epoch       = int(pars.val['epoch'])
learn_rate  = float(pars.val['learn_rate'])
train_rate  = float(pars.val['train_rate']) # percentage of data to train
argu_ratio = 0.5
model_dir   = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model'),ID)

##checking :
if not os.path.exists(model_dir):
	os.makedirs(model_dir)

## Logging
print ("[mode] %d"%(mode))
pars.print_vals()
sys.stdout.flush()

## data:
raw = Data_reader()
raw.Read_csv(sys.argv[2],'big5',train_fx)
Ndata = len(raw.data[0])
Ntrain = int(Ndata*train_rate)
Ntest  = Ndata - Ntrain
print ("Total data : %d"%(Ndata))
print ("Train data : %d"%(Ntrain))
print ("Valid data : %d"%(Ntest))	
sys.stdout.flush()

## data processing:
#print(np.shape(raw.data[1]))
#exit(1)
train_x = torch.DoubleTensor(raw.data[1]/255).view(-1,1,Fig_y,Fig_x)
train_y = torch.from_numpy(raw.data[0]).int()

#print (train_y)
del raw ## release source
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



## build model :
model = cnn_fix_v8(Fig_y,Fig_x,Classify)
if mode :
	print ("[load model state]")
	sys.stdout.flush()
	model.load_state_dict(torch.load(os.path.join(model_dir,'cnn.model')))

#print (model)

optimizer = torch.optim.Adam(model.parameters(),lr=learn_rate)
loss_fx = torch.nn.CrossEntropyLoss()
loss = None
## training :

# move to gpu:
#train_loader.gpu()
model.cuda()
model.train()
for e in range(epoch):
	#print ("e %3d]")
	acc = 0
	cnt = 0
	for i , (x,y) in enumerate(train_loader):
		cnt += 1
		## data argumented:
		argu = RandVflip( x.numpy(),argu_ratio,2)
		argu = RandHflip( argu     ,argu_ratio,2)
		#argu = RandRotate(argu     ,argu_ratio,2,45)
		argu = torch.from_numpy(argu)
		
		pred = model.forward(Variable(argu.cuda(),requires_grad=False))
		loss = loss_fx(pred,Variable( y.cuda(),requires_grad=False))	
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		pclass = np.argmax(pred.cpu().data.numpy(),axis=1)
		acc += np.mean(pclass == y.numpy())
		#exit(1)
	acc/=cnt
	#loss.cpu()
	print ("e %3d] Xs: %4.4f Ac: %3.4f"%(e,loss.cpu().data.numpy(),acc))
	sys.stdout.flush()

## save :
model.cpu()
torch.save(model.state_dict(),os.path.join(model_dir,'cnn.model'))



## validate:

model.cuda()
model.eval()
acc =0
cnt =0
for i ,(x,y) in enumerate(valid_loader):
	pred = model.forward(Variable(x.cuda()))
	#pred.cpu()
	pred = np.argmax(pred.cpu().data.numpy(),axis=1)
	#print (pred)
	#print (y.numpy())
	acc+= np.mean(pred == y.numpy())
	cnt+=1

acc/=cnt

print ("valid acc: %4.6lf"%(acc))



#print (model)




