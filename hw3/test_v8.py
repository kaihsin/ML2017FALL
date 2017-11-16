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

if len(sys.argv) < 2:
	print ("test.py <.prop> <data>") 
	exit(1)

pars = Parser()
pars.parse(sys.argv[1])

ID          = pars.val['ID']
model_dir   = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model'),ID)

## data:
raw = Data_reader()
raw.Read_csv(sys.argv[2],'big5',train_fx)

## data processing:
train_x = torch.DoubleTensor(raw.data[1]/255).view(-1,1,Fig_y,Fig_x)
#train_y = torch.from_numpy(raw.data[0]).int()

del raw


## build model :
model = cnn_fix_v8(Fig_y,Fig_x,Classify)
model.load_state_dict(torch.load(os.path.join(model_dir,'cnn.model')))

print ("Total: %d"%(len(train_x)))

#model.cuda()
model.eval()
Batch_sz = 128
res = []
for i in np.arange(0,len(train_x),Batch_sz):
	print (i)
	if i+Batch_sz > len(train_x):
		pred = model.forward(Variable(train_x[i:]))
	else:
		pred = model.forward(Variable(train_x[i:i+Batch_sz]))
	#pred.cpu()
	pred = np.argmax(pred.data.numpy(),axis=1)
	res = np.hstack((res,pred))


#print (len(res))


##save path :
sav_path = sys.argv[3]

#f = open(os.path.join(model_dir,'Out.csv'),'w')
f = open(sav_path,'w')
f.write('id,label\n')
for l in range(len(res)):
	f.write('%d,%d\n'%(l,res[l]))
f.close()






