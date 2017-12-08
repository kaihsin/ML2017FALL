import numpy as np 
import torch 
import torch.utils.data
from torch.autograd import Variable
from parser import *
from read_data import *
from tools import *
from rnn import *	
if (len(sys.argv) < 1):
	print (".py <.cfg> <dict_path> <data_test>")
	exit(1)

cfg_path = sys.argv[1]
dict_path = sys.argv[2]
data_path = sys.argv[3]

# read .cfg:
pars = Parser()
pars.parse(cfg_path)

ID          = pars.val['ID']
Batch_sz    = int(pars.val['batch_sz'])*2
embed_dim   = int(pars.val['embed_dim'])
dict_type   = int(pars.val['dict_type'])
model_dir   = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)),'model'),ID)

pars.print_vals()
sys.stdout.flush()


## read dict for tokenize:
tk_dict = Read_dict(dict_path)

## prep data:
raw = Read_test(data_path,tk_dict)

## padding :
raw = Padding(raw,tk_dict["PADD"],40)


## slice validation:
train_x = torch.from_numpy(raw).long()

#print (train_y)
del raw ## release source



print (len(tk_dict))



## model :
model = cnn_rnn_GRU(len(tk_dict),embed_dim,int(tk_dict["PADD"])).float()
model.load_state_dict(torch.load(os.path.join(model_dir,'rnn.model')))
			
model.cuda()
model.eval()
res = np.array([])
print ("Total: %d"%(len(train_x)))
for x in np.arange(0,len(train_x),Batch_sz):
	print (x)
	if x+Batch_sz > len(train_x):
		pred = model(Variable(train_x[x:],requires_grad=False).cuda())
	else:
		pred = model(Variable(train_x[x:x+Batch_sz],requires_grad=False).cuda())
	
	res = np.hstack((res,np.round(pred.cpu().data.numpy().flatten())))

f = open(os.path.join(model_dir,"Out.csv"),'w')
f.write("id,label\n")
for i in range(len(res)):
	f.write("%d,%d\n"%(i,res[i]))
	#print(res[i])
f.close()




