import os,sys
import torch
import torch.utils.data as torutils
from torch.autograd import Variable
from myutils.data_reader import *
from myutils.parser import *

def sigmoid(z):
	return (1.+torch.exp(-z))**-1

def manipX(ipt):
    ## manipulate the data-set here to the data in dstream
    ## ipt : ndarray
    ipt = ipt.astype(np.float)
    ipt = np.delete(ipt,1,1)
    #print (ipt)
    return ipt

def manipY(ipt):

    ipt = ipt.astype(np.float)
    ipt = ipt.flatten()
    #ipt = ipt[:]
    #ipt = np.delete(ipt,2,1)
    return ipt

#===========================================================================
pars = Parser()
pars.parse(sys.argv[1])
Xpath = sys.argv[2]
Ypath = sys.argv[3]
is_res = int(sys.argv[4])
ID = pars.val['ID']
learn_rate = float(pars.val['learn_rate'])
Nepoch     = int(pars.val['epoch'])
#regu_rate  = float(pars.val['regu_rate'])
model_dir = os.path.join("model",ID)
drop_out   = float(pars.val['dropout'])
batchsz   = int(pars.val['batchsz'])

if not os.path.exists(model_dir):
	os.system("mkdir %s"%(model_dir))


## layout :
pars.print_vals()



### Read Data:
Xreader = Data_reader()
Xreader.Read_csv(Xpath,"big5",manipX)
Yreader = Data_reader()
Yreader.Read_csv(Ypath,"big5",manipY)

NFeature = len(Xreader.data[0])
NSample  = len(Xreader.data)
NTrain   = int(NSample*0.8)




# shuffle first:
idmap = None
if is_res:
	idmap = np.load(os.path.join(model_dir,"shufid.npy"))
	print ("[load] shufid")
else :
	idmap = np.random.permutation(NSample)
	np.save(os.path.join(model_dir,"shufid"),idmap)
all_x = Xreader.data[idmap]
all_y = Yreader.data[idmap]

# Feature scaling :
if is_res:
	Xmean = np.load(os.path.join(model_dir,"FS_mean.npy"))
	Xstd  = np.load(os.path.join(model_dir,"FS_std.npy") )
	print ("[load] FS_mean")
	print ("[load] FS_std" )
else :
	Xmean = np.mean(all_x,axis=0)
	Xstd  = np.std(all_x,axis=0)
	np.save(os.path.join(model_dir,"FS_mean"),Xmean)
	np.save(os.path.join(model_dir,"FS_std"),Xstd)

all_x = (all_x - Xmean)/Xstd


## valid / train => torch tensor
train_x = torch.from_numpy(all_x[:NTrain])
valid_x = torch.from_numpy(all_x[NTrain:NSample])
train_y = torch.from_numpy(all_y[:NTrain])
valid_y = torch.from_numpy(all_y[NTrain:NSample])

##create dataset:
train_loader  = torutils.DataLoader(torutils.TensorDataset(train_x,train_y),batch_size=batchsz,shuffle=True)
valid_loader = torutils.DataLoader(torutils.TensorDataset(valid_x,valid_y),batch_size=len(valid_x),shuffle=False)


#------------------------------------------------------------------
## create model :
ClassifyNet = None
ClassifyNet = torch.nn.Sequential(\
				torch.nn.Linear(NFeature, NFeature),\
                torch.nn.Dropout(drop_out),\
                torch.nn.Softplus(),\
				torch.nn.Linear(NFeature, NFeature),\
                torch.nn.Dropout(drop_out),\
                torch.nn.ReLU(),\
				torch.nn.Linear(NFeature, NFeature),\
                torch.nn.Dropout(drop_out),\
                torch.nn.ReLU(),\
                torch.nn.Linear(NFeature, 1),\
                #torch.nn.Dropout(0.5),\
                torch.nn.Sigmoid()\
             ).double()


if is_res:
	ClassifyNet.load_state_dict(torch.load(os.path.join(model_dir,"model")))
	print ("[load] Net(model)")

optimizer = torch.optim.SGD(ClassifyNet.parameters(), lr=learn_rate)
loss_fx   = torch.nn.BCELoss()

print ("Start")
sys.stdout.flush()
for en in range(Nepoch):
	error=0
	for ns , (tr_x,tr_y) in enumerate(train_loader):
		pred = ClassifyNet(Variable(tr_x))
		#exit(1)
		error += np.sum(np.abs(np.round(pred.data.numpy().flatten()) - tr_y.numpy().flatten()))
		#exit(1)
		loss = loss_fx(pred.squeeze(),Variable(tr_y))
		#exit(1)
		ClassifyNet.zero_grad()
		loss.backward()
		optimizer.step()

	error /= len(train_x)
	correct = 1.-error
	print ("e %3d] Xs: %4.6f As: %4.6f"%(en,loss.data.numpy(),correct))
	sys.stdout.flush()

## Save:
torch.save(ClassifyNet.state_dict(),os.path.join(model_dir,"model"))

## Validate
accuracy = []
for n , (va_x,va_y) in enumerate(valid_loader):
	#print ("va",n)
	pred = ClassifyNet(Variable(va_x))
	#print (np.shape(pred.data.numpy().flatten()))
	#print (np.shape(va_y.numpy().flatten()))
	accuracy = 1.- np.mean(np.abs(np.round(pred.data.numpy().flatten()) - va_y.numpy().flatten()))

print (accuracy)
sys.stdout.flush()
