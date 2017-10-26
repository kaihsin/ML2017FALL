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
outpath = sys.argv[3]

ID = pars.val['ID']
learn_rate = float(pars.val['learn_rate'])
Nepoch     = int(pars.val['epoch'])
model_dir = os.path.join( os.path.join(os.path.dirname(os.path.realpath(__file__)),"model"),ID)

### Read Data:
Xreader = Data_reader()
Xreader.Read_csv(Xpath,"big5",manipX)

NFeature = len(Xreader.data[0]) 
NSample  = len(Xreader.data)


Xmean = np.load(os.path.join(model_dir,"FS_mean.npy"))
Xstd  = np.load(os.path.join(model_dir,"FS_std.npy") )

# Feature scaling 
Xreader.data = (Xreader.data - Xmean)/Xstd


## valid / train => torch tensor
test_x = torch.from_numpy(Xreader.data)


#------------------------------------------------------------------
## create model :
#ClassifyNet = None
ClassifyNet = torch.load(os.path.join(model_dir,"model"))
#print ("[load] Net(model)")


## Validate 
accuracy = []
ClassifyNet.eval() # change to inference mode.
pred = ClassifyNet(Variable(test_x))

pred = np.round(pred.data.numpy().flatten())

f = open(outpath ,'w')
f.write("id,label\n")
for e in range(len(pred)):
	f.write("%d,%d\n"%(e+1,pred[e]))
f.close()
#print (pred)
