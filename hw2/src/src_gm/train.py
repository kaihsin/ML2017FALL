import os,sys,time
#sys.path.append("../")
from myutils.data_reader import *
from myutils.parser import *
from myutils.neuron import *
np.random.seed(int(time.time()))
#np.random.seed(99)
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
is_res = 0
if len(sys.argv)>4:
	is_res = int(sys.argv[4])


ID         = pars.val['ID']
cut_rate   = float(pars.val['cut_rate'])
model_dir = os.path.join("model",ID)



print ("ID = %s"%(ID))
print ("cut_rate = %f"%(cut_rate))

## prepare 
if not os.path.exists(model_dir) :
	os.system("mkdir %s"%(model_dir))


## Read Data
Xreader = Data_reader()
Xreader.Read_csv(Xpath,"big5",manipX)
Yreader = Data_reader()
Yreader.Read_csv(Ypath,"big5",manipY)

NFeature = len(Xreader.data[0])
NSample  = len(Xreader.data)
NTrain   = int(NSample*cut_rate)
NValida  = NSample - NTrain

## suffle indx for cut :
idshufmap = None

## Define Model :
if is_res :
	print ("Load shufmap")
	idshufmap = np.load(os.path.join(model_dir,"shufmap.npy"))
	
else:
	#idshufmap = np.arange(NSample)
	idshufmap = np.random.permutation(NSample)
	


## x & y , train + valid
train_x = Xreader.data[idshufmap[0:NTrain]]
train_y = Yreader.data[idshufmap[0:NTrain]]
validate_x = Xreader.data[idshufmap[NTrain:NSample]]
validate_y = Yreader.data[idshufmap[NTrain:NSample]]


## classify :
C1_x = train_x[np.argwhere(train_y==1).flatten()]
C2_x = train_x[np.argwhere(train_y==0).flatten()]
C1_mean = np.mean(C1_x,axis=0)
C2_mean = np.mean(C2_x,axis=0)
C1_cov = np.cov(C1_x.T)
C2_cov = np.cov(C2_x.T)
P_C1 = len(C1_x) / (len(C1_x)+len(C2_x))
P_C2 = 1.-P_C1
del C1_x
del C2_x

## Share Cov : 
inv_ShCov = ( P_C1*C1_cov  + P_C2*C2_cov )
del C1_cov
del C2_cov

## check ill-defined:
if not np.linalg.matrix_rank(inv_ShCov) == inv_ShCov.shape[0] :
	inv_ShCov = np.linalg.pinv(inv_ShCov)
	#print ("ill-defined ")
else :
	inv_ShCov = np.linalg.inv(inv_ShCov)


model = Neuron(NFeature)
model.W = np.dot((C1_mean - C2_mean),inv_ShCov)
model.b = -0.5*( np.dot(np.dot(C1_mean,inv_ShCov),C1_mean.T) - np.dot(np.dot(C2_mean,inv_ShCov),C2_mean.T) )+ np.log(P_C1/P_C2)

## Save model :
model.save(os.path.join(model_dir,"model"))

np.save(os.path.join(model_dir,"shufmap"),idshufmap)


## Validation :
predic_y = np.round( model.infer(validate_x,sigmoid) ) 
print (np.shape(predic_y))
print (np.shape(validate_y))
accuracy = 1. - np.mean( np.abs(predic_y-validate_y) ) 
print (accuracy)


