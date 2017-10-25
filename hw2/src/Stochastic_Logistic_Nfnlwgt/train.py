import os,sys,time
sys.path.append("../")
from myutils.data_reader import *
from myutils.neuron import *
from myutils.variable import *
from myutils.parser import *

np.random.seed(int(time.time()))

def manip(ipt):
	## manipulate the data-set here to the data in dstream
	## ipt : ndarray 
	ipt[ipt==' ?'] = '-1' 
	To_process = [1,3,5,6,7,8,9,13]
	for i in To_process:
		keys = np.unique(ipt[:,i])
		#print (keys)
		dic_keys = dict(zip(keys,np.arange(len(keys))))
		for key in keys:
			ipt[ipt[:,i]==key,i] = dic_keys[key]
	ipt[ipt[:,14]==' <=50K',14] = 0
	ipt[ipt[:,14]==' >50K',14]  = 1	
	ipt = ipt.astype(np.float)
	ipt = np.delete(ipt,2,1)	
	#print (ipt)
	#print (keys)
		
	return ipt

#===========================================================================
pars = Parser()
pars.parse(sys.argv[1])
spath = sys.argv[2]
is_res = 0
if len(sys.argv)>3:
	is_res = int(sys.argv[3])


ID         = pars.val['ID']
Nepoch     = int(pars.val['epoch'])
learn_rate = float(pars.val['learn_rate'])
cut_rate   = float(pars.val['cut_rate'])
regu_rate  = float(pars.val['regu_rate'])
order      = int(pars.val['order'])

model_dir = os.path.join("model",ID)

print ("ID = %s"%(ID))
print ("epoch = %d"%(Nepoch))
print ("learn_rate = %f"%(learn_rate))
print ("cut_rate = %f"%(cut_rate))
print ("regu_rate = %f"%(regu_rate))
print ("order = %d"%(order))

## prepare 
if not os.path.exists(model_dir) :
	os.system("mkdir %s"%(model_dir))


## Read Data
raw = Data_reader()
raw.Read_csv(spath,"big5",manip)
NFeature = len(raw.data[0]) - 1
NSample  = len(raw.data)
NTrain   = int(NSample*cut_rate)
NValida  = NSample - NTrain


## features:
all_x = raw.data[:,0:-1]
FS_mean = None
FS_std  = None

## suffle indx for cut :
idshufmap = None

## Define Model :
model = Neuron(NFeature*order)
if is_res :
	print ("Load model")
	model.load(os.path.join(model_dir ,"model.npy"))
	print ("Load shufmap")
	idshufmap = np.load(os.path.join(model_dir,"shufmap.npy"))
	print ("Load FS_mean")
	FS_mean = np.load(os.path.join(model_dir,"FS_mean.npy"))
	print ("Load FS_std")
	FS_std  = np.load(os.path.join(model_dir,"FS_std.npy"))
	
else:
	idshufmap = np.arange(NSample)
	np.random.shuffle(idshufmap)

	FS_mean = np.mean(all_x,axis=0)
	FS_std  = np.std(all_x,axis=0)
	#print (np.shape(FS_mean))	
	## Save mean & stdiv:
	np.save(os.path.join(model_dir,"FS_mean"),FS_mean)
	np.save(os.path.join(model_dir,"FS_std"),FS_std)


## Feature scaling :
all_x = (all_x - FS_mean)/FS_std

## x & y , train + valid
train_x = all_x[idshufmap[0:NTrain]]
train_y = raw.data[idshufmap[0:NTrain],-1]
validate_x = all_x[idshufmap[NTrain:NSample]]
validate_y = raw.data[idshufmap[NTrain:NSample],-1]

for i in range(order-1):
	train_x = np.hstack((train_x,train_x[:,:NFeature]**(i+1)))
	validate_x = np.hstack((validate_x,validate_x[:,:NFeature]**(i+1)))

#print (train_y)
idxmap  = np.arange(len(train_x))

## SGD
for epoch in range(Nepoch):
	np.random.shuffle(idxmap)
	acpt = 0
	for n in idxmap:
		f = sigmoid( model.infer(train_x[n]) )
		acpt += (1. - np.abs(np.round(f) - train_y[n]))
		Xentropy = -(train_y[n]*np.log(f) + (1.-train_y[n])*np.log(1.-f))

		grad_b = (f  - train_y[n])
		grad_W = grad_b*train_x[n] + regu_rate*model.W

		model.W -= learn_rate*grad_W
		model.b -= learn_rate*grad_b

	acpt /= len(idxmap)
	print ("%3d] Xs: %4.8f Ac: %4.8f"%(epoch,Xentropy,acpt))
	
## Save Model
model.save(os.path.join(model_dir,"model"))
np.save(os.path.join(model_dir,"shufmap"),idshufmap)

## Validation :
f = sigmoid( model.infer(validate_x))
f = np.round(f)

accuracy = (1. - np.mean(np.abs(f-validate_y)))
print (accuracy)




