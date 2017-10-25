import os,sys
#sys.path.append("../")
from myutils.data_reader import *
from myutils.neuron import *
from myutils.variable import *
from myutils.parser import *

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
	#ipt[ipt[:,14]==' <=50K',14] = 0
	#ipt[ipt[:,14]==' >50K',14]  = 1	
	ipt = ipt.astype(np.float)
	ipt = np.delete(ipt,2,1)	
	#print (ipt)
	#print (keys)
		
	return ipt

#===========================================================================
config  = sys.argv[1]
spath   = sys.argv[2]
outpath = sys.argv[3]

pars = Parser()
pars.parse(config)
order = int(pars.val["order"])


##Model path 
model_path = os.path.join( os.path.dirname( os.path.realpath(__file__) ) , "model/%s"%(pars.val['ID']) )

## Read Data
raw = Data_reader()
raw.Read_csv(spath,"big5",manip)
NFeature = len(raw.data[0])


## feature scaling:
test_x = raw.data 
FS_mean = np.load(os.path.join(model_path,"FS_mean.npy"))
FS_std  = np.load(os.path.join(model_path,"FS_std.npy"))
test_x = (test_x - FS_mean)/FS_std

for i in range(order-1):
    test_x = np.hstack((test_x,test_x[:,:NFeature]**(i+1)))




## Define Model :
model = Neuron(NFeature)
model.load(os.path.join(model_path,"model.npy"))

## test :
f = sigmoid(model.infer(test_x))
f = np.round(f)

## Save : 
fout = open(outpath,'w')
fout.write("id,label\n")
for i in range(len(f)) :
	fout.write('%d,%d\n'%(i+1,f[i]))
fout.close()



 









