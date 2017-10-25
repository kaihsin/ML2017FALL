import os,sys,time
#sys.path.append("../")
from myutils.data_reader import *
from myutils.parser import *
from myutils.neuron import *
#np.random.seed(int(time.time()))
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
Xpath = sys.argv[1]
model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"model/GM91")
outpath = sys.argv[2]

if not os.path.exists(model_dir):
	print ("[ERROR] model not exists.")
	exit(1)


## Read Data
Xreader = Data_reader()
Xreader.Read_csv(Xpath,"big5",manipX)

NFeature = len(Xreader.data[0])
NSample  = len(Xreader.data)


## model 
model = Neuron(NFeature)
model.load(os.path.join(model_dir,"model.npy"))

## Validation :
predic_y = np.round( model.infer(Xreader.data,sigmoid) ) 

f = open(outpath,'w')
f.write("id,label\n")
for y in range(len(predic_y)):
	f.write("%d,%d\n"%(y+1,predic_y[y]))
f.close()




