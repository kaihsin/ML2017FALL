import numpy as np 
#import scipy as sp
import pandas as pd
import os,sys
from utils import *
# simple linear regration using simple GD

if len(sys.argv) < 2 :
	print ("python train.py <.cfg> <isResume(Default=0)>")
	exit(1)
CfgPath = sys.argv[1]
is_ReadWb = False
if len(sys.argv) > 2:
	is_ReadWb = int(sys.argv[2])

#lmbd = 0.01
#if len(sys.argv) > 3:
#	lmbd = float(sys.argv[3])	

ID, feature , Order ,N_iter , learn_rate,lmbd, train_pth , _ = ReadConf(CfgPath)

#learn_rate = 0.001
ModelPath = os.path.join("model",ID)
if not os.path.exists(ModelPath):
	os.system("mkdir %s"%(ModelPath))

## training data:
# train_x : [NBatch x NDays x (NTypes x 9hrs)]
# train_y : [NBatch x NDays ] 
train_x,train_y = MakeTrainData_ex( ReadData(train_pth,1,3),feature )
shape = np.shape(train_x)
Batch_N  = shape[0]
Batch_Sz = shape[1]
Type_N  = shape[2]
Hour_N  = shape[3]


## flat batches :
train_x = train_x.reshape((Batch_N*Batch_Sz,Type_N,Hour_N))
train_y = train_y.reshape((Batch_N*Batch_Sz))
Batch_Sz = Batch_N*Batch_Sz
Batch_N  = 1



## Feature Scaling :
#[ Ntypes x 9hrs ]
#print (Batch_Sz)
std_ij  = [ [ np.std(train_x[:,i,j]) for j in range(Hour_N) ] for i in range(Type_N) ]
mean_ij = [ [ np.mean(train_x[:,i,j]) for j in range(Hour_N) ] for i in range(Type_N) ]
SFactor = np.array(std_ij)**-1
Offset  = np.array(mean_ij)
train_x = SFactor * (train_x - Offset)

trn_X = []
for o in range(Order):
	trn_X.append(train_x**(o+1))

# Init weight
Model_W = []
b = None
if is_ReadWb :

	# read W 
	for o in range(Order):
		Model_W.append( np.load(os.path.join(ModelPath,"W%d.npy"%(o+1))) )

	f = open(os.path.join(ModelPath,"b.dat"),'r')
	b = np.float(f.readline())
	f.close()
	
	print ("[Read Ws,b]")

else :
	for o in range(Order):
		Model_W.append( 0.1*np.ones((Type_N,Hour_N)))
	b = 0.1




## GD 
 
for itr in range(N_iter):
	tmp = Model_W[0] * trn_X[0] 
	regu = np.sum(lmbd*Model_W[0]**2)
	for o in range(Order-1):
		tmp += Model_W[o+1] * trn_X[o+1]
		regu += np.sum(lmbd*Model_W[o+1]**2)
	#regu /= Order 
	new_y = np.sum( tmp , axis =(1,2)) + b  
	diff_y = train_y - new_y
	D_yOp = diff_y[:,np.newaxis,np.newaxis]
	
	#print (np.shape(d_y))
	Loss_reduce = np.mean ( ( diff_y )**2)  + regu/len(diff_y)
	
	for o in range(Order):
		grad_W = np.mean(-2*D_yOp*trn_X[o],axis=0)
		Model_W[o] = Model_W[o] - learn_rate * (grad_W + 2*lmbd * Model_W[o])

	grad_b = np.mean(-2*diff_y)
	b = b - learn_rate * grad_b	
	
	

	if itr%100== 0 :
		print (Loss_reduce)


# Save weights:
print ("Save Ws Matrix.")
for o in range(len(Model_W)):
	np.save(os.path.join(ModelPath,"W%d"%(o+1)),Model_W[o]) # .npy

print ("Save b.")
f = open (os.path.join(ModelPath,"b.dat"),"w")
f.write("%11.12f"%(b))
f.close()

print ("Save SFactor Array.")
np.save(os.path.join(ModelPath,"SF"),SFactor)

print ("Save Offset Array.")
np.save(os.path.join(ModelPath,"OF"),Offset)





