import numpy as np 
import scipy as sp
import pandas as pd
import os,sys
from utils import *
Spth = os.path.dirname(os.path.realpath(__file__))
#print (Spth)
#exit(1)

#if len(sys.argv) < 2 :
#    print ("python train.py <.cfg> <isResume(Default=0)>")
#    exit(1)
#CfgPath = sys.argv[1]
test_pth =sys.argv[1]
Output_Path = sys.argv[2]
ID ,feature,Order, _ , _ , _ ,_ , _ = ReadConf(os.path.join(Spth,"SGDo2.cfg"))
test_x = MakeTestData_ex( ReadData(test_pth,0,2),feature )
ModelPath = os.path.join(os.path.join(Spth,"model"),ID)

# load model :
Model_W = []
SFactor = np.load(os.path.join(ModelPath,"SF.npy"))
Offset  = np.load(os.path.join(ModelPath,"OF.npy"))
for o in range(Order):
	Model_W.append(np.load(os.path.join(ModelPath,"W%d.npy"%(o+1))))

f = open(os.path.join(ModelPath,"b.dat"),'r')
b = np.float(f.readline())
f.close()


# scaled:
test_x = ( test_x - Offset ) * SFactor
tst_X = []
for o in range(Order):
	tst_X.append(test_x**(o+1))

tmp = Model_W[0] * tst_X[0]
for o in range(Order-1):
	tmp += Model_W[o+1] * tst_X[o+1]
 
predict_y = np.sum( tmp , axis =(1,2)) + b
np.set_printoptions(precision=6,suppress=True)

#print (len(predict_y))
#print (predict_y)

## Output :
###f = open(os.path.join(ModelPath,"Output.csv"),'w')
f = open(Output_Path,'w')
#header:
f.write("id,value\n")
for t in range(len(predict_y)):
	f.write("id_%d,%f\n"%(t,predict_y[t]))
f.close()

