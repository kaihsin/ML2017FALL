import numpy as np 
import os,sys
from data_reader import *
from parser import *
import pandas as pd

def process(ipt):
    return ipt[:,1:]

testpath = sys.argv[2]
model_dir = 'model/auto'
sav_path = sys.argv[3]

pars = Parser()
pars.parse(sys.argv[1])
ID = pars.val['ID']
model_dir = os.path.join(model_dir,ID)

prob = Data_reader()
prob.Read_csv(testpath,Format='utf-8',func=process)   
Task = prob.data

pred_label = np.load(os.path.join(model_dir,'pred.npy'))
Ans = []

Ans = []
Step = 4000

for i in np.arange(0,len(Task),Step):
    print ("%d/%d"%(i,len(Task)))
    endl = i+ Step
    if endl > len(Task):
        endl = len(Task)
    idx = Task[i:endl]
    out = pred_label[idx]
    Ans.append(out[:,0] == out[:,1])

#f.close()

Ans = np.concatenate(Ans).astype(np.int)

#print (Ans)
#print (len(Ans))

df = pd.DataFrame(data=Ans,index=range(len(Ans)))
df.index.name = 'ID'
df.to_csv(sav_path,encoding='utf-8',index=True,header=["Ans"])

#f = open(os.path.join(model_dir,'Out.csv'),'w')
#f.write("ID,Ans\n")
