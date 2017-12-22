import numpy as np 
## data processing func:
def movie_fx(ipt):
    
    label = np.array(ipt[:,0],dtype=np.int)
    
    ipt   = np.array([np.array(ipt[t,1].split(' '),dtype=np.float) for t in range(len(ipt))])

    return ipt
