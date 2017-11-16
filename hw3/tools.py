import numpy as np 
import scipy as sp
import scipy.ndimage

def Hflip(ipt):
	return np.flipud(ipt)

def Vflip(ipt):
	return np.fliplr(ipt)

def RandVflip(ipt,ratio,axis):
	out = np.copy(ipt)
	rnd = np.random.choice(len(ipt),int(len(ipt)*ratio),replace=False)
	if not len(rnd):
		return ipt
	out[rnd] = np.apply_along_axis(lambda a : np.flip(a,0),axis,out[rnd])
	return out

def RandHflip(ipt,ratio,axis):
	out = np.copy(ipt)
	rnd = np.random.choice(len(ipt),int(len(ipt)*ratio),replace=False)
	if not len(rnd):
		return ipt
	
	out[rnd] = np.apply_along_axis(lambda a : np.flip(a,0),axis+1,out[rnd])
	return out

def RandRotate(ipt,ratio,axis,deg):
	out = np.copy(ipt)
	rnd = np.random.choice(len(ipt),int(len(ipt)*ratio),replace=False)
	if not len(rnd):
		return ipt
	if axis == 0 :
		print ("[ERROR] RandRotatae axis must >=1")
		exit(1)

	out[rnd] = sp.ndimage.rotate(out[rnd],deg,axes=(axis+1,axis),reshape=False,mode='nearest')
	return out

"""
A = np.arange(5*16).reshape(5,1,4,4)
#np.fliplr(A[1])
print (A)
print (RandRotate(A,1.00,axis=2,deg=90))
"""


