import numpy as np 
import os,sys



def Padding(sentense,padd_id,size=None):
	maxx = np.max(np.vectorize(lambda x: len(x))(sentense))
	if not size is None :
		if size < maxx:
			print ("ERROR Padding, size cannot smaller than maxx")
			print (maxx)
			exit(1)
		maxx = size

	out = []
	for id in range(len(sentense)):
		out.append(np.pad(sentense[id],(0,maxx-len(sentense[id])),'constant',constant_values=(0,padd_id)))
		
	return np.array(out)

