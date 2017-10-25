import numpy as np 
#import scipy as sp
import os,sys


def sigmoid(z):
	return np.clip((1. + np.exp(-z))**-1,1.0E-14,0.99999999999999)


class Neuron:
	def __init__(self,dim):
		self.dim = dim
		self.W   = 0.01*np.random.randn(dim)
		self.b   = 0.0
		#self.output_layer = output_layer
	
	def infer(self,x,output_layer=None):
		"""
			@input : 
				x : [NBatch x Ndim]

			@output:
				y : [NBatch]		
				
			@description :
				take input x and evaluate y = Wx + b				
	
		"""
		if output_layer is None :
			return np.dot(x,self.W) + self.b
		else:
			return output_layer ( np.dot(x,self.W) + self.b )	

	def save(self,sav_path):
		#np.append(self.W,b)
		np.save(sav_path,np.append(self.W,self.b))
	def load(self,rpath):
		self.W = np.load(rpath)
		self.b = self.W[-1]
		self.W = np.delete(self.W,-1,0)



	


