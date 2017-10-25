import numpy as np 
import os,sys,math

class Variable:
	def __init__(self,data):
		self.data = data

	def __getitem__(self,key):
		return self.data[key]

	def __setitem__(self,key,value):
		self.data[key] = value	

	def shuffle(self):
		np.random.shuffle(self.data)

	
