import numpy as np 
#import scipy as sp 
import pandas as pda
import os,sys
from err import *

class Data_reader:
	def __init__(self):
		self.spath = ""
		self.data = None

	def Read_csv(self,data_path,Format="big5",func=None):
		try :
			if not os.path.exists(data_path) :		
				raise err.ERROR("[Data_Streamer][Read] data_path not exists.");
		except err.ERROR as e:
			print (str(e))
			exit(1)

		self.spath = data_path
		self.data = pda.read_csv(self.spath,encoding = Format)
		self.data = np.array(self.data.ix[:].values)
		#print (self.DataFrame)
		
		if not func is None :
			self.data = func(self.data)
 		
	
		

	
