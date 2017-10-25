import numpy as np
import os,sys
from . import err

class Parser :
	def __init__(self):
		self.val = {}

	def parse(self,rcpath):
		try:
			if not os.path.exists(rcpath):
				raise err.ERROR("[Parser][parse] path not exists.")

		except err.ERROR as e:
			print (str(e))
			exit(1)
	
		f = open(rcpath,'r')
		lines = f.readlines()
		f.close()
		for line in lines:		
			if '#' in line:
				continue
			if '=' in line:
				key = line.split('=')[0]
				v = line.split('=')[1]
				key = key.strip()
				v = v.strip()
				self.val[key] = v

	def print_vals(self):
		print ("========================")
		for key in self.val.keys():
			print ("%s = "%(key),self.val[key])
		print ("========================")

