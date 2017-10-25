import os,sys

class ERROR(Exception):
	def __init__(self,msg):
		self.msg = msg

	def __str__(self):
		return "[ERROR] " + self.msg

	
