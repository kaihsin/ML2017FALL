import numpy as np
import pandas as pd
import scipy as sp 
#import scipy,pandas
import os,sys
#from utils import *

def ReadConf(cfgfpath):
	train_pth = None
	test_pth  = None
	feature = None
	Order = 1
	N_iter = 1
	learn_rate = 0.001
	BatchSz = 1
	ID = None
	f = open(cfgfpath,'r')
	lines  = f.readlines()
	for line in lines : 
		if 'Train' in line:
			train_pth = line.split('=')[-1]
			train_pth = train_pth.split('\n')[0]
			train_pth = train_pth.strip()
		elif 'Test' in line:
			test_pth  = line.split('=')[-1]
			test_pth  = test_pth.split('\n')[0]
			test_pth = test_pth.strip()
		elif 'Feature' in line	:
			line  = line.split('=')[-1]
			line = line.split('\n')[0]
			line = line.strip()
			if not 'ALL' in line:
				feature = np.array(line.split(' '),np.int)
		elif 'ID' in line:
			line = line.split('=')[-1]
			line = line.split('\n')[0]
			line = line.strip()
			ID = line.strip()
		elif 'Order' in line:
			line = line.split('=')[-1]
			line = line.split('\n')[0]
			line = line.strip()
			Order = int(line)
		elif 'N_iter' in line:
			line = line.split('=')[-1]
			line = line.split('\n')[0]
			line = line.strip()
			N_iter = int(line)
		elif 'learn_rate' in line:
			line = line.split('=')[-1]
			line = line.split('\n')[0]
			line = line.strip()
			learn_rate = float(line)
		elif 'BatchSz' in line:
			line = line.split('=')[-1]
			line = line.split('\n')[0]
			line = line.strip()
			BatchSz = int(line)
	f.close()
	if ID is None:
		print ("ERROR, ID not set in conf.")
		exit(1)
	return ID, feature, Order, N_iter,learn_rate,BatchSz, train_pth, test_pth


def ReadData(fpath,skiprow,skipcol):
	raw = pd.read_csv(fpath,header=None,skiprows=skiprow,encoding = "big5")
	#print (raw)
	raw = raw.ix[:,skipcol:len(raw.columns)]
	raw = np.array(raw.values)
	#raw = np.array(raw.as_matrix) 
	#print (raw)
	#print ( raw.as_matrix)
	raw[raw=='NR'] = '-1'
	#print(raw)
	raw = raw.astype(np.float)
	return raw

def MakeTrainData(rawData):
	shape = np.shape(rawData)
	
	# [ NDay x (NTypes x Nhours) ]
	rawData = rawData.reshape((int(shape[0]/18),18,shape[1]))
	
	# 9 hour unroll :
	# [NBatch x NDay x (NTypes x 9)]
	train_x = np.array([rawData[:,:,i:i+9] for i in range(15)])
	#shape = np.shape(train_x)
	#train_x.reshape(shape[0]*shape[1],shape[2])
	
	# [NBatch x NDay ]
	train_y = np.array([rawData[:,9,i+9] for i in range(15)]) 
	#print (np.shape(train_y))

	return train_x , train_y 

def MakeTrainData_ex(rawData,Feature=None):

	shape = np.shape(rawData)
	# [ NDay x (NTypes x Nhours) ]
	rawData = rawData.reshape((int(shape[0]/18),18,shape[1]))
	# 9 hour unroll :

	# [NBatch x NDay x NTypes x 9]
	iptData = np.array([rawData[:,:,i:i+9] for i in range(15)])

	if Feature is None:
		train_x = iptData
	else:
		Feature = np.array(Feature)
		train_x = iptData[:,:,Feature,:] 
		
    
	#>>Y [NBatch x NDay ]
	train_y = np.array([rawData[:,9,i+9] for i in range(15)])
    #print (np.shape(train_y))
		
	# return will have [NBatch x NDay x NFeature x 9 ] for train_x 
	#                  [NBatch x NDay ] for train_y
	return train_x , train_y

#rawData = MakeTrainData( ReadData("data/train.csv",1,3) )

def MakeTestData(rawData):
	shape = np.shape(rawData)
	
	# [N_id x (NTypes x Nhours)]
	rawData = rawData.reshape((int(shape[0]/18),18,shape[1]))

	return rawData

def MakeTestData_ex(rawData,Feature=None):
	shape = np.shape(rawData)

	# [N_id x (NTypes x Nhours)]
	rawData = rawData.reshape((int(shape[0]/18),18,shape[1]))

	if Feature is None:
		test_x = rawData
	else:
		Feature =np.array(Feature)
		test_x = rawData[:,Feature,:]

	return test_x


