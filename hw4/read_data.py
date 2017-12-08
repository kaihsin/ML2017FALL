import numpy as np
import os,sys
import re

def Read_label(fpath,dic=None,is_rm_symbols=False,safe=False):
	if not os.path.exists(fpath):
		print("[ERROR][Read_label] fpath not exists.")
		exit(1)

	f = open(fpath,'r')
	senten = []
	label = []	
	loss = 0 
	for s in f:
		s = s.split(' +++$+++ ',1)
		lbl = int(s[0])
		s = s[-1].strip().rstrip('.')
		if is_rm_symbols:
			s = re.sub('[.,!?\'"()]', '', s) # delete all ,"!? ...

		s = re.sub('\s{2,}', ' ', s) # replace 2 or more spaces with only 1
		s = s.split()

		if not dic is None:
			if safe:
				for w in s:
					if w not in dic:
						print ("[ERROR][Read_label] %s not in dict"%(w))
			s = np.vectorize(lambda x: -1 if x not in dic else dic.get(x))(s)
			s = s[s>=0]
			if len(s) == 0 :
				loss += 1
				continue	
		senten.append(np.array(s))
		label.append(lbl)
	print (loss)
	senten = np.array(senten)
	label  = np.array(label,dtype=np.int)
	return label, senten

def Read_no_label(fpath,dic=None,is_rm_symbols=False,truncate=None,safe=False):
	if not os.path.exists(fpath):
		print("[ERROR][Read_no_label] fpath not exists.")
		exit(1)

	f = open(fpath,'r')
	senten = []
	raw_senten = []
	loss = 0
	loss_l  = 0 
	for s in f:
		bak_s = s
		s = s.strip()
		if s == '':
			continue

		s = s.rstrip('.')
		if is_rm_symbols:
			s = re.sub('[.,!?\'"()]', '', s) # delete all ,"!? ...

		s = re.sub('\s{2,}', ' ', s) # replace 2 or more spaces with only 1
		s = s.split()

		if not dic is None:
			if safe:
				for w in s:
					if w not in dic:
						print ("[ERROR][Read_label] %s not in dict"%(w))
			s = np.vectorize(lambda x: -1 if x not in dic else dic.get(x))(s)
			s = s[s>=0]
			if len(s) == 0 :
				loss += 1
			
			if not truncate is None:
				if len(s) > truncate:
					loss_l += 1
					continue				
				#continue
		senten.append(np.array(s))	
		raw_senten.append(bak_s)

	print (loss)
	print (loss_l)
	senten = np.array(senten)

	return senten, raw_senten 

def Read_test(fpath,dic=None,is_rm_symbols=False,safe=False):
	if not os.path.exists(fpath):
		print("[ERROR][Read_test] fpath not exists.")
		exit(1)

	f = open(fpath,'r')
	senten = []
	f.readline()
	loss = 0
	for s in f.readlines():
		s = s.split(',',1)[-1].strip()
		s = s.rstrip('.')

		if is_rm_symbols:
			s = re.sub('[.,!?\'"()]', '', s) # delete all ,"!? ...

		s = re.sub('\s{2,}', ' ', s) # replace 2 or more spaces with only 1
		s = s.split()

		if not dic is None:
			if safe:
				for w in s:
					if w not in dic:
						print ("[ERROR][Read_label] %s not in dict"%(w))
			s = np.vectorize(lambda x: -1 if x not in dic else dic.get(x))(s)
			s = s[s>=0]

			if len(s) == 0 :
				#print ("[ERROR] not enough elem in dict")
				#exit(1)
				loss += 1
		senten.append(np.array(s))
	print (loss)
	print (len(senten))
	senten = np.array(senten)
	return senten

def Read_dict(fpath):
	if not os.path.exists(fpath):
		print("[ERROR][Read_dict] fpath not exists.")
		exit(1)


	dictA = np.load(fpath)
	val = np.array(range(len(dictA)))
	dictA = dict(zip(dictA,val))
	print ("Load Dict : %d"%(len(dictA)))
	return dictA
