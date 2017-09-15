import os,sys

ipt_file = sys.argv[1] 

f = open(ipt_file,'r')
raw_Words = []
lines = f.readlines()
for line in lines:
	line = line.split('\n')[0]
	raw_Words.extend(line.split(' '))
f.close() 

#print(raw_Words)
#idxs = sorted(range(len(raw_Words)), key=lambda k: raw_Words[k])
#Sorted_Wrd = [ raw_Words[s] for s in idxs ] 

my_dict = {i:raw_Words.count(i) for i in raw_Words}
Keys = list(my_dict.keys())
#print(Keys)
#exit(1)

key_p = [ raw_Words.index(key) for key in Keys ] 
mp = sorted(range(len(key_p)), key=lambda k: key_p[k])

f = open("Q1.txt",'w')
for i in range(len(mp)) :
	#print(Keys[mp[i]])
	if i == len(mp) - 1:
		f.write("%s %d %d"%(Keys[mp[i]],i,my_dict[Keys[mp[i]]]))
	else:
		f.write("%s %d %d\n"%(Keys[mp[i]],i,my_dict[Keys[mp[i]]]))
	
	#print("%s %d %d\n",key,my_dict[key],
f.close()
