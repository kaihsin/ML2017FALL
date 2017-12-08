# step1 :
python 1-train2.py confv11.gru.cfg data/Dict.most.npy $1 0 
# step2 :
python 2-trasfer.py confv11.gru.cfg data/Dict.most.npy $2 
# step3 :
python 3-train2_semi.py confv11.gru.cfg data/Dict.most.npy $1 0
 
