import numpy as np 
import os,sys
from skimage import io
#from skimage.viewer import ImageViewer 
from pca import *


L        = 600
channels = 3

# Dir:
datadir = sys.argv[1]
#modeldir = sys.argv[2]
input_pic = os.path.join(datadir,sys.argv[2])

mean_face, es,eigVs = None,None,None

files   = os.listdir(datadir)
print("[loading]")
imgs = np.array([ io.imread(os.path.join(datadir,f)) for f in files ])
print ("done loading")
Ndata = len(imgs)
imgs = imgs.reshape((Ndata,L**2*channels))
print ("pca")
mean_face ,es , eigVs = PCA(imgs,keep_dim=20)
#mean_face = mean_face.reshape((L,L,channels))
#eigVs     = eigVs.reshape((len(eigVs),L,L,channels))    
#print("save")
#np.save(os.path.join(modeldir,'mean_face'),mean_face)
#np.save(os.path.join(modeldir,'eig_face' ),eigVs   )
#np.save(os.path.join(modeldir,'eig_val'  ),es      )
#
#else:
#    print ("Load")
#    mean_face = np.load(os.path.join(modeldir,'mean_face.npy'))
#    es        = np.load(os.path.join(modeldir,'eig_val.npy'))
#    eigVs     = np.load(os.path.join(modeldir,'eig_face.npy'))

## reconstruction :
print ("recon.")
raw_img = io.imread(input_pic).reshape((L**2*channels))
raw_img = raw_img-mean_face
eigVs   = eigVs[:4]
rec_img = np.dot(np.dot(raw_img,eigVs.T),eigVs) 
        
rec_img = rec_img + mean_face
rec_img -= np.min(rec_img)
rec_img /= np.max(rec_img)
rec_img = (rec_img*255).astype(np.uint8)

io.imsave("reconstruct.jpg",rec_img.reshape((L,L,channels)))
print ("done")



#mean_fig = mean_fig.astype(np.uint8).reshape((Ly,Lx,channels))
#ImageViewer(mean_fig).show()
#eV10 = eigVs[9]
#eV10 -= np.min(eV10)
#eV10 /= np.max(eV10)
#eV10 = (eV10*255).astype(np.uint8).reshape((Ly,Lx,channels))
#ImageViewer(eV10).show()


#ImageViewer(imgs[0]).show()







