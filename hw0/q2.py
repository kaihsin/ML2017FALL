from PIL import Image
from PIL import ImageEnhance
import os,sys

ipt_file = sys.argv[1]
#read image
img = Image.open(ipt_file)

#test:
#pix1 = list(img.getdata())
enhancer = ImageEnhance.Brightness(img)
imgw = enhancer.enhance(0.5)
#pix2  = list(imgw.getdata())
imgw.save("Q2.jpg","JPEG")

#for i in range(10):
#	print(pix1[i],pix2[i])



