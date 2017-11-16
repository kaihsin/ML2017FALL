import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os,sys


class cnn_fix_v8(nn.Module):
	def __init__(self,Fig_y,Fig_x,outclass):
		super(cnn_fix_v8,self).__init__()

		## checking :
		
		self.Conv = nn.Sequential( ## input [1 . Y . X ]
			nn.Conv2d(
				in_channels=1,
				out_channels = 32,
				kernel_size  = 3,
				stride       = 1,
				padding      = 1
			),
			nn.BatchNorm2d(32),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.Conv2d(
				in_channels=32,
				out_channels = 32,
				kernel_size  = 3,
				stride       = 1,
				padding      = 1
			),
			nn.BatchNorm2d(32),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.Conv2d(
				in_channels=32,
				out_channels = 64,
				kernel_size  = 3,
				stride       = 1,
				padding      = 1
			),
			nn.BatchNorm2d(64),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			
			nn.MaxPool2d(2),

			nn.Conv2d(
				in_channels=64,
				out_channels = 64,
				kernel_size  = 5,
				stride       = 1,
				padding      = 2
			),
			nn.BatchNorm2d(64),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.Conv2d(
				in_channels=64,
				out_channels = 64,
				kernel_size  = 5,
				stride       = 1,
				padding      = 2
			),
			nn.BatchNorm2d(64),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.Conv2d(
				in_channels=64,
				out_channels = 128,
				kernel_size  = 5,
				stride       = 1,
				padding      = 2
			),
			nn.BatchNorm2d(128),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.MaxPool2d(2),
			nn.Conv2d(
				in_channels=128,
				out_channels = 128,
				kernel_size  = 5,
				stride       = 1,
				padding      = 2
			),
			nn.BatchNorm2d(128),
            nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.MaxPool2d(2),
		).double() ## > output [ F_num , (Fig_x-Fsz)/Stride/2 , (Fig_x-Fsz)/Stride/2 ]
		
		
		#print (F_num*int((Fig_y-F_sz)/Stride/2) * int((Fig_x-F_sz)/Stride/2))	
		self.DNN = nn.Sequential(
			nn.Linear(128*6*6, 256 ),
			nn.BatchNorm1d(256),
			nn.Dropout(0.2),
			nn.ReLU(),
			nn.Linear(256, 512 ),
			nn.BatchNorm1d(512),
			nn.Dropout(0.7),
			nn.ReLU(),
			nn.Linear(512, outclass ),
			#nn.LogSoftmax()
		).double()
		
	def forward(self,x):
		o  = self.Conv(x)
		

		o = o.view(o.size(0),-1)
		
		o = self.DNN(o)
		#exit(1)
		#print (o)

		return o


