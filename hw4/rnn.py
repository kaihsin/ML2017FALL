import torch 
import torch.nn as nn
from torch.autograd import Variable



"""
	Simple RNN using nn.LSTM unit

"""

class rnn(nn.Module):
	def __init__(self,vocab_size,embedding_dim,Padd_index):
		super(rnn,self).__init__()
		self.emb = nn.Embedding(
				num_embeddings = vocab_size,
				embedding_dim = embedding_dim,
				padding_idx = Padd_index
		)
		
		self.rnn = nn.LSTM(
			input_size = embedding_dim,
			hidden_size = 64,
			num_layers = 1,
			batch_first = True
		)
		self.out = nn.Linear(64,1)
		self.sgactiv  = nn.Sigmoid()
	
	def forward(self,x):
		# x : [batch x time ]
		#x = x.long()
		embd = self.emb(x)	
		# embd : [batch x time x embedding_dim]
		rnn_o , (h_n,h_c) = self.rnn(embd,None) 
		out  = self.out(rnn_o[:,-1,:])	# get last 
		out  = self.sgactiv(out)
		#print (out)
		return out

class rnn2(nn.Module):
	def __init__(self,vocab_size,embedding_dim,Padd_index):
		super(rnn2,self).__init__()
		self.emb = nn.Embedding(
				num_embeddings = vocab_size,
				embedding_dim = embedding_dim,
				padding_idx = Padd_index
		)
		
		self.rnn = nn.LSTM(
			input_size = embedding_dim,
			hidden_size = 128,
			num_layers = 1,
			batch_first = True
		)
		self.out = nn.Sequential(
			nn.Linear(128,64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(64,1)
		)
		self.sgactiv  = nn.Sigmoid()
	
	def forward(self,x):
		# x : [batch x time ]
		#x = x.long()
		embd = self.emb(x)	
		# embd : [batch x time x embedding_dim]
		rnn_o , (h_n,h_c) = self.rnn(embd,None) 
		out  = self.out(rnn_o[:,-1,:])	# get last 
		out  = self.sgactiv(out)
		#print (out)
		return out

class cnn_rnn_GRU(nn.Module):
	def __init__(self,vocab_size,embedding_dim,Padd_index):
		super(cnn_rnn_GRU,self).__init__()
		self.emb = nn.Embedding(
				num_embeddings = vocab_size,
				embedding_dim = embedding_dim,
				padding_idx = Padd_index
		)
		
		self.Kern_sz = 3
		self.Ochann  = 128
		self.cnn = nn.Sequential(
			nn.Conv2d(
                in_channels  = 1,
                out_channels = self.Ochann,
                kernel_size  = [self.Kern_sz,embedding_dim],
                stride       = 1,
                padding      = (1,0)
			),
			nn.BatchNorm2d(self.Ochann),
			nn.ReLU(),
			nn.Dropout2d(0.1),
			#nn.MaxPool2d((2,1))
			nn.Conv2d(
                in_channels  = self.Ochann,
                out_channels = self.Ochann*2,
                kernel_size  = [self.Kern_sz,1],
                stride       = 1,
                #padding      = (0,1)
			),
			nn.BatchNorm2d(self.Ochann*2),
			nn.ReLU(),
			nn.Dropout2d(0.1),
			nn.MaxPool2d((2,1))
		)

		self.rnn = nn.GRU(
			input_size = self.Ochann*2,
			hidden_size = 128,
			num_layers = 1,
			batch_first = True
		)
	
		self.out = nn.Sequential(
			nn.Linear(128,64),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(64,1)
		)
		
		self.sgactiv  = nn.Sigmoid()
	
	def forward(self,x):
		# x : [batch x time ]
		#x = x.long()
		embd = self.emb(x)	
		sz = embd.size()
		Slen = sz[1]
		Ebd_sz = sz[2]

		embd = embd.view(-1,1,sz[1],sz[2])
		#print (embd.size())
		embd = self.cnn(embd)
		#print (embd.size())
		embd = embd.view(-1,self.Ochann*2,int((Slen-self.Kern_sz+1)/2))
		#print (embd.size())
		#exit(1)
		embd = embd.permute(0,2,1)
		#print (embd.size())
		#exit(1)
		# embd : [batch x time x embedding_dim]
		rnn_o , h = self.rnn(embd,None) 
		out  = self.out(rnn_o[:,-1,:])	# get last 
		out  = self.sgactiv(out)

		#print (out)
		return out
