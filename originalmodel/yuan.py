import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from megatron.tokenizer import tokenization_enc_dec
import math
#from transformers import AutoTokenizer

vocab = []
fr = open("vocab.txt",encoding="utf-8")
for line in fr:
	vocab.append(line.strip())
fr.close()
vocab_size = len(vocab)

word2index = { w: i for i,w in enumerate(vocab) }
index2woed = { i: w for i,w in enumerate(vocab) }

#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 
tokenizer = tokenization_enc_dec.EncDecTokenizer(vocab_file="vocab.txt")

train_data = []
fr = open("data/test.txt",encoding="utf-8")
for line in fr:
    temp = line.strip()
    cut_list = tokenizer.tokenize(temp)
    #cut = jieba.cut(temp)
    #cut_list = [x for x in cut]
    if len(cut_list) == 0:
        continue
    else:
        print(cut_list)
    word_index = []
    for i,word in enumerate(cut_list):
        index = 0
        if word in vocab:
            index = word2index[word]
        word_index.append(index)
    train_data.append(torch.tensor(word_index))
fr.close()
train_data.append(torch.zeros(2048))
train_data = pad_sequence([train_data[i] for i in range(len(train_data))],batch_first=True)
print(train_data)

'''
embedding = nn.Embedding(vocab_size,512)
for i in range(len(train_data)):
    print(i)
    print(train_data[i])
    print(embedding(torch.tensor(train_data[i],dtype=torch.int)))
'''

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, embed, pad_size, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out



class transformer_block(nn.Module):
	def __init__(self,
				 vocab_size = vocab_size,
         seq_len = 2048,
				 N = 40,
				 d_model = 480,
				 d_ff = 3072,
				 h = 24,
				 d_k = 64,
				 d_v = 64,
				 dropout = 0.1,
				 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
				 ):
		super(transformer_block,self).__init__()
		self.embedding = nn.Embedding(vocab_size,d_model)
		self.position_embedding = PositionalEncoding(d_model,seq_len,dropout,device)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model,h,d_ff,dropout)
		self.encoder = nn.TransformerEncoder(self.encoder_layer,N)
		#self.wv = nn.Linear(d_model,vocab_size, bias = False)

	def forward(self,x):
		Input = self.embedding(x[0])
		Input = self.position_embedding(Input).unsqueeze(0)
		for i in range(1,len(x)):
			temp = self.embedding(x[i])
			temp = self.position_embedding(temp).unsqueeze(0)
			Input = torch.cat((Input,temp),0)
		print(Input.shape)
		out = self.encoder(Input)
		#out = self.wv(out)
		return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = transformer_block()
model = model.to(device)
train_data = train_data.to(device)
print(train_data.shape)
out = model(train_data) 
print(out)
	
'''
transformer_model = transformer_block()
for i in range(len(train_data)-1):
	Input = train_data[i].unsqueeze(0)
	print(Input.shape)
	out = transformer_model(Input) 
	print(out)
'''