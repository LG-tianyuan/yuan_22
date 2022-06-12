import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from megatron.tokenizer import tokenization_enc_dec
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

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  
 
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
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

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale:
        Return:
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out

class transformer_block(nn.Module):
	def __init__(self,
				 seq_len=2048,
				 vocab_size = vocab_size,
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
		self.embedding = Embeddings(vocab_size,d_model)
		self.position_embedding = Positional_Encoding(d_model,seq_len,dropout,device)
		self.encoder = Encoder(d_model,h,d_ff,dropout)
		self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(N)])

	def forward(self,x):
		print(x)
		out = self.embedding(x)
		print(out)
		print(len(out))
		print(len(out[0]))
		out = self.position_embedding(out)
		print(out)
		print(len(out))
		print(len(out[0]))
		for encoder in self.encoders:
			out = encoder(out)
		return out

transformer_model = transformer_block()
for i in range(len(train_data)-1):
	Input = train_data[i]
	print(Input.shape)
	out = transformer_model(Input) 
	print(out)	

