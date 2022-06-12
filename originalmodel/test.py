import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from model.Transformer import Transformer
# import sys
# sys.path.append('/home/asc/yuan/demo/')
from megatron.tokenizer import tokenization_enc_dec
#from transformers import AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator

seq_len = 2048
vocab = []
fr = open("vocab.txt",encoding="utf-8")
for line in fr:
	vocab.append(line.strip())
fr.close()
vocab_size = len(vocab)

word2index = { w: i for i,w in enumerate(vocab) }
index2word = { i: w for i,w in enumerate(vocab) }

#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 
tokenizer = tokenization_enc_dec.EncDecTokenizer(vocab_file="vocab.txt")

train_data = []
fr = open("../data/test.txt",encoding="utf-8")
for line in fr:
    temp = line.strip()
    cut_list = tokenizer.tokenize(temp)
    #cut = jieba.cut(temp)
    #cut_list = [x for x in cut]
    if len(cut_list) == 0:
        continue
    if len(cut_list) > 2048:
        cut_list = cut_list[:2047]
    # else:
    #     print(len(cut_list))
    #     #print(cut_list)
    word_index = []
    for i,word in enumerate(cut_list):
        index = 0
        if word in vocab:
            index = word2index[word]
        word_index.append(index)
    train_data.append(torch.tensor(word_index))
fr.close()
train_data.append(torch.zeros(seq_len))
train_data = pad_sequence([train_data[i] for i in range(len(train_data))],batch_first=True)
#print(train_data)
print(len(train_data[0]))

accelerator = Accelerator()
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = accelerator.device

#model = Transformer(seq_len=100,vocab_size=vocab_size)
model = Transformer(seq_len=seq_len,vocab_size=vocab_size,N=40,d_ff=3072,h=24,d_model=480)
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

dataset = load_dataset('text',data_files='../data/test0.txt')
data = torch.utils.data.DataLoader(dataset,shuffle=True)

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model,optimizer,data = accelerator.prepare(model,optimizer,data)

model.train()
train_data = train_data.to(device)
for Input in train_data:
    with torch.no_grad():
        Input = torch.t(Input)
    out=model(Input)
    crossentropyloss=nn.CrossEntropyLoss()
    optimizer.zero_grad()
    loss=crossentropyloss(out,Input)
    print('loss:\n',loss.item())
    #loss.backward()
    accelerator.backward(loss)
    optimizer.step()