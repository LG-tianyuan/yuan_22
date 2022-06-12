raw_data_path='./rawdata/001.txt'
#raw_data_path='./data/test2.txt'
id_path='./data16/Data16-'
#id_path='./dataset/test2.pth'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from megatron.tokenizer import tokenization_enc_dec

# b = torch.load(id_path)
# print(b)

seq_length=2048
vocab = []
fr = open("./demo/vocab.txt",encoding="utf-8")
for line in fr:
	vocab.append(line.strip())
fr.close()
vocab_size = len(vocab)

word2index = { w: i for i,w in enumerate(vocab) }
index2woed = { i: w for i,w in enumerate(vocab) }

#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 
tokenizer = tokenization_enc_dec.EncDecTokenizer(vocab_file="./demo/vocab.txt")

size=50000

train_data = []
fr = open(raw_data_path,encoding="utf-8")
j=0
i=0
for line in fr:
	j+=1
	if(j%100==0):
		print(i,j,end=' - ')
	if(j==size):
		train_data = np.array(train_data,dtype=np.uint16)
		np.save(id_path+str(i)+'.npy',train_data)
		print('\ns i=',i)
		i+=1
		train_data = []
		j=0
	temps = line.strip().split('<n>')
	for temp in temps:
		#print(temp)
		cut_list = tokenizer.tokenize(temp)
		if len(cut_list) == 0:
		    continue
		word_index = []
		for _,word in enumerate(cut_list):
		    index = 0
		    if word in vocab:
		        index = word2index[word]
		    word_index.append(index)
		if(len(word_index)>seq_length):
			word_index = word_index[:seq_length]
			#print(temp)
		train_data.append(word_index)
# for line in fr:
# 	j+=1
# 	if(j%100==0):
# 		print(i,j,end=' - ')
# 	if(j==size):
# 		train_data = np.array(train_data,dtype=object)
# 		np.save(id_path+str(i)+'.npy',train_data)
# 		print('\ns i=',i)
# 		i+=1
# 		train_data = []
# 		j=0
# 	temp = line.strip().replace('<n>','')
# 	#print(temp)
# 	cut_list = tokenizer.tokenize(temp)
# 	if len(cut_list) == 0:
# 	    continue
# 	word_index = []
# 	for _,word in enumerate(cut_list):
# 	    index = 0
# 	    if word in vocab:
# 	        index = word2index[word]
# 	    word_index.append(index)
# 	if(len(word_index)>seq_length):
# 		word_index = word_index[:seq_length]
# 		#print(temp)
# 	train_data.append(word_index)


fr.close()
train_data = np.array(train_data,dtype=np.uint16)
np.save(id_path+str(i)+'.npy',train_data)

print('----------------------ok---------------------\nfile_num:',i)