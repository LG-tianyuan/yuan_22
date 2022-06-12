'''
import torch
#import torch.nn.functional as F
#from datasets import load_dataset


#dataset = load_dataset('text',data_files='data/test.txt')
#data = torch.utils.data.DataLoader(dataset, shuffle=True)

a = torch.tensor([1,2,3]).unsqueeze(0)
b = torch.tensor([4,5,6]).unsqueeze(0)
a = torch.cat((a,b),0)
a = torch.cat((a,b),0)
print(a)
c = torch.rand(2,3,4)
print(c)

from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from model.Transformer import Transformer
model = Transformer(seq_len=2048,vocab_size=53228,N=40,d_ff=3072,h=24,d_model=480)
estimate_zero2_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)
'''
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from model.Transformer import Transformer
model = Transformer(seq_len=2048,vocab_size=53228,N=40,d_ff=3072,h=24,d_model=480)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=4, num_nodes=1)