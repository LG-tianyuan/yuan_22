import torch
from torch.utils.data.distributed import DistributedSampler

device=torch.device('cuda')
hidden_dim=100
train_data = torch.randn(100, hidden_dim, device=device, dtype=torch.half)
train_label = torch.empty(100,dtype=torch.long,device=device).random_(hidden_dim)
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
print("train_data:")
print(train_data)
print("train_label:")
print(train_label)
print("train_dataset:")
print(train_dataset)