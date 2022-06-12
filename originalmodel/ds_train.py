import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data.distributed import DistributedSampler
from model.Transformer import Transformer

def load_data(path):
    train_data = []
    fr = open(path,encoding="utf-8")
    for line in fr:
        data = []
        temps = line.strip().split(',')
        for temp in temps:
            data.append(temp)
        train_data.append(data)
    train_data = torch.tensor(train_data)
    return train_data

def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path


def get_data_loader(path,model,device):
    batch_size = model.train_micro_batch_size_per_gpu()
    train_data = load_data(path)
    train_dataset = torch.utils.data.TensorDataset(train_data)
    sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--zero', type=int, default=0)
    args = parser.parse_args()  #args=''

    config_dict["zero_optimization"]["stage"] = args.zero
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args


def print0(msg):
    if torch.distributed.get_rank() == 0:
        print(msg, flush=True)


rank = int(os.environ['RANK'])
print('seed:', 2222 + rank)
torch.random.manual_seed(2222 + rank)

config_dict = {
    "train_batch_size": 8,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
        }
    },
    "fp16": {
        "enabled": True,
        "initial_scale_power": 15
    },
    "zero_optimization": {
        "stage": 0,
        "reduce_bucket_size": 20
    }
}
#        "initial_scale_power": 15
args = get_args('/tmp/', config_dict)
hidden_dim = 4

model = Transformer(seq_len=seq_len,vocab_size=vocab_size,N=40,d_ff=3072,h=24,d_model=480)

model, _, _,_ = deepspeed.initialize(args=args,
                                     model=model,
                                     model_parameters=model.parameters(),
                                     dist_init_required=True)


def print_params(tag, model):
    if torch.distributed.get_rank() == 0:
        for n, p in model.named_parameters():
            print0("{} {}:{}".format(tag, n, p))


data_loader = get_data_loader(path='data/test_index.txt',model=model,
                              device=model.device)
crossentropyloss=nn.CrossEntropyLoss()
#print_params('pre-train', model)
for Input in enumerate(data_loader):
    out=model(Input)
    optimizer.zero_grad()
    loss=crossentropyloss(out,Input)
    print('loss:\n',loss.item())
    model.backward(loss)
    optimizer.step()
