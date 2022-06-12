import os
import json
import argparse
import torch
import torch.nn as nn
import deepspeed
from deepspeed.runtime.lr_schedules import LRRangeTest
from deepspeed.runtime.lr_schedules import  WarmupLR
from model.Transformer_model import Transformer
from megatron import mpu
#import matplotlib.pyplot as plt
import numpy as np

def load_data(path):
    train_data = []
    fr = open(path,encoding="utf-8")
    i = 0
    for line in fr:
        data = []
        temps = line.strip().split(',')
        for temp in temps:
            data.append(int(temp))
        if i == 0:
            train_data = torch.tensor(data).unsqueeze(0)
        else:
            data = torch.tensor(data).unsqueeze(0)
            train_data = torch.cat((train_data,data),0)
        i += 1
    return train_data

def get_data_loader(path,batch_size):
    train_data = load_data(path)
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size)
    return train_dataset,train_loader

config_dict = {
    "train_batch_size": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 4e-4,
      	        "betas": [
        	0.8,
        	0.999
      	        ],
      	        "eps": 1e-8,
            "weight_decay": 0.001
        }
    },
    "zero_optimization": {
        "stage":3,
        "reduce_bucket_size": 20,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/nvme_data"
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/nvme_data"
        }
    },
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1
}
'''
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
'''

def create_config_from_dict(tmpdir, config_dict):
    config_path = os.path.join(tmpdir, 'temp_config.json')
    with open(config_path, 'w') as fd:
        json.dump(config_dict, fd)
    return config_path

def create_moe_param_groups(model):
    from deepspeed.moe.utils import is_moe_param

    params_with_weight_decay = {'params': [], 'name': 'weight_decay_params'}
    moe_params_with_weight_decay = {
        'params': [],
        'moe': True,
        'name': 'weight_decay_moe_params'
    }

    for module_ in model.modules():
        moe_params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and is_moe_param(p)
        ])
        params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and not is_moe_param(p)
        ])

    return params_with_weight_decay, moe_params_with_weight_decay

def get_args(tmpdir, config_dict):
    parser = argparse.ArgumentParser()
    #cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    #train
    parser.add_argument('-b',
                        '--batch_size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument("--local_rank",
    					type=int, 
    					default=0,
    					help='local rank passed from distributed launcher')
    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")
    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')
    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        default=1,
                        type=int,
                        help='(moe) number of total experts')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument('--zero', 
    					type=int, 
    					default=0)
    parser.add_argument('--moe-param-groups',
                        default=False,
                        action='store_true',
                        help='(moe) create separate moe param groups, required when using ZeRO w. MoE')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()  #args=''

    config_dict["train_batch_size"] = args.batch_size
    config_dict["zero_optimization"]["stage"] = args.zero
    print('config_dict["zero_optimization"]', config_dict["zero_optimization"])
    config_path = create_config_from_dict(tmpdir, config_dict)

    args.deepspeed_config = config_path
    return args

def main():
    args = get_args('/tmp/', config_dict)
    deepspeed.init_distributed()
    mpu.initialize_model_parallel() 
    if args.moe:
        deepspeed.utils.groups.initialize(ep_size=args.ep_world_size, mpu=mpu)
    model = Transformer(seq_len=2048,vocab_size=53228,N=40,d_ff=3072,h=24,d_model=480)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if args.moe_param_groups:
        parameters = create_moe_param_groups(model)

    # Learning Rate Schedulers
    lr_optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    lr_scheduler = WarmupLR(lr_optimizer)
    
    trainset,data_loader = get_data_loader(path='data/test0_index.txt',batch_size=1)

    model, _, _, lr_scheduler= deepspeed.initialize(args=args,
                                     model=model,
                                     model_parameters=parameters,
                                     training_data=trainset,
                                     lr_scheduler=lr_scheduler,
                                     mpu=mpu,
                                     dist_init_required=True)

    #fp16 = model.fp16_enabled()

    
    crossentropyloss=nn.CrossEntropyLoss()
    for i,Input in enumerate(data_loader):
        Input = Input[0].to(model.local_rank)
        Input,out=model(Input)
        print('Input:',Input.shape)
        print('out:',out.shape)
        loss=crossentropyloss(out,Input)
        print('loss:\n',loss.item())
        model.backward(loss)
        model.step()
        lr_scheduler.step()

if __name__ == "__main__":
    main()












