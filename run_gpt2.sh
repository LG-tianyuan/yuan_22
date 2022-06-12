#! /bin/bash

# Change for multinode config
MP_SIZE=8

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

config_json="run_gpt2.json"
gpt_options=" \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 40 \
       --hidden-size 3072 \
       --num-attention-heads 24 \
       --batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 10 \
       --resume-dataloader \
       --use-npy-data-loader \
       --train-data-path 'data/' \
       --input-data-sizes-file 'sizes.txt' \
       --lazy-loader \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00005 \
       --no-load-optim \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --checkpoint-num-layers 40 \
       --deepspeed-activation-checkpointing \
       --fp16 \
       --log-interval 1 \
"
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"


run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} pretrain_gpt2.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
