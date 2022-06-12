Megatron is a large, powerful transformer. This repo is for ongoing research on training large, powerful transformer language models at scale. Currently, we support model-parallel, multinode training of [GPT2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [BERT](https://arxiv.org/pdf/1810.04805.pdf) in mixed precision. 

Our codebase is capable of efficiently training a 72-layer, 8.3 Billion Parameter GPT2 Language model with 8-way model and 64-way data parallelism across 512 GPUs. We find that bigger language models are able to surpass current GPT2-1.5B wikitext perplexities in as little as 5 epochs of training.

For BERT training our repository trains BERT Large on 64 V100 GPUs in 3 days. We achieved a final language modeling perplexity of 3.15 and SQuAD F1-score of 90.7.
<!--
do we want to make any claims about GPT2 speed, convergence, or model release
-->

# Setup
We officially support only python3.6.

To use this repo please install the latest supported versions of PyTorch with GPU support. 

Additionally, part of this codebase leverages tensorflow-cpu to (optionally) perform dataloading of TFRecords for BERT training. We recommend either utilizing the provided Dockerfile in [`./docker/`](./docker) or creating a virtual environment (to avoid breaking existing tf installations) and install our `requirements.txt`. 

```
python -m pip install virtualenv
virtualenv bert_env
source bert_env/bin/activate
pip install -r requirements.txt
```


# Usage
We've provided 5 scripts that pretrain BERT and 3 scripts that pretrain GPT2. Save and load model checkpoints with `--save` and `--load`. Additionally we provide GPT2 scripts for interactive text generation and zero shot evaluation of GPT2 on wikitext and LAMBADA.

## BERT Pretraining
`bash scripts/pretrain_bert.sh`

This script runs single gpu BERT pretraining and is mainly for debugging purposes. The optimization arguments are set with 64-way distributed training in mind.

To use this script place your `--train-data` in loose json format with one json per line. The text field of your json dictionaries should correspond to `--text-key`. 

```
python pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 4 \
       --seq-length 512 \
       --max-preds-per-seq 80 \
       --max-position-embeddings 512 \
       --train-iters 1000000 \
       --save checkpoints/bert_345m \
       --load checkpoints/bert_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-large-uncased \
       --presplit-sentences \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --fp16 \
       --fp32-embedding
```

## GPT2 Pretraining
`bash scripts/pretrain_gpt2.sh`

This script runs single gpu gpt2 pretraining and is mainly for debugging purposes. The optimization arguments are set with 64-way distributed training in mind. 

It follows largely the same format as the previous script with a few notable differences: the `--tokenizer-type` has been switched to a `GPT2BPETokenizer`, the `--lr-decay-style` has been switched to cosine decay, and activation checkpointing has been turned on with `--checkpoint-activations` and `--checkpoint-num-layers` set to checkpoint every `1` layers.

Additionally GPT2 uses a different parameter initialization from BERT designed for training deep residual networks. To train BERT with this initialization use `--deep-init`.

```
python pretrain_gpt2.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --batch-size 8 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 320000 \
       --save checkpoints/gpt2_345m \
       --load checkpoints/gpt2_345m \
       --resume-dataloader \
       --train-data wikipedia \
       --lazy-loader \
       --tokenizer-type GPT2BPETokenizer \
       --cache-dir cache \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --checkpoint-activations \
       --fp16
```

## GPT2 Text Generation
`bash scripts/generate_text.sh`

Starts an interactive terminal session that generates text either conditionally or unconditionally depending on what the user enters into the prompt. Specify the model in the script by setting the `CHECKPOINT_PATH` variable and the appropriate model configuration. 

The script is capable of greedy sampling, top-k, or top-p sampling as specified by the appropriate variables within the script.

## GPT2 Evaluation
We support 3 modes of GPT2 evaluation with [`./scripts/run_gpt2_eval.py`](./scripts/run_gpt2_eval.py): wikitext ppl evaluation, lambada cloze accuracy, large corpora ppl evaluation.

### Wikitext PPL evaluation
For even comparison with prior works we evaluate wikitext perplexity on the word-level wikitext test dataset, which can be downloaded [here](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip), and appropriately compute perplexity given the change in tokens when using our subword tokenizer.

We use the following command to run wikitext evaluation:

```
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --model-path <gpt2_345_path> \
  --data-path <wikitext_tokens_test_path> \
  --batch-size 16 \
  --cache-dir cache
```

### Lambada Cloze Accuracy
To compute Lambada cloze accuracy (the accuracy of predicting the last token given the preceding tokens) we utilize a detokenized, processed version of the Lambada dataset we sourced from [here](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl).

We use the following command to run lambada evaluation:

```
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --model-path <gpt2_345_path> \
  --data-path <lambada_test_path> \
  --batch-size 16 \
  --cloze-eval \
  --cache-dir cache
```

### Large Corpora PPL evaluation
This functionality allows one to evaluate the gpt2 model on a loose json file. With the following command we evaluate the gpt2 model for 5000 iterations at a batch size of 16 on a webtext test data split. We recommend that the user presplit their dataset before training a model according to the procedure outlined [below](#partitioning-datasets-into-train-val-test).

```
python scripts/run_gpt2_eval.py \
  --model-parallel-size 1 \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --model-path <gpt2_345_path> \
  --data-path <webtext_test_path> \
  --batch-size 16 \
  --eval-iters 5000 \
  --webtext-eval \
  --cache-dir cache
```

## Distributed BERT or GPT2 Pretraining
`bash scripts/pretrain_bert_distributed.sh` or `bash scripts/pretrain_gpt2_distributed.sh`

To use these scripts, follow the same data preparation procedure as in earlier sections. This script uses the pytorch distributed launcher to launch distributed training. As such, multinode training can be achieved by properly setting environment variables for the `env://` init method. See the official pytorch [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility) for further description of these [environment variables](https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization). By default multinode training uses the nccl distributed backend.

## Model Parallel BERT or GPT2 Pretraining
`bash scripts/pretrain_bert_model_parallel.sh` or `bash scripts/pretrain_gpt2_model_parallel.sh`

These scripts build upon the distributed training scripts and are identical in setup. They differ in use of the `--model-parallel-size` flag. For model parallelism of 2 and a world size of 8, the scripts will launch training with 4-way distributed data parallelism and 2-way model parallelism.

We note that we have experimented with multiple distributed data parallel implementations: a simple one of our own which performs gradient all-reduce at the end of back propagation step, and torch's distributed data parallel wrapper which overlaps gradient reduction with back propagation computation. To switch between these two options toggle the `USE_TORCH_DDP` flag (the default is set to `False` and uses our DDP implementation) at the top of `pretrain_bert.py` and `pretrain_gpt2.py`. We find that torch distributed data parallelism is more efficient at larger model parallel sizes. For example, for the 8.3 billion parameters model running on 512 GPUs, the scaling increases from 60% to 74% when torch's distributed data parallel is used. However, the overlapping method requires more memory and for some configurations (e.g., 2.5 billion parameters using 2-way model parallel and 1.2 billion parameters with no model parallel) can make the overall training slower as a result. We empirically found that using a smaller model in those cases improves the training time.

## Distributed BERT Pretraining with TFRecords
`bash scripts/pretrain_bert_tfrecords_distributed.sh`

This script takes advantage of TensorFlow BERT's [`create_pretraining.py`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/create_pretraining_data.py) script to pre-cache the dataset in the TFRecord format. To convert the data to pytorch tensors we use a `TFRecordDataset` and tensorflow eager mode to turn the TFRecords into numpy matrices before loading them into pytorch gpu tensors. This greatly reduces the overhead of dataprocessing and speeds up training. Pass a whitespace-separated list of TFRecord paths to `--train-data` and enable the `--use-tfrecords` flag. Multinode training can be achieved as described in the [previous section](#distributed-bert-pretraining).

## Train Custom Sentence Piece Tokenizer and Pretrain BERT
`bash scripts/pretrain_bert_sentencepiece.sh`

This script runs BERT pretraining with a `sentencepiece` tokenizer. If no sentencepiece tokenizer exists at `--tokenizer-path` one will be trained automatically. The sentencepiece tokenizer can be used with the previous scripts (NOTE: sentencepiece training can only happen during single gpu pretraining). `<--tokenizer-path>.vocab` can be used with [`create_pretraining_data.py`](https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow/LanguageModeling/BERT/create_pretraining_data.py) to make a TFRecord dataset with the given tokenization.


# Data sets
We do not host any datasets for GPT2 or BERT training, however, we detail their collection so that our results may be reproduced.

## Collecting Wikipedia Training Data
We recommend following the wikipedia data extraction process specified by google research: "the recommended pre-processing is to download [the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [WikiExtractor.py](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text." 

We recommend using the `--json` argument when using WikiExtractor, which will dump the wikipedia data into loose json format (one json per line), making it more manageable and readily consumable by our codebase. We recommend further preprocessing this json dataset by preprocessing the dataset with nltk punctuation standardization, and presplitting each document into newline separated sentences. This can be done with the provided script `./scripts/presplit_sentences_json.py` and will allow for faster data processing during training time. Pretraining with presplit data should be run with the `--presplit-sentences` flag as shown above. (Note that if you'd like to use wikipedia data for GPT2 training you should still clean it with nltk/spacy/ftfy, but do not split it into newline seperated sentences)

Once the json dataset is ready make sure to set the path in line 27 of `data_utils/corpora.py`.

If your system is memory limited we also recommend running pretraining with the `--lazy-loader` argument as we've done. After preprocessing the dataset once, this will allow the dataset to be lazily loaded from disk, as opposed to storing it in memory. Make sure to run the code once on a 

## Collecting GPT2 Webtext Data
We utilize the publicly available [OpenWebText](https://github.com/eukaryote31/openwebtext) library from [jcpeterson](https://github.com/jcpeterson/openwebtext) and [eukaryote31's](https://github.com/eukaryote31/openwebtext) work to download urls. We then filtered, cleaned, and deduplicated all downloaded content according to the procedure described in our [openwebtext](./openwebtext) directory. For reddit URLS corresponding to content upto october 2018 we arrived at approximately 37GB of content.

We recommend creating an alias for this dataset as described below.

## Aliasing datasets with corpora.py
As mentioned in the previous Wikipedia data section we recommend aliasing datasets with human readable names (eg. `--train-data wikipedia`). This helps avoid forgetting arguments when submitting jobs, and allows one to combine datasets that would otherwise require different commandline options/data structures.

Examples of how to create these dataset objects can be found in [`./data_utils/corpora.py`](./data_utils/corpora.py). We recommend that the objects inherit from or adhere to the interface laid out by `torch.utils.data.Dataset` objects.

Any created datasets should be then added to the `NAMED_CORPORA` dictionary object in [`./data_utils/corpora.py`](./data_utils/corpora.py). At runtime one can specify one or more corpora from the commandline with `--train-data corpus1 corpus2 corpus3`, `--valid-data corpus1 corpus2 corpus3`, or `--test-data ...`.

## Partitioning datasets into Train/Val/Test
We support multiple ways to partition corpora into train/val/test splits. By specifying a `--split 95,5` commandline argument, the corpora specified by `--train-data` will have it's documents split proportionally into a 95%, 5% train/val split. The split is performed lazily on the fly and is efficient and deterministic from run to run given the same `--seed`. Note that if `--valid-data` or `--test-data` is specified then the train data will still be split accordingly, but `--valid-data`/`--test-data` will still be used as the validation/test source.

We do realize that this method, while effective, introduces noise into the development process, since different seeds will change the dataset and outcome. To have fixed training/validation/test sets across all your runs please utilize our script [`./scripts/split_json.py`](./scripts/split_json.py)

------

# Guide

**数据预处理：**

`pre-process.py`

**tokenizer:**

`EncDecTokenizer`

**训练文件：**

`pretrain_gpt2.py`

`run_gpt2_test.json`

`run_gpt2_test.sh`

**originmodel文件夹：**

花了一大半时间搭建和测试的模型

## Proposed Method

> teamwork with a teammate

In our experiments, we trained our model with mainly two distributed training framework, Megatron-LM and DeepSpeed.

- Data preprocessing

We preprocess the raw data provided by the preliminary round of ASC22 with the script(Yuan/model/pre-process.py) we wrote and the vocab provided by Yuan1.0. We first process the vocab.txt provided by Yuan 1.0 into a tokenizer, divide the text in the raw data into sequences with <n>, and then segment them, and then convert them into indexes and save them to disk in the format of numpy.array.

- Model

Based on GPT-2 code and modified it to meet our need.

- Training Strategies and Optimization

According to our training environment, 1 node with 4 16G Tesla V100 before and 1 node 8 16G Tesla P100 later, we have tried fp16 or mixed precision to accelerates our training while reducing memory usage. We have made comparations bewteen ZeRO of stage 2 and 3 to find a more suitable strategies to boost memory efficiency. We have tested the performance of different optimizers and learning rate scehdulers through a certain amount of training experiments. Finally, we adopted the distributed training strategies of model parallelism and data parallelism with size of 8. Since we have only one node, we didn't adopt pipeline parallelism which is not compatible with ZeRO2 or ZeRO3 and may increase the communication time as well as produce unexpected bubbles. For the memory optimization, we trained the model with fp16 ZeRO2 optimizer and activation checkpointing which we found better performance during the training before. Since we couldn't increase the training batch_size due to the constraint memory, we tried gradient accumulation to speed up convergence with a learning rate scheduler of LRRangeTest.

# Some notes

> Date 2022-03-07

**：记`2022ASC`预赛的赛题三——Yuan Large Language Model Challenge，带给我的一些东西。**

该仓库包含的所有文件都是为解决`2022ASC`预赛赛题三而不断试错留下来的。

这次比赛真的是从零开始学起，比赛开始前只学会了简单地使用KNN、朴素贝叶斯来解决机器学习分类问题，甚至连python都不是很熟悉。加入超算团队后，才从指导老师那得知有深度学习框架这么一样东西，当时立马就网上搜索教程，下载Anaconda、TensorFlow，上mooc找了一门TensorFlow的课，奈何太忙没腾出多少时间来学，几周时间下来才看了几节课。所以比赛开始的时候，拿到赛题一看，NLP？Large Language Model？transformer？pytorch？megatron？确实是一脸茫然。

记录一下解赛题的主要过程吧：

- 2022.1.15 
  - 寒假放假回家

- 2022.1.16 - 2022.1.20 
  - 了解一些预训练模型，看了一下Yuan的源代码，一点都看不懂

- 2022.1.20 
  - 开始学习transformer的模型结构

- 2022.1.20 - 2022.1.27 
  - 一边学习一边研究赛题要求构建的模型
- 2022.1.28 - 2022.2.4

  - 开始尝试搭建模型，没错，仅仅是transformer encoder这一部分。

  - 产生了一些疑问：通过公式计算的参数量和实际输出的参数量不一致！损失函数是什么意思？怎么实现？

- 2022.2.5 - 2022.2.12
  - 学习分布式训练策略及其思想，同时研究源代码中megatron-lm的函数调用关系、结构等。

  - 一直卡在损失函数这里

  - 开始测试搭建好的模型（太单纯不过的模型）

- 2022.2.12 - 2022.2.15

  - 经过和学长交流后确定损失函数为交叉熵（此后一直怀疑损失函数的实现是不是不符合要求）

  - 同时转向开始学习deepspeed

- 2022.2.16 - 2022.2.22

  - 开始借助deepspeed进行分布式训练，并使得模型成功跑在单个GPU上而没有出现显存不足，好像好给力

- 2022.2.22 - 2022.2.27

  - 成功实现4个GPU分割数据训练模型，但是损失函数值超大，收敛不了！

- 2022.2.27

  - 真是一个标志性的日子。意味着我们过去的工作要推倒重来！经过和博士生学长的交流得知，自己构建的模型风险很大，里面隐藏的错误太多，很难也需要很多时间去调试测试，建议找个成熟的模型跑起来再往赛题要求方向改。

- 2022.2.28

  - 经过一天一夜的调试修改，终于实现用deepspeedexample里面同时搭建在megatron-lm和deepspeed之上的gpt2模型跑上我们的数据。这是才发现，按照赛题要求的超参数设置的模型得到的参数量真的是整整的4.7B，看来模型“拿”对了，拿来主义真的好！我又看到了希望。但是，，，跑起来，4个 16G V100的GPU放不下呀！为啥之前这么设置超参数的时候能放得下？玩球。哦，不对，他们的模型和我们搭建的不一样，我们用的d_model他没用！不是d_model不影响公式计算得出的参数量吗？按照我们以前的模型改过来呀！刷刷把全连接层的size由“hidden_size-4hidden_size-hidden_size”改为“d_model-hidden_size-d_model"。运行看看，啊！参数量有不对啊！少了这么多！嘿嘿！应该没关系吧，原论文的d_model=512也设置的差不多呀，应该没问题。好！跑一晚上

- 2022.3.1

  - 完蛋！赛题这个全连接层的设置要求我们当时没看清楚，必须是“hidden_size-4hidden_size-hidden_size”，啊原来研究了一个多月的模型是不符合要求的啊！崩溃ing... 其实这也是意料之中吧，你这个模型搭建起来多简洁啊，连个啥并行训练设置接口都没有，而且参数量的问题一直都在。没办法，比赛还没结束，我们得坚持下去，我们还有三天，还有希望，说不定还能跑出结果来！马上联系老师，加GPU！这天晚上就跑了一个N=30层的encoder模型，跑了12个小时左右？训练了2w2个iterator，loss值下降到了八点几！还是能正常下降，挺稳的！

- 2022.3.2

  - 盼望已久的一个包含8块GPU的节点终于来了，裸机，争分夺秒地重新配了驱动、cuda、pytorch以及各种包，花了两个多小时，终于配好了。运行一下，模型并行，4.7B，8个16G，能放下！一看训练速度，OMG！P100的GPU是真的拉（虽然也不便宜），训练速度是在V100上跑的近1/3啊，10亿的token不到两天的时间能训练得完吗？而且适合于训练4.7B的模型的学习率等其他一些参数还没测试过呢！但是只剩下不到两天了！为了输出完整的tensorboard，不耽误其他赛题的提交，我们还是赌了一把，选择了减少训练量，选择了一个之前感觉合适的学习率开始进行训练。

- 2022.3.3

  - 承接之前的基础，很认真地写好了proposal

- 2022.3.4

  - 结果出来了，loss没收敛，训练量也没达到赛题要求，但还是交了上去，我都不好意思面对队友和老师了（/捂脸）。ddl到了，虽然结果对于赛题要求来说是无效的，但至少是我们努力争取的一个脚印吧。

  - 没办法，真正是深度学习小白的我们，之前走太多的弯路了，当我们将要走回正道的时候，时间和条件已经不允许了。

  - 打一场打不赢的比赛，跑一个终究是无效的结果，不管怎样，至少到了最后，我们还是努力尝试去争取了。而这个过程中所学习到的，或许是我参加这个比赛最大的意义吧。

  - 实践告诉我们，方向很重要，努力很重要，他人的指引和建议也很重要。

  - 记录下这些东西，算是给这场比赛经历一个句号吧。

  - 后续有时间还是继续学习一下NLP

