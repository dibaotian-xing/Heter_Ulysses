#!/bin/bash

# Runs the Qwen3 0.6B model

export CUDA_VISIBLE_DEVICES=3,4
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=/data/ssd/lijie/Qwen3-0.6B-mcore #<Specify path>
# TENSORBOARD_LOGS_PATH=$2 #<Specify path>
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=/home/lijie/data/gpt2_small_document/my-gpt2-small_text_document #<Specify path and file prefix>_text_document
TOKENIZER_MODEL_PATH=/data/ssd/lijie/Qwen3-0.6B-hf

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 28 
    --hidden-size 1024
    --ffn-hidden-size 3072 
    --num-attention-heads 16 
    --seq-length 4096 
    --max-position-embeddings 4096
    --num-query-groups 8
    --group-query-attention
    --normalization RMSNorm
    --position-embedding-type rope
    --kv-channels 128 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models
    --micro-batch-size 1 
    --global-batch-size 256 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --disable-bias-linear
    --attention-dropout 0
    --hidden-dropout 0
    --swiglu
    --qk-layernorm
    --rotary-base 1000000
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-iters 0
    --ckpt-format torch_dist
    --ddp-average-in-collective
    --overlap-grad-reduce
    --use-flash-attn
    --no-gradient-accumulation-fusion
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 1
    --context-parallel-size 2
    --cp-comm-type a2a  
)
# --heter-ulysses-config-path examples/qwen/config/qwen3_0.6b_seq4096.json

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path $DATA_PATH 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --train-iters 100
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} 2>&1 | tee logs/homo-ulysses-2a6000-`date +%F-%H%M`.log
