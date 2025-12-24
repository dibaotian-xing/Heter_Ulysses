#!/bin/bash
set -e
set -x

#should accept gpu_type_id, gpu_rank, hidden_size, num_attn_heads, num_gqa_groups

export CUDA_VISIBLE_DEVICES=$2
export CUDA_DEVICE_MAX_CONNECTIONS=1

gpu_type_id=$1

DATA_PATH=/home/lijie/data/gpt2_small_document/my-gpt2-small_text_document
TOKENIZER_MODEL_PATH=/home/lijie/Qwen3-0.6B-hf

GPT_MODEL_ARGS=(
    --num-layers 1
    --hidden-size $3
    --ffn-hidden-size $ffn_hidden_size
    --num-attention-heads $4 
    --seq-length $seq_length
    --max-position-embeddings $seq_length
    --num-query-groups $5
    --group-query-attention
    --normalization RMSNorm
    --position-embedding-type rope
    --kv-channels $kv_channels 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models
    --micro-batch-size $profile_bsz 
    --global-batch-size $profile_bsz 
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

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL_PATH}
    --data-path $DATA_PATH 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval -1 
    --eval-interval 1000 
    --eval-iters 10
    --train-iters 100
)

PROFILE_HETER_ULYSSES_ARGS=(
    --profile-heter-ulysses
    --cluster-type $CLUSTER_TYPE
    --gpu-type-id $gpu_type_id
    --heter-ulysses-model-name $model_name
)

torchrun --nproc_per_node 1 pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${PROFILE_HETER_ULYSSES_ARGS[@]}
