#!/bin/bash
set -e
set -x

# model information, should not set layer num
export model_name="qwen3_0.6b"
export hidden_size=1024
export ffn_hidden_size=3072
export num_attention_heads=16
export num_query_groups=8
export kv_channels=128

# bsz and seqlen
export profile_bsz=4
export seq_length=4096

# heterogeneous settings, each gpu type should choose one rank to be profiled
export CLUSTER_TYPE="a6000x2_id0-1"
export gpu_type_id=(0 1) # the indexes of gpu types of this node
export gpu_type_rank=(0 1) # gpu_type_rank[i] should be a rank belongs to gpu_type_id[i]

# for attn time profile
export num_query_groups_diff=4
# for tf layer time profile
export seq_length_diff=2048

export SCRIPT_PATH='examples/profile/models/qwen3/single_run.sh'
source examples/profile/profile_time.sh

# profile_diff_num_gqa_groups_time
export seq_length=$((seq_length+seq_length_diff))
# profile_diff_num_gqa_groups_time

export seq_length=$((seq_length-seq_length_diff))
python ipalg/post_process_time_profile.py \
    --cluster_type $CLUSTER_TYPE \
    --model_name $model_name \
    --seq_length $seq_length \
    --seq_length_diff $seq_length_diff \
    --num_query_groups $num_query_groups \
    --num_query_groups_diff $num_query_groups_diff \
    --batch_size $profile_bsz \
    --gpu_type_id_list "${gpu_type_id[@]}"