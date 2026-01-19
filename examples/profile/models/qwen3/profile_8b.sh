#!/bin/bash
set -e
set -x

# model information, should not set layer num
export model_name="qwen3_8b"
export hidden_size=4096
export ffn_hidden_size=12288
export num_attention_heads=32
export num_query_groups=8
export kv_channels=128

# bsz and seqlen
export profile_bsz=4
export seq_length=4096

# heterogeneous settings, each gpu type should choose one rank to be profiled
export CLUSTER_TYPE="a800_x4+a800_120w_x4"
export gpu_type_id=(0 1) # the indexes of gpu types of this node
export gpu_type_rank=(0 4) # gpu_type_rank[i] should be a rank belongs to gpu_type_id[i]
export profile_port=(37001 37002) #profile_port[i] should be the master port used by gpu_type_id[i]

# for attn time profile
export num_query_groups_diff=4
# for tf layer time profile
export seq_length_diff=2048

export SCRIPT_PATH='examples/profile/models/qwen3/single_run.sh'
source examples/profile/profile_diff_settings.sh

profile_diff_num_gqa_groups
export seq_length=$((seq_length+seq_length_diff))
profile_diff_num_gqa_groups

export seq_length=$((seq_length-seq_length_diff))
python ipalg/profile_post_process.py \
    --cluster_type $CLUSTER_TYPE \
    --model_name $model_name \
    --seq_length $seq_length \
    --seq_length_diff $seq_length_diff \
    --num_query_groups $num_query_groups \
    --num_query_groups_diff $num_query_groups_diff \
    --batch_size $profile_bsz \
    --gpu_type_id_list "${gpu_type_id[@]}"