#!/bin/bash
set -e
set -x

# profile attn time for different gpu types
function profile_diff_num_gqa_groups(){
    for (( i=0 ; i<${#gpu_type_id[@]} ; i=i+1 ))
    do
      bash $SCRIPT_PATH ${gpu_type_id[i]} ${gpu_type_rank[i]} $hidden_size \
        $num_attention_heads $num_query_groups
      num_gqa_groups_more=$((num_query_groups+num_query_groups_diff))
      ngroups=$((num_attention_heads/num_query_groups))
      num_attn_heads_more=$((ngroups*num_gqa_groups_more))
      hidden_size_more=$((num_gqa_groups_more*kv_channels))
      bash $SCRIPT_PATH ${gpu_type_id[i]} ${gpu_type_rank[i]} $hidden_size_more \
        $num_attn_heads_more $num_gqa_groups_more
    done
}

