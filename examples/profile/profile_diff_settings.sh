#!/bin/bash
set -e
set -x

function profile_some_gpu_type(){
  bash $SCRIPT_PATH $1 $2 $hidden_size \
    $num_attention_heads $num_query_groups $3 $4
  num_gqa_groups_more=$((num_query_groups+num_query_groups_diff))
  ngroups=$((num_attention_heads/num_query_groups))
  num_attn_heads_more=$((ngroups*num_gqa_groups_more))
  hidden_size_more=$((num_gqa_groups_more*kv_channels))
  bash $SCRIPT_PATH $1 $2 $hidden_size_more \
    $num_attn_heads_more $num_gqa_groups_more $3 $4
}

function profile_diff_num_gqa_groups(){
    for (( i=0 ; i<${#gpu_type_id[@]} ; i=i+1 ))
    do
     (
      profile_some_gpu_type ${gpu_type_id[i]} ${gpu_type_rank[i]} ${profile_port[i]} time
     ) &
    done

    wait

    profile_some_gpu_type ${gpu_type_id[0]} ${gpu_type_rank[0]} ${profile_port[0]} memory
}
