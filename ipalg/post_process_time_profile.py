# ========================================================================
# Copyright 2025 BigAI-Group@Nanjing University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
"""Post process Heter Ulysses time profiling results."""

import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ipalg.utils import read_json_config, write_json_config


def get_attn_time_per_gqa_group(config_dict, gqa_group_less, gqa_group_more, seqlen, gpu_type_id, bsz):
    sum_less, sum_more = 0, 0
    for iter in range(10, 20):
        key_less = f"attn_time_gpu_type{gpu_type_id}_seqlen{seqlen}_gqa_group{gqa_group_less}_iter{iter}_bsz{bsz}"
        key_more = f"attn_time_gpu_type{gpu_type_id}_seqlen{seqlen}_gqa_group{gqa_group_more}_iter{iter}_bsz{bsz}"
        sum_less += config_dict[key_less]
        sum_more += config_dict[key_more]
    
    mean_less_per_sample = sum_less / (10 * bsz)
    mean_more_per_sample = sum_more / (10 * bsz)

    return (mean_more_per_sample - mean_less_per_sample)/(gqa_group_more - gqa_group_less)


def get_other_time_per_token(
    config_dict, gqa_group, seqlen_less, seqlen_more, gpu_type_id, bsz,
    attn_time_per_gqa_group_seqlen_less, attn_time_per_gqa_group_seqlen_more,
):
    sum_less, sum_more = 0, 0
    for iter in range(10, 20):
        key_less = f"tf_layer_time_gpu_type{gpu_type_id}_seqlen{seqlen_less}_gqa_group{gqa_group}_iter{iter}_bsz{bsz}"
        key_more = f"tf_layer_time_gpu_type{gpu_type_id}_seqlen{seqlen_more}_gqa_group{gqa_group}_iter{iter}_bsz{bsz}"
        sum_less += config_dict[key_less]
        sum_more += config_dict[key_more]
    
    mean_less_per_sample = sum_less / (10 * bsz)
    mean_more_per_sample = sum_more / (10 * bsz)

    other_time_seqlen_less = mean_less_per_sample - attn_time_per_gqa_group_seqlen_less * gqa_group
    other_time_seqlen_more = mean_more_per_sample - attn_time_per_gqa_group_seqlen_more * gqa_group

    print(f"{other_time_seqlen_more=}, {other_time_seqlen_less=}")
    return (other_time_seqlen_more - other_time_seqlen_less)/(seqlen_more - seqlen_less)


def post_process(args):
    config_path = f'examples/profile/models/configs/profile_time_{args.model_name}_{args.cluster_type}.json'
    assert os.path.exists(config_path), f"config path {config_path} doesn't exist!"
    config_dict = read_json_config(config_path)
    attn_time_seqlen_less_list, attn_time_seqlen_more_list, other_time_list = [], [], []
    for gpu_type_id in args.gpu_type_id_list:
        # attn time for less seqlen
        attn_time_per_gqa_group_seqlen_less = get_attn_time_per_gqa_group(
            config_dict, args.num_query_groups, args.num_query_groups + args.num_query_groups_diff,
            args.seq_length - args.seq_length_diff, gpu_type_id, args.batch_size
        )
        attn_time_seqlen_less_list.append(attn_time_per_gqa_group_seqlen_less)
        
        # attn time for more seqlen
        attn_time_per_gqa_group_seqlen_more = get_attn_time_per_gqa_group(
            config_dict, args.num_query_groups, args.num_query_groups + args.num_query_groups_diff,
            args.seq_length, gpu_type_id, args.batch_size
        )
        attn_time_seqlen_more_list.append(attn_time_per_gqa_group_seqlen_more)

        # other time for less gqa group
        other_time_per_token = get_other_time_per_token(
            config_dict, args.num_query_groups, args.seq_length - args.seq_length_diff, args.seq_length,
            gpu_type_id, args.batch_size, attn_time_per_gqa_group_seqlen_less, attn_time_per_gqa_group_seqlen_more
        )
        other_time_list.append(other_time_per_token)

    config_dict[f"attn_time_per_gqa_group_seqlen{args.seq_length - args.seq_length_diff}"] = attn_time_seqlen_less_list
    config_dict[f"attn_time_per_gqa_group_seqlen{args.seq_length}"] = attn_time_seqlen_more_list
    config_dict[f"other_time_per_token"] = other_time_list
    write_json_config(config_dict, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_type", type=str, default=None, help="gpu types for the cluster. Default=None")
    parser.add_argument("--model_name", type=str, default=None, help="name for the profiled model. Default=None")
    parser.add_argument("--seq_length", type=int, default=None, help="seq_length in the profiling. Default=None")
    parser.add_argument("--seq_length_diff", type=int, default=None, 
                        help="seq_length_diff in the profiling. Default=None")
    parser.add_argument("--num_query_groups", type=int, default=None, 
                        help="num_query_groups in the profiling. Default=None")
    parser.add_argument("--num_query_groups_diff", type=int, default=None, 
                        help="num_query_groups_diff in the profiling. Default=None")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size in the profiling. Default=None")
    parser.add_argument('--gpu_type_id_list', nargs='+', type=int, default=[],
                       help='List of gpu type indexes in the cluster. Default=[]')
    args = parser.parse_args()
    post_process(args)