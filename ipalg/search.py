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
"""search json config for Heter Ulysses"""

import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import numpy as np
from ipalg.ipalg import IPAlg
from ipalg.utils import read_json_config, write_json_config


def search(args):
    num_gpu_type = len(args.gpu_type_num_list)
    gpu_nums = np.array(args.gpu_type_num_list)
    sl_tot = args.seq_length
    gn_tot = args.num_query_groups
    tot_gpu_num = int(np.sum(gpu_nums))
    a2a_comm_e_path = f"examples/profile/environment/configs/" \
        f"all2all_bandwidth_{tot_gpu_num}_gpus{args.cluster_type}.json"
    a2a_comm_e_dict = read_json_config(a2a_comm_e_path)
    a2a_comm_e = a2a_comm_e_dict['a2a_coe'] / 1024 / 1024 # (ms/MB) -> (ms/B)
    profile_time_path = f"examples/profile/models/configs/" \
        f"profile_time_{args.model_name}_{args.cluster_type}.json"
    profile_time_dict = read_json_config(profile_time_path)
    attn_time_per_gpa_group_key = f"attn_time_per_gqa_group_seqlen{args.seq_length}"
    time_g = np.array(profile_time_dict[attn_time_per_gpa_group_key])
    other_time_per_token_key = "other_time_per_token"
    time_l = np.array(profile_time_dict[other_time_per_token_key])
    hd = args.head_dim
    bsz = args.batch_size
    nt = args.num_hidden_layers
    nq = args.num_attention_heads // args.num_query_groups
    profile_mem_path = f"examples/profile/models/configs/profile_memory_{args.model_name}.json"
    profile_mem_dict = read_json_config(profile_mem_path)
    attn_act_mem_per_gqa_group_key = f"attn_act_mem_per_gqa_group_seqlen{args.seq_length}"
    mem_g = profile_mem_dict[attn_act_mem_per_gqa_group_key]
    mem_l = profile_mem_dict["tf_layer_other_act_mem_per_token"]
    mem_e = profile_mem_dict["other_layer_act_mem_per_token"]
    model_states_size = args.model_parameter_size * 16
    M = np.array(args.gpu_type_mem_capacity_list) - model_states_size
    precision = args.precision

    ipalg = IPAlg(
                num_gpu_type,
                gpu_nums,
                sl_tot,
                gn_tot,
                a2a_comm_e,
                time_g,
                time_l,
                mem_g,
                mem_l,
                mem_e,
                M,
                bsz,
                hd,
                nt,
                nq,
                precision
            )

    opt_tot_time, opt_seqlens, opt_headnums = ipalg.fit()
    print('optimized total time:', opt_tot_time)
    print('optimized sequence lengths:', opt_seqlens)
    print('optimized head nums:', opt_headnums)

    result = {}
    result['num_gpu_type'] = num_gpu_type
    result['gpu_nums'] = gpu_nums.tolist()
    opt_seqlens_ex = []
    opt_headnums_ex = []
    for i in range(len(opt_seqlens)):
        opt_seqlens_ex += [int(opt_seqlens[i])] * gpu_nums[i]
        opt_headnums_ex += [int(opt_headnums[i])] * gpu_nums[i]
    result['seq_lens'] = opt_seqlens_ex
    result['headnums'] = opt_headnums_ex
    write_json_config(
        result, 
        f"examples/qwen/config/{args.model_name}_{args.cluster_type}_seqlen{args.seq_length}.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_type", type=str, default=None, help="gpu types for the cluster. Default=None")
    parser.add_argument("--model_name", type=str, default=None, help="name for the model. Default=None")
    parser.add_argument("--seq_length", type=int, default=None, help="seq_length in the training. Default=None")
    parser.add_argument("--num_query_groups", type=int, default=None, 
                        help="num_query_groups in the training. Default=None")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size in the training. Default=None")
    parser.add_argument("--num_hidden_layers", type=int, default=None, 
                        help="number of hidden layers for the model. Default=None")
    parser.add_argument("--num_attention_heads", type=int, default=None, 
                        help="number of attention heads for the model. Default=None")
    parser.add_argument("--head_dim", type=int, default=None, 
                        help="dim of each head for the model. Default=None")
    parser.add_argument('--gpu_type_num_list', nargs='+', type=int, default=[],
                       help='List of numbers of gpus for different gpu types in the cluster. Default=[]')
    parser.add_argument('--gpu_type_mem_capacity_list', nargs='+', type=int, default=[],
                       help='List of memory capacity of gpus for different gpu types in the cluster. Default=[]')
    parser.add_argument('--model_parameter_size', type=float, default=None,
                       help='The size of model parameter. Use B as the unit. Default=None')
    parser.add_argument("--precision", type=str, default='fp32', choices=['fp32', 'fp16', 'bf16'], 
                       help="precision for the model. Default=fp32")
    args = parser.parse_args()
    search(args)
