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

import os
import json
import torch


def read_json_config(path):
    return json.load(open(path, "r", encoding="utf-8"))


def write_json_config(config, path):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=4)


def print_peak_memory(prefix, max_mem, curr_mem):
    print(prefix, "[Allocated]")
    print("\tMax memory: %.2f MB\tCurrent memory : %.2f MB" % (max_mem, curr_mem))


def get_and_print_peak_memory(prefix, device):
    """Get and print peak memory for specific device.

    Args:
        prefix (str): Prefix messages for printing
        device (int): local_rank for current process
        allocation_type (str, optional): Choose from "allocated" or "reserved". Defaults to 'allocated'.

    Returns:
        int, int: Max memory usage and current memory usage
    """
    max_mem = torch.cuda.max_memory_allocated(device) / 2**20
    curr_mem = torch.cuda.memory_allocated(device) / 2**20
    print_peak_memory(prefix, max_mem, curr_mem)
    return max_mem, curr_mem


def get_profile_mem_dict_and_path(args):
    profile_mem_path = f"examples/profile/models/configs/profile_memory_{args.heter_ulysses_model_name}.json"
    return read_json_config(profile_mem_path) \
            if os.path.exists(profile_mem_path) else {}, profile_mem_path


def profile_memory(args, stage=""):
    if args.gpu_type_id == 0:
        local_rank = torch.distributed.get_rank()
        other_key = f"seqlen{args.seq_length}_gqa_group{args.num_query_groups}_bsz{args.global_batch_size}"
    
        profile_mem_dict, profile_mem_path = get_profile_mem_dict_and_path(args)
        
        it = args.curr_iteration

        if stage == "Before Forward":
            torch.cuda.reset_peak_memory_stats(local_rank)
            _, cur_mem = get_and_print_peak_memory("\n" + f"iter{it} " + stage, local_rank)
        else:
            _, cur_mem = get_and_print_peak_memory(f"iter{it} " + stage, local_rank)
        
        mem_dict_key = f"iter{it} {other_key} {stage}"
        profile_mem_dict[mem_dict_key] = cur_mem
        write_json_config(profile_mem_dict, profile_mem_path)