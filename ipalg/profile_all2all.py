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
"""Profile all2all communication."""

import argparse
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import time

import torch
import datetime

import torch.distributed
from ipalg.utils import read_json_config, write_json_config


def write_profile_result(comm_coe, world_size, cluster_type):
    print("********************")
    print("Final result:")
    print("a2a_coe (ms/MB): ", comm_coe)
    print("********************")
    comm_type = "all2all"
    total_num_gpus = world_size
    env_config_path = (
        f"./examples/profile/environment/configs/{comm_type}_bandwidth_"
        f"{int(total_num_gpus)}_gpus{cluster_type}.json"
    )
    config = read_json_config(env_config_path) if os.path.exists(env_config_path) else {}
    config["a2a_coe"] = comm_coe
    write_json_config(config, env_config_path)
    print("Already written a2a_coe into env config file %s!" % (env_config_path))


def profile(args):
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=2))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    seqlen_per_rank = 2048
    bsz = 2
    headnum_q = 16
    headnum_kv = 8

    all2all_tensor_q = torch.randn(
        (world_size, seqlen_per_rank, bsz, headnum_q // world_size, 128), 
        device=device, 
        dtype=torch.float16
    )

    all2all_tensor_kv = torch.randn(
        (world_size, seqlen_per_rank, bsz, headnum_kv // world_size, 128), 
        device=device, 
        dtype=torch.float16
    )

    warmup_iters, iters = 1, 10
    all2all_stream = torch.cuda.Stream()
    a2a_message_size_q = 2 * (world_size - 1) / \
        world_size * seqlen_per_rank * bsz * headnum_q * 128 * 2 * iters / 1024 / 1024
    a2a_message_size_kv = 2 * (world_size - 1) / \
        world_size * seqlen_per_rank * bsz * headnum_kv * 128 * 2 * iters / 1024 / 1024
    a2a_message_size = a2a_message_size_q + 2 * a2a_message_size_kv
    cluster_type = args.cluster_type

    def all2all_comm(stream):
        reqs = []
        output = [torch.empty_like(all2all_tensor_q)] + [torch.empty_like(all2all_tensor_kv)] * 2
        for i in range(4):
            if i < 3:
                reqs.append(torch.distributed.all_to_all_single(
                    output[i],
                    all2all_tensor_kv if i != 0 else all2all_tensor_q,
                    async_op=True
                ))
            if i > 0:
                with torch.cuda.stream(stream):
                    reqs[i - 1].wait()
        torch.cuda.current_stream().wait_stream(stream)

    #warm
    x = torch.ones((1), dtype=torch.float16).to(device)
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

    def trace_handler(prof):
        print(f"Rank {rank} in trace_handler")
        
        #warm up, deadlock otherwise
        test_t = torch.ones((1)).to(device)
        torch.distributed.all_reduce(
            test_t, 
            op=torch.distributed.ReduceOp.SUM
        )
        
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
        time.sleep(0.5 * rank)
        print("Results  rank %d :" % (rank))
        print(table)
        table = table.split("\n")

        def split_line(line):
            line = line.split("  ")
            ls = []
            for s in line:
                if len(s):
                    ls.append(s.strip())
            return ls

        def str2time(s):
            if "ms" in s:
                return float(s[:-2])
            elif "us" in s:
                return float(s[:-2]) * 1e-3
            else:
                return float(s[:-1]) * 1e3

        for line in table:
            if "Name" in line:
                title = split_line(line)
            if "ncclDevKernel_SendRecv" in line:
                result = split_line(line)
        for i, _ in enumerate(title):
            # print('%s: %s'%(title[i],result[i]))
            if "CUDA total" in title[i]:
                cuda_total_idx = i
        a2a_time = str2time(result[cuda_total_idx])
        comm_coe = a2a_time / a2a_message_size

        print("**********")
        print(f"a2a_coe (ms/MB): ", comm_coe)
        print("**********")

        # pylint: disable=not-callable
        comm_coe = torch.tensor([comm_coe]).to(device)
        torch.distributed.all_reduce(
            comm_coe, 
            op=torch.distributed.ReduceOp.SUM
        )
        comm_coe = comm_coe.cpu().numpy()[0] / world_size
        if rank == 0:
            time.sleep(1)
            write_profile_result(comm_coe, world_size, cluster_type)

    if world_size == 1:
        write_profile_result(0, world_size, args)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
        on_trace_ready=trace_handler,
    ) as p:
        # Warming up
        if rank == 0:
            print("Warming up...")
        for _ in range(warmup_iters):
            all2all_comm(all2all_stream)
        p.step()

        if rank == 0:
            print("Profiling...")
        for _ in range(iters):
            all2all_comm(all2all_stream)
        p.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_type", type=str, default=None, help="gpu types for the cluster. Default=None")
    profile_args = parser.parse_args()
    profile(profile_args)