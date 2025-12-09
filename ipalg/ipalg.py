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
"""Mixed Integer Quadratic Programming"""

import math

import gurobipy as gp
import numpy as np
from gurobipy import GRB
import json


class IPAlg:
    """Integer programming algorithm
    
    Args:
        num_gpu_type(int): number of gpu types
        gpu_nums(np.ndarray): gpu numbers of per gpu type
        sl_tot(int): the seqlen of the whole training process
        gn_tot(int): number of GQA groups of the whole training process
        comm_e(float): the communication coefficient between different gpus
        time_g(np.ndarray): time to compute forward propagation of a single GQA group of attention 
                    for the whole sequence per gpu type
        time_l(np.ndarray): time of per gpu type to compute forward propagation of the whole transformer layer 
                    (except attention block) for a single token
        mem_g(int): the memory cost of activation per GQA group when compute attention block 
        mem_l(int): the memory cost of activation per token when compute the whole transformer layer 
                    (except attention block)
        mem_e(int): the sum of activation memory cost per token in non-transformer layers
        M(np.ndarray): the memory capacity for activations of per gpu type
        bsz(int): training batch size
        hd(int): hidden size per attention head
        nt(int): the number of transformer layers in the model
        nq(int): the number of queries in each GQA group
        precision(string): fp32, fp16 or bf16
    """

    def __init__(self, num_gpu_type, gpu_nums, sl_tot, gn_tot, comm_e, 
                    time_g, time_l, mem_g, mem_l, mem_e, M, bsz, hd, nt, nq, precision) -> None:
        assert gpu_nums.ndim == 1
        assert time_g.ndim == 1
        assert time_l.ndim == 1
        assert M.ndim == 1
        assert gpu_nums.shape[0] == num_gpu_type
        assert time_g.shape[0] == num_gpu_type
        assert time_l.shape[0] == num_gpu_type
        assert M.shape[0] == num_gpu_type
        assert precision in ('fp32', 'fp16', 'bf16')

        self.num_gpu_type = num_gpu_type
        self.gpu_nums = gpu_nums
        self.sl_tot = sl_tot
        self.gn_tot = gn_tot
        self.comm_e = comm_e
        self.time_g = time_g
        self.time_l = time_l
        self.mem_g = mem_g
        self.mem_l = mem_l
        self.mem_e = mem_e
        self.M = M
        self.bsz = bsz
        self.hd = hd
        self.nt = nt
        self.nq = nq
        self.c_type = 2 if precision in ('fp16', 'bf16') else 4

        self.model = gp.Model("Minimize heterogeneous Deepspeed Ulysses")
        self.model.setParam("LogToConsole", 1)
        # self.model.setParam("LogFile", "gurobi.log")
        self.model.setParam("MIPGap", 1e-4)
        self.model.setParam("TimeLimit", 30)
        self.model.setParam("MIPFocus", 1)
        self.model.setParam("NonConvex", 2)
        self.model.setParam("NumericFocus", 1)

    def fit(self):
        return self._do_fit()

    def _do_fit(self):
        seqlens = self.model.addMVar((self.num_gpu_type), vtype=GRB.INTEGER)
        num_gqa_groups = self.model.addMVar((self.num_gpu_type), vtype=GRB.INTEGER)
        self.model.addConstr(seqlens >= 1)
        self.model.addConstr(num_gqa_groups >= 1)
        self.model.addConstr(self.sl_tot == gp.quicksum(seqlens * self.gpu_nums))
        self.model.addConstr(self.gn_tot == gp.quicksum(num_gqa_groups * self.gpu_nums))
        for i in range(self.num_gpu_type):
            self.model.addConstr(
                self.M[i] >= self.bsz * \
                    (self.mem_e * seqlens[i] + self.nt * (self.mem_g * num_gqa_groups[i] + self.mem_l * seqlens[i]))
            )
        
        all2all_times = self.model.addMVar((self.num_gpu_type), vtype=GRB.CONTINUOUS)
        for i in range(self.num_gpu_type):
            self.model.addConstr(
                all2all_times[i] == self.hd * self.bsz * self.c_type * self.comm_e * (5 * self.nq + 4) * \
                    (num_gqa_groups[i] * self.sl_tot + seqlens[i] * self.gn_tot - 2 * num_gqa_groups[i] * seqlens[i])
            )
        all2all_time_max = self.model.addVar(vtype=GRB.CONTINUOUS)
        self.model.addConstr(all2all_time_max == gp.max_(all2all_times.tolist()))
        
        compute_times = self.model.addMVar((self.num_gpu_type), vtype=GRB.CONTINUOUS)
        for i in range(self.num_gpu_type):
            self.model.addConstr(
                compute_times[i] == self.bsz * (self.time_g[i] * num_gqa_groups[i] + self.time_l[i] * seqlens[i])
            )
        compute_time_max = self.model.addVar(vtype=GRB.CONTINUOUS)
        self.model.addConstr(compute_time_max == gp.max_(compute_times.tolist()))

        tot_time = self.model.addVar(vtype=GRB.CONTINUOUS)
        self.model.addConstr(tot_time == all2all_time_max + compute_time_max * 3.5)
        self.model.setObjective(tot_time, GRB.MINIMIZE)
        self.model.optimize()

        if self.model.Status == GRB.Status.INFEASIBLE:
            print("No solution for current batch size and current memory limit.")
            return math.inf, None, None
        elif self.model.Status == GRB.Status.OPTIMAL:
            print("Final Solution = ", tot_time.X)
        else:
            # TODO: support more status codes, such as TLE and so on.
            raise RuntimeError(f"Wrong status code {self.model.Status}")
        print("Total time is {}".format(tot_time.X))
        return tot_time.X, seqlens.X, num_gqa_groups.X


# num_gpu_type = 2
# #suppose gpu type1 ranks 0-3, type2 ranks 4-7
# gpu_nums = np.array([4, 4])
# sl_tot = 2048
# hn_tot = 32
# comm_e = 0.012351302083333333 / 1024 / 1024
# time_a_tmp = 0.278662 / 8
# time_a = np.array([time_a_tmp, time_a_tmp * 1.2])
# time_o_tmp = (1.6565558314323425 - time_a_tmp * hn_tot) / sl_tot
# time_l = np.array([time_o_tmp, time_o_tmp * 1.2])
# hd = 128
# bsz = 1
# nt = 32
# # softmax fp32 others fp16, QK^T, softmax(QK^T), softmax(QK^T)V
# mem_a = 2 * (sl_tot * sl_tot * (2 + 4)) + sl_tot * hd * 2
# mem_l = int((636.015625 * 1024 * 1024 - mem_a * hn_tot) / sl_tot + 0.5)
# mem_e = int(84.0595703125 * 1024 * 1024 / 1024 + 0.5)
# M_tmp = int((32 - 7 * 16 * (1 / 8 + 0.003)) * 1024 * 1024 * 1024) 
# M = np.array([M_tmp, M_tmp])
# print(M)
# precision = 'fp16'

# ipalg = IPAlg(
#             num_gpu_type,
#             gpu_nums,
#             sl_tot,
#             hn_tot,
#             comm_e,
#             time_a,
#             time_l,
#             mem_a,
#             mem_l,
#             mem_e,
#             M,
#             bsz,
#             hd,
#             nt,
#             precision
#         )

# opt_tot_time, opt_seqlens, opt_headnums = ipalg.fit()
# print('optimized total time:', opt_tot_time)
# print('optimized sequence lengths:', opt_seqlens)
# print('optimized head nums:', opt_headnums)

# result = {}
# result['num_gpu_type'] = num_gpu_type
# result['gpu_nums'] = gpu_nums.tolist()
# opt_seqlens_ex = []
# opt_headnums_ex = []
# for i in range(len(opt_seqlens)):
#     opt_seqlens_ex += [int(opt_seqlens[i])] * gpu_nums[i]
#     opt_headnums_ex += [int(opt_headnums[i])] * gpu_nums[i]
# result['seq_lens'] = opt_seqlens_ex
# result['headnums'] = opt_headnums_ex
# with open("examples/qwen/config/test.json", "w", encoding="utf-8") as f:
#     json.dump(result, f, indent=4)