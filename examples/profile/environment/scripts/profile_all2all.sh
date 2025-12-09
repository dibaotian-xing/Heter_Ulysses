#!/bin/bash
set -x
#set -e
#set -o pipefail

export PYTHONPATH=$PWD:$PYTHONPATH

export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
export MASTER_PORT=9991
export CUDA_VISIBLE_DEVICES=1,2,3,4

export MASTER_ADDR=localhost
export NCCL_SOCKET_IFNAME=eth0
export NODE_RANK=0

export CLUSTER_TYPE="a6000x4_id1-4"

DISTRIBUTED_ARGS="--nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank=$NODE_RANK"

torchrun $DISTRIBUTED_ARGS ipalg/profile_all2all.py --cluster_type ${CLUSTER_TYPE}