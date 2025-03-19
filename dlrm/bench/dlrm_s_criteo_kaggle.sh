#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch and caffe2

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

# ngpus="1"
# _gpus=$(seq -s, 0 $((ngpus-1)))
# cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
cuda_arg="CUDA_VISIBLE_DEVICES=0"

dlrm_pt_bin="python dlrm_s_pytorch.py"
# dlrm_c2_bin="python dlrm_s_caffe2.py"

# Define the list of mini-batch sizes to iterate over
# mini_batch_sizes=(4096 8192 16384 32768 65536)
mini_batch_sizes=(4096)

# Define the list of num-workers values to iterate over
# num_workers_list=(4 8 12)
num_workers_list=(12)

# Loop over each mini-batch size
for batch_size in "${mini_batch_sizes[@]}"; do
    echo "Running with mini-batch size: $batch_size"
    
    # Loop over each num-workers value
    for num_workers in "${num_workers_list[@]}"; do
        echo "Running with num-workers: $num_workers"
        
        # Repeat 4 times for each batch_size and num-workers
        for i in {1..1}; do
            echo "Iteration $i for mini-batch size $batch_size and num-workers $num_workers"
            
            # Command with the current batch size and num-workers
            cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yuzhuyu/u55c/criteo_sharded/train_1/train.txt --processed-data-file=/home/yuzhuyu/u55c/criteo_sharded/train_1/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --print-freq=100 --print-time --mini-batch-size=${batch_size} --num-workers=${num_workers} --use-gpu $dlrm_extra_option 2>&1 | tee run_kaggle_pt_batch_${batch_size}_workers_${num_workers}_iter_${i}.log"
            
            echo $cmd
            eval $cmd
            
            echo "Finished iteration $i for mini-batch size: $batch_size and num-workers: $num_workers"
            echo -e "\n"
        done
    done
done

echo "done"

# echo "run pytorch ..."
# cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yuzhuyu/u55c/criteo_sharded/train_1/train.txt --processed-data-file=/home/yuzhuyu/u55c/criteo_sharded/train_1/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=8 --use-gpu $dlrm_extra_option 2>&1 | tee run_kaggle_pt_gpu_1.log"
# echo $cmd
# eval $cmd

# echo "run pytorch ..."
# cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yuzhuyu/criteo_sharded/train.txt --processed-data-file=/home/yuzhuyu/criteo_out/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=1024 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=8 --use-gpu $dlrm_extra_option 2>&1 | tee run_kaggle_pt_gpu_1.log"
# echo $cmd
# eval $cmd

# echo "run pytorch ..."
# # WARNING: the following parameters will be set based on the data set
# # --arch-embedding-size=... (sparse feature sizes)
# # --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 $dlrm_extra_option 2>&1 | tee run_kaggle_pt.log

# echo "run caffe2 ..."
# # WARNING: the following parameters will be set based on the data set
# # --arch-embedding-size=... (sparse feature sizes)
# # --arch-mlp-bot=... (the input to the first layer of bottom mlp)
# $dlrm_c2_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time $dlrm_extra_option 2>&1 | tee run_kaggle_c2.log

echo "done"
