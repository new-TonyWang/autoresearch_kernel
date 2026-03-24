#!/bin/bash
# Run all benchmark rounds and save logs to respective directories

source /inspire/hdd/global_user/wangtongyu-25057/miniconda3/etc/profile.d/conda.sh
cd /inspire/hdd/project/qianghuaxuexi/public/wty/ai4ai/autoresearch_kernel/001_attention_softmax_dropout_value_matmul_backward

export CUDA_VISIBLE_DEVICES=1

for round in round_25 round_26; do
    echo "Testing $round..."
    logfile="$round/benchmark.log"
    echo "=== Testing $round at $(date) ===" > "$logfile"
    
    conda run -n ai4ai sol-execbench --compile-timeout 600 . --solution ./$round/solution.json >> "$logfile" 2>&1
    
    echo "=== Completed $round at $(date) ===" >> "$logfile"
    echo "$round done"
done

