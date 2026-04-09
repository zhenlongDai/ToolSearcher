#!/bin/bash
# PlanThenSelection Inference Script

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN

python -m baseline_method.PlanThenSelection.src.inferencePipeline \
  --config './baseline_method/PlanThenSelection/configs/plan_then_selection.yaml' \
  --data_file_path '/ossfs/workspace/hy65/dzl/code/toolPlaner/data/appworld_dataset/tool_selection.parquet' \
  --save_file_path "./baseline_method/output_dir/PlanThenSelection/appworld_per20.json" \
  #--debug_mode True