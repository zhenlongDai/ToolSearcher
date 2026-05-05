python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/hy58/dzl/checkpoints/searchr1_gspo_Qwen3/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy65/dzl/code/toolPlaner/checkpoints/Qwen3/GSPO_Qwen3_4B'