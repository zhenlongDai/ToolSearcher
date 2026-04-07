python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/hy58/dzl/data/checkpoints/searchr1_gspo/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy65/dzl/code/toolPlaner/checkpoints/searchr1_gspo_wCL'