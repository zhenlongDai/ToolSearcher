python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/hy240/dzl/checkpoints/MARAG_R1_wCL/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy65/dzl/code/toolPlaner/checkpoints/MARAG_R1_wCL'