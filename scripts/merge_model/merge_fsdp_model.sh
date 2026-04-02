python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/checkpoints/toolsearcher_s1_v1/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy58/dzl/data/checkpoints/toolsearcher_s1_v1'