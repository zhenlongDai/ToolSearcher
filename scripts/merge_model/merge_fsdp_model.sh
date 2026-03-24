python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/temp_checkpoints/toolsearcher_correct_search_mask/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy57/dzl/checkpoints/toolsearcher_correct_search_mask'