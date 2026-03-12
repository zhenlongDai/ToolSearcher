python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/temp_checkpoints/toolsearcher_only_grpo_F1_ablation_search/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy65/dzl/data/checkpoints/toolsearcher_only_grpo_F1_ablation_search'