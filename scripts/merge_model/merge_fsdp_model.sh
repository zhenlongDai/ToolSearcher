python -m utils.verl_util.model_merge \
    'merge' \
    --backend 'fsdp' \
    --local_dir '/ossfs/workspace/hy65/dzl/code/toolPlaner/checkpoints/tool_plan/qwen2.5-7b-instruct_tool_plan_test_v2/global_step_56/actor'\
    --target_dir '/ossfs/workspace/hy65/dzl/code/toolPlaner/checkpoints/qwen2.5-7b-instruct_tool_plan_grpo'