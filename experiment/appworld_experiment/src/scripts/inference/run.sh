
experiment_name="test_v1.0"
model_name="gpt-4o-mini-ca"
agent_name="full_code_agent"
dataset_name="test_normal"
python -m src.run \
    --experiment_name "$experiment_name" \
    --model_name "$model_name" \
    --agent_name "$agent_name" \
    --dataset_name "$dataset_name" \
    --with_evaluation \
    --root "./appworld"