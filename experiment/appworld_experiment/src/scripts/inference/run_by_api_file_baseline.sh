export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
experiment_name="test_normal_GDPO_7B_v1_2"
model_name="gpt-5-mini-ca"
agent_name="full_code_agent"
dataset_name="test_normal" #test_normal/test_challenge
api_file_path="/ossfs/workspace/hy65/dzl/code/toolPlaner/experiment_results/appworld_test/GDPO_wCL/GDPO_wCL_top5.json"

python -m src.run \
    --experiment_name "$experiment_name" \
    --model_name "$model_name" \
    --agent_name "$agent_name" \
    --dataset_name "$dataset_name" \
    --predict_api_mode "predefine" \
    --api_file_path "$api_file_path" \
    --with_evaluation \
    --root "./appworld"