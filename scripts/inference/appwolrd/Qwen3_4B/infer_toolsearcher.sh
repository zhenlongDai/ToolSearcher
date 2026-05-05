# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
DATASET_NAME="appworld"
CONFIG_PATH="$PROJECT_DIR/inference/config"
METHOD_NAME="appworld_test/toolsearcher"
#VAL_DATA="./data/appworld_dataset/tool_selection.parquet"
VAL_DATA="./data/appworld_dataset/test.parquet"

TOOL_CONFIG="$CONFIG_PATH/$DATASET_NAME/api_search_tool_config.yaml"
EXPERIMENT_NAME='toolsearcher_Qwen3_4B_v2' # change topk in config
 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_DISABLED=true
export NCCL_P2P_DISABLE=1


python -m utils.inference_util.SGLangRollout \
    --config-path="$CONFIG_PATH" \
    --config-name='tool_search_multiturn_infer' \
    data.val_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=30000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path='/ossfs/workspace/hy65/dzl/code/toolPlaner/checkpoints/Qwen3/toolsearcher_Qwen3_4B' \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8\
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.project_name="$METHOD_NAME"\
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG"
