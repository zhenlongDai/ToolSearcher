# run on 8xH20
# make sure your current working directory is the root of the project

set -x

ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"


TRAIN_DATA="/ossfs/workspace/hy65/dzl/data/tool_data/searchr1/data/searchR1_processed_direct/train_part.parquet"
VAL_DATA="/ossfs/workspace/hy65/dzl/data/tool_data/searchr1/data/searchR1_processed_direct/test.parquet"

TOOL_CONFIG="$CONFIG_PATH/tool_config/search_tool_config.yaml"
EXPERIMENT_NAME='qwen2.5-3b-instruct_function_weave_v1' 
# mkdir -p logs

# # Set GPUs and run with a suitable log path
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# nohup bash examples/sglang_multiturn/search_r1_like/run_qwen2.5-3b_instruct_search_multiturn.sh \
#   trainer.experiment_name=qwen2.5-3b-it_rm-searchR1-like-sgl-multiturn-$(now) \
#   > logs/searchR1-like$(now).log 2>&1 &
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export WANDB_MODE=offline
#export WANDB_VERIFY_SSL=false

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='search_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=3000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path='/ossfs/workspace/hy65/dzl/model/Qwen2.5-3B-Instruct' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=10000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='search_r1_like_async_rl' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA"  \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_epochs=1  \
    actor_rollout_ref.rollout.calculate_log_probs=True\
    2>&1 | tee "logs/searchR1-like/$EXPERIMENT_NAME.log"

# python /ossfs/workspace/hy65/dzl/code/simple_gpu_forward.py
#actor_rollout_ref.rollout.trace.backend=weave\
#actor_rollout_ref.rollout.mode=async\