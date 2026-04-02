export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
#export WANDB_MODE=offline
accelerate launch --main_process_port 29400 --num_processes 8 --config_file ./baseline_method/SFT/configs/zero2.yaml -m baseline_method.SFT.train \
    --config ./baseline_method/SFT/configs/config_RAG.yaml


# export CUDA_VISIBLE_DEVICES=0,1
# export NCCL_P2P_DISABLE=1
# export WANDB_MODE=offline
# accelerate launch --main_process_port 29400 --num_processes 2 --config_file ./baseline_method/SFT/configs/zero2.yaml -m baseline_method.SFT.train \
#     --config ./baseline_method/SFT/configs/config_RAG.yaml