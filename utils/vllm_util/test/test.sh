export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=WARN 
python -m utils.vllm_util.InferencePipeline_example \
  --config './utils/vllm_util/test/direct.yaml'