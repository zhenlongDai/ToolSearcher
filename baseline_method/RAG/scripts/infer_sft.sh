export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN 
python -m baseline_method.RAG.src.inferencePipeline\
  --config './baseline_method/RAG/configs/rag.yaml' \
  --data_file_path './baseline_method/RAG/data/stabletoolbench_dataset/tool_selection_top100.parquet'\
  --save_file_path "./baseline_method/output_dir/RAG/SFT_Qwen3-4B_top100.json"\
  --use_lora True \
  --lora_path "/ossfs/workspace/hy58/dzl/checkpoints/RAG_SFT_Qwen3-4B/checkpoint-260"
  #--debug_mode True