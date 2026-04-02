python -m baseline_method.construct_data.construct_training_data \
  --origin_data_file "./data/toolplan_qarquet_data/train.parquet"\
  --save_local_dir "./baseline_method/dataset/RAG_SFT_data"\
  --prompt_template_path "./baseline_method/RAG/prompt_template/api_search_prompt.txt"\
  --save_file_name "train"\
  --data_mode "full" \
  --port 1350\
  --topk 30 \
  --truncate_mode "yes"\
  --tokenizer_path '/ossfs/workspace/hy65/dzl/model/Qwen2.5-7B-Instruct'


python -m baseline_method.construct_data.construct_training_data \
  --origin_data_file "./data/toolplan_qarquet_data/eval.parquet"\
  --save_local_dir "./baseline_method/dataset/RAG_SFT_data"\
  --prompt_template_path "./baseline_method/RAG/prompt_template/api_search_prompt.txt"\
  --save_file_name "dev"\
  --data_mode "full" \
  --port 1350\
  --topk 30 \
  --truncate_mode "yes"\
  --tokenizer_path '/ossfs/workspace/hy65/dzl/model/Qwen2.5-7B-Instruct'


