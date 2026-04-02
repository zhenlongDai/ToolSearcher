export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN 
python -m baseline_method.APIPredictor.inferencePipeline\
  --config './baseline_method/APIPredictor/configs/api_predictor.yaml' \
  --data_file_path './data/appworld_dataset/apipredictor.parquet'\
  --save_file_path "./baseline_method/output_dir/APIPredictor/appworld.json"
 