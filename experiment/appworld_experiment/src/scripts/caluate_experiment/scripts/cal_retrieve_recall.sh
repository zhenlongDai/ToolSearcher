python -m src.caluate_experiment.cal_retrieve_Recall \
    --experiment_name Qwen3_Embedding_retrieve_recall \
    --model_name Qwen3_Embedding \
    --dataset_name train 

# python -m src.caluate_experiment.cal_retrieve_Recall \
#     --experiment_name unixcoder_retrieve_recall \
#     --model_name unixcoder \
#     --dataset_name dev 
# 6bdbc26_1, 6bdbc26_2, 6bdbc26_3
# {
#     "embedding_name": "Qwen3_Embedding",
#     "embedding_kwargs": {
#         "model_name": "Qwen3_Embedding",
#         "model_path": "/data/LLMs/Qwen3-Embedding-0.6B"
#     },
#     "faiss_kwargs": {
#         "index_name": "appworld_index",
#         "index_dir": "./src/retrieve_dataset/faiss_indexes",
#         "save_embedding_name": "Qwen3-Embedding-0.6B"
#     },
#     "topk": 30
# }