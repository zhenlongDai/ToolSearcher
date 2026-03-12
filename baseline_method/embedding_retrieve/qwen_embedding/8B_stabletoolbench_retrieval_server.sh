export CUDA_VISIBLE_DEVICES=1
python -m retriever.retrieval_server \
    --retrieval_api_docs_dataset_path "./experiment/stabletoolbench_experiment/StableToolBench/server/tools"\
    --port 1380 \
    --retrieval_model_name 'Qwen3_Embedding'\
    --retrieval_model_path '/ossfs/workspace/hy58/dzl/model/retriever/Qwen3-Embedding-8B'\
    --save_embedding_name 'Qwen3-Embedding-8B_token512' \
    --index_dir "./data/retrieval_dataset/stabletoolbench/faiss_indexes" \
    --index_name "stabletoolbench_index" \
    --truncate True \
    --max_token_len 512