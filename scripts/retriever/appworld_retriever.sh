export CUDA_VISIBLE_DEVICES=1
python -m retriever.retrieval_server \
    --retrieval_api_docs_dataset_name "appworld" \
    --retrieval_api_docs_dataset_path "./data/appworld_dataset/api_docs.json"\
    --port 1360 \
    --index_dir "./data/retrieval_dataset/appworld/faiss_indexes" \
    --index_name "appworld_index" \
#    --use_faiss_gpu True
#    --debug_mode True 