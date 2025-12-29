
export CUDA_VISIBLE_DEVICES=1
python -m retriever.retrieval_server \
    --retrieval_api_docs_dataset_path "/ossfs/workspace/hy65/dzl/code/toolwork/DataConstruct/process_data/tools_v1" #\
#    --use_faiss_gpu True
#    --debug_mode True