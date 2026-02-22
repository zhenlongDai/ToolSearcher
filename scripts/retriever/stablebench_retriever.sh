export CUDA_VISIBLE_DEVICES=1
python -m retriever.retrieval_server \
    --retrieval_api_docs_dataset_path "/ossfs/workspace/hy65/dzl/code/toolPlaner/experiment/stabletoolbench_experiment/StableToolBench/server/tools"\
    --port 1360 \
    --index_dir "./data/retrieval_dataset/stabletoolbench/faiss_indexes" \
    --index_name "stabletoolbench_index"
#    --use_faiss_gpu True
#    --debug_mode True