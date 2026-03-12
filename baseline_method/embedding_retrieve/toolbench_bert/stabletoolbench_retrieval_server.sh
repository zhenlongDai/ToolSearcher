export CUDA_VISIBLE_DEVICES=1
python -m retriever.retrieval_server \
    --retrieval_api_docs_dataset_path "./experiment/stabletoolbench_experiment/StableToolBench/server/tools"\
    --port 1370 \
    --retrieval_model_name 'toolbench_IR_bert'\
    --retrieval_model_path '/ossfs/workspace/hy65/dzl/model/retriever/toolbench_IR_bert'\
    --save_embedding_name 'toolbench_IR_bert' \
    --index_dir "./data/retrieval_dataset/stabletoolbench/faiss_indexes" \
    --index_name "stabletoolbench_index" 