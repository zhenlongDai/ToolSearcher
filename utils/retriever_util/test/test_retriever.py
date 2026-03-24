from utils.retriever_util.retriever import FaissEmbeddingRetriever

if __name__ == "__main__":


    embedding_kwargs = {
        "model_name": "Qwen3_Embedding",
        "model_path": "/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B",
        "singleton": True
    }
    
    faiss_kwargs = {
        "index_dir": "/ossfs/workspace/hy65/dzl/code/toolPlaner/data/retrieval_dataset/faiss_indexes",
        "save_embedding_name": "test_Qwen3-Embedding-0.6B",
        "index_name": "test_index",
        "use_faiss_gpu": True
    }
    retriever = FaissEmbeddingRetriever(
        embedding_name=embedding_kwargs['model_name'],
        embedding_kwargs=embedding_kwargs,
        faiss_kwargs=faiss_kwargs
    )
    texts = ["这是一个测试文本", "另一个测试文本", "更多的文本数据用于检索"] 
    retriever.add_texts(texts)
    retriever.save_embedding_dataset()
    results = retriever.search("测试文本", topk=2)
    print("检索结果1：", results)
    
    results = retriever.batch_search(["测试文本"], topk=2)
    print("检索结果2：", results)
    
    results = retriever.batch_search(["测试文本","检索数据"], topk=2)
    print("检索结果3：", results)
    
    