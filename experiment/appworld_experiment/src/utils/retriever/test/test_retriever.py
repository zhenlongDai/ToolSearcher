from src.utils.retriever.retriever import FaissEmbeddingRetriever

if __name__ == "__main__":

    # embedding_kwargs = {
    #     "model_name": "unixcoder",
    #     "model_path": "/data/dzl/package/model/unixcoder-base"
    # }
    embedding_kwargs = {
        "model_name": "Qwen3_Embedding",
        "model_path": "/ossfs/workspace/dzl/model/retriever/Qwen3-Embedding-0.6B"
    }
    
    faiss_kwargs = {
        "index_name": "test_index",
        "index_dir": "./src/retrieve_dataset/faiss_indexes",
        "save_embedding_name": "Qwen3_Embedding"
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
    print("检索结果：", results)