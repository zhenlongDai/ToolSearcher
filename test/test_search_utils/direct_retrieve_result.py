import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from retriever.base import Config
from retriever.multi_category_retriever import MultiCategoryRetriever
from retriever.toolbench_retreiver import construct_toolbench_category_infos
from utils.retriever_util.retriever import FaissEmbeddingRetriever
class RetrieverServer:
    def __init__(self, config):
        self.config = config
        self.api_docs_dataset_name = self.config.retrieval_api_docs_dataset_name
        self.multicategory_retriever = MultiCategoryRetriever(config)
        if config.retrieval_api_docs_dataset_name == "toolbench":
            self.CategoryInfo_list = construct_toolbench_category_infos(config) 
  
    def search(self, category, query, num, return_score):
        return self.multicategory_retriever.search(query, topk = num, category = category)

class QueryRequest(BaseModel):
    category: Optional[str] = None
    query: str
    topk: Optional[int] = None
    return_scores: bool = False

def test_retrieve(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
       "category": "Commerce",
        "query": "What is an API that scrape the latest CGC comics added on the online market?",
        "topk": 3,
        "return_scores": True
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    return_result = retriever_server.search(
        category=request.category,
        query=request.query,
        num=request.topk,
        return_score=request.return_scores
    )


    result, scores = return_result
    
    # Format response
    resp = []
    if request.return_scores:
        # If scores are returned, combine them with results
        for doc, score in zip(result, scores):
            resp.append({"api_doc": doc, "score": score})

    else:
        resp = result
    return {"result": resp}

def cal_direct_result(query, documents):
    from sentence_transformers import SentenceTransformer
    queries = [
        query
    ]
    # Load the model
    model = SentenceTransformer(config.retrieval_model_path)
    query_embeddings = model.encode(queries, prompt_name="query")
    document_embeddings = model.encode(documents)
    # Compute the (cosine) similarity between the query and document embeddings
    similarity = model.similarity(query_embeddings, document_embeddings)
    print("Similarity:")
    print(similarity)
    print("-----")


def cal_faissretreiver(query, texts):
    embedding_kwargs = {
        "model_name": "Qwen3_Embedding",
        "model_path": "/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B"
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
    retriever.add_texts(texts)
    retriever.save_embedding_dataset()
    results = retriever.search(query, topk=3)
    print("检索结果1：", results)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--retrieval_model_name", type=str, default="Qwen3_Embedding", help="Retrieval model name")
    parser.add_argument("--retrieval_model_path", type=str, default="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B", help="Retrieval model path")
    parser.add_argument("--retrieval_topk", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--index_dir", type=str, default="./data/retrieval_dataset/faiss_indexes", help="Index directory")
    parser.add_argument("--save_embedding_name", type=str, default="Qwen3-Embedding-0.6B", help="Save embedding name")
    parser.add_argument("--index_name", type=str, default="toolbench_index", help="Index name")
    parser.add_argument("--add_retireval_without_category", type=bool, default=True, help="add retireval's mode without category")
    parser.add_argument("--retrieval_api_docs_dataset_name", type=str, default="toolbench", help="Retrieval api docs dataset name")
    parser.add_argument("--retrieval_api_docs_dataset_path", type=str, default="path", help="Retrieval api docs dataset path")
    parser.add_argument("--retrieval_batch_size", type=int, default=128, help="Retrieval batch size")
    parser.add_argument("--use_faiss_gpu", type=bool, default=False, help="Use faiss gpu")
    parser.add_argument("--debug_mode", type=bool, default=False, help="Use faiss gpu")
    
  
    args = parser.parse_args()
    
    # 1) Build a config (could also parse from arguments).
    # In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_model_name=args.retrieval_model_name,
        retrieval_model_path=args.retrieval_model_path,
        retrieval_topk=args.retrieval_topk,
        index_dir=args.index_dir,
        save_embedding_name=args.save_embedding_name,
        index_name=args.index_name,
        use_faiss_gpu=args.use_faiss_gpu,
        retrieval_api_docs_dataset_name=args.retrieval_api_docs_dataset_name,
        retrieval_api_docs_dataset_path=args.retrieval_api_docs_dataset_path,
        retrieval_batch_size=args.retrieval_batch_size,
        add_retireval_without_category=args.add_retireval_without_category,
        debug_mode = args.debug_mode
    )
       
    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever_server = RetrieverServer(config)
    print(config.use_faiss_gpu)
 
    category = "Commerce"
    query = "What is an API that scrape the latest CGC comics added on the online market?"
    topk = 3
    return_scores = True
    
    req = QueryRequest(
        category=category,
        query=query,
        topk=topk,
        return_scores=return_scores
    )
    res = test_retrieve(req)
    print(">>> req result:")
    print(res)
    print("----------")
    documents = []
    scores = []
    res = res['result']
    for res_item in res:
        api_doc = str(res_item['api_doc'])
        score = res_item['score']
        documents.append(api_doc)
        print(type(api_doc))
        scores.append(score)
    
    cal_direct_result(query, documents)
    cal_faissretreiver(query, documents)
    
    
    
    
    
    
    # CategoryInfo_list = retriever_server.CategoryInfo_list
    # target_category = "Commerce"
    # target_category_info = [info for info in CategoryInfo_list if info.category == target_category][0]
    # print(target_category_info.category)
    # print(len(target_category_info.API_docs))
    # ----------
    # tensor([[0.3614, 0.3028, 0.3093]])
    # -----
    # [0.3833269774913788, 0.383326917886734, 0.383326917886734]