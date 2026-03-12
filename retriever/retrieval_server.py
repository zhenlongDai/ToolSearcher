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
class RetrieverServer:
    def __init__(self, config):
        self.config = config
        self.api_docs_dataset_name = self.config.retrieval_api_docs_dataset_name
        self.multicategory_retriever = MultiCategoryRetriever(config)
    
    def search(self, category, query, num, return_score):
        return self.multicategory_retriever.search(query, topk = num, category = category)



#####################################
# FastAPI server below
#####################################


class QueryRequest(BaseModel):
    category: Optional[str] = None
    query: str
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
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
        combined = []
        for doc, score in zip(result, scores):
            combined.append({"api_doc": doc, "score": score})
        resp.append(combined)
    else:
        resp.append(result)
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument("--retrieval_model_name", type=str, default="Qwen3_Embedding", help="Retrieval model name")
    parser.add_argument("--retrieval_model_path", type=str, default="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B", help="Retrieval model path")
    parser.add_argument("--retrieval_topk", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--index_dir", type=str, default="./data/retrieval_dataset/faiss_indexes", help="Index directory")
    parser.add_argument("--save_embedding_name", type=str, default="Qwen3-Embedding-0.6B", help="Save embedding name")
    parser.add_argument("--index_name", type=str, default="toolbench_index", help="Index name")
    parser.add_argument("--use_faiss_gpu", type=bool, default=False, help="Use faiss gpu")
    parser.add_argument("--add_retireval_without_category", type=bool, default=True, help="add retireval's mode without category")
    parser.add_argument("--retrieval_api_docs_dataset_name", type=str, default="toolbench", help="Retrieval api docs dataset name")
    parser.add_argument("--retrieval_api_docs_dataset_path", type=str, default="path", help="Retrieval api docs dataset path")
    parser.add_argument("--retrieval_batch_size", type=int, default=128, help="Retrieval batch size")
    parser.add_argument("--port", type=int, default=1350, help="port id")
    parser.add_argument("--debug_mode", type=bool, default=False, help="Use faiss gpu")
    parser.add_argument("--max_token_len", type=int, default=300, help="token length of each doc")
    parser.add_argument("--truncate", type=bool, default=False, help="whether truncate api doc")
    
  

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
        debug_mode = args.debug_mode,
        truncate = args.truncate,
        max_token_len = args.max_token_len
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever_server = RetrieverServer(config)
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=args.port)
