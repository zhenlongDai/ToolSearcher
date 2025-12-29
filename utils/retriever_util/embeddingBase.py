import torch.nn.functional as F
import torch
import numpy as np
import torch
from tqdm import tqdm
class EmbeddingBase:
    def __init__(self, model_name: str, model_path: str, **kwargs):
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = kwargs.get("batch_size", 2)
        print("batchsize of Embedding:", self.batch_size)

    def encode(self, batch_texts):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_query_embedding(self, query: str) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_query_embeddings(self, queries: list[str]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_embedding_dimension(self) -> int:
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_sentence_embeddings(self, texts: list[str])-> list[torch.Tensor]:
        embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size),  desc="get_sentence_embeddings"): 
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.encode(batch_texts)
            if isinstance(batch_embeddings, torch.Tensor):
                new_batch_embeddings = batch_embeddings.detach().cpu().numpy()
            embeddings.extend(new_batch_embeddings)
            del batch_embeddings
            torch.cuda.empty_cache()
        #  embeddings: list of np.ndarray to np.ndarray
        #embeddings = np.array(embeddings).astype('float32')
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        #embeddings = embeddings.astype(np.float32, order="C")
        return embeddings
    
    def compute_similarity(self, query_embedding, doc_embedding):
        # 计算查询向量与文档向量之间的相似度（例如余弦相似度）
        cosine_sim = F.cosine_similarity(query_embedding, doc_embedding, dim=0)
        return cosine_sim.item()
        