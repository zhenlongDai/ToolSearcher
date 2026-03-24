from typing import Any, Dict, List, Optional, Iterable, Callable
import numpy as np
import torch
import utils.retriever_util
from utils.retriever_util.embeddingFactory import EmbeddingFactory
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch

class VectorStore:

    def __init__(self, dim: int, dtype: str = "float32"):
        self.dim = dim
        self.dtype = dtype
        self._emb_list: List[np.ndarray] = []   # 暂存向量
        self.id2idx: Dict[Any, int] = {}        # 任意 id -> 索引
        self._emb_matrix: Optional[np.ndarray] = None  # [N, dim]，延迟构建

    def _ensure_matrix(self):
        if self._emb_matrix is None:
            if len(self._emb_list) == 0:
                self._emb_matrix = np.zeros((0, self.dim), dtype=self.dtype)
            else:
                self._emb_matrix = np.stack(self._emb_list, axis=0).astype(self.dtype)

    def add(self, _id: Any, emb: np.ndarray):
        """
        添加单条向量：
        - _id: 任意可 hash 的标识符（不要求连续），如 "api_xxx"、123 等；
        - emb: numpy 向量，shape 必须是 (dim,)
        """
        if emb.shape != (self.dim,):
            raise ValueError(f"Embedding shape {emb.shape} != ({self.dim},)")
        if _id in self.id2idx:
            return  # 允许重复添加，但会覆盖
            #raise ValueError(f"ID '{_id}' already exists in VectorStore.")

        idx = len(self._emb_list)
        self.id2idx[_id] = idx
        self._emb_list.append(emb.astype(self.dtype))
        self._emb_matrix = None  # 标记需要重建矩阵

    def add_batch(
        self,
        items: Iterable[Dict[str, Any]],
        id_key: str = "id",
        emb_key: str = "embedding",
    ):
        """批量添加，items: [{"id": 任意id, "embedding": np.ndarray}, ...]"""
        for item in items:
            self.add(item[id_key], item[emb_key])

    def build_matrix(self):
        """可选：手动构建一次大矩阵 [N, dim]（也可以不调，get 时会自动构建）。"""
        self._ensure_matrix()

    def get(self, _id: Any) -> Optional[np.ndarray]:
        """按单个 id 取向量；不存在则返回 None。"""
        idx = self.id2idx.get(_id, None)
        if idx is None:
            return None
        self._ensure_matrix()
        return self._emb_matrix[idx]

    def get_batch(self, ids: List[Any]) -> np.ndarray:
        """
        按一组 id 取向量矩阵：
        - 返回 shape (M, dim)，M 是实际存在的 id 数；
        - 不存在的 id 会跳过。
        """
        self._ensure_matrix()
        idxs = [self.id2idx[_id] for _id in ids if _id in self.id2idx]
        if not idxs:
            return np.zeros((0, self.dim), dtype=self.dtype)
        return self._emb_matrix[idxs]

    def __len__(self):
        return len(self.id2idx)


class EmbeddingVectorStore:
    """
    - 输入 {id, text}；用 embedding_encoder 编出向量；存到内部的 VectorStore；之后可以通过 id / id 列表取回向量。
    """

    def __init__(
        self,
        embedding_name: str,
        model_name: str,
        model_path: str,
        dtype: str = "float32",
    ):
        """
        dim: 向量维度
        embedding_encoder: 函数，输入 List[str]，输出 shape (B, dim) 的 numpy 数组
        """
        EmbeddingFactory.auto_import_all(utils.retriever_util)
        embedding_kwargs = {
            "model_name": model_name, 
            "model_path": model_path,
            "batch_size": 16,
            "singleton": True}
        self.encoder = EmbeddingFactory.create(embedding_name, **embedding_kwargs)
        self.dim = self.encoder.get_embedding_dimension()
        self.store = VectorStore(dim=self.dim, dtype=dtype)

    def build_from_id_texts(
        self,
        items: List[Dict[str, Any]],
        id_key: str = "id",
        text_key: str = "text",
        batch_size: int = 64,
    ):
        """
        批量从 [{id, text}, ...] 构建：
        - 任意 id；
        - text 用 encoder 编成向量；
        """
        batch_ids: List[Any] = []
        batch_texts: List[str] = []

        def flush_batch():
            if not batch_ids:
                return
            with torch.no_grad():
                tensor_embs = self.encoder.encode(batch_texts) # [B, dim]
                embs = tensor_embs.cpu().numpy().astype("float32")
                torch.cuda.empty_cache()
                del tensor_embs
            if embs.shape[1] != self.dim:
                raise ValueError(f"Encoder output dim {embs.shape[1]} != expected {self.dim}")
            for _id, emb in zip(batch_ids, embs):
                self.store.add(_id, emb)
            batch_ids.clear()
            batch_texts.clear()
            del embs
            
            torch.cuda.empty_cache()

        for item in tqdm(items, desc="Building vector store"):
            _id = item[id_key]      # 任意类型 id
            text = item[text_key]
            batch_ids.append(_id)
            batch_texts.append(text)
            if len(batch_ids) >= batch_size:
                flush_batch()

        flush_batch()
        self.store.build_matrix()

    # def add_single(self, _id: Any, text: str):
    #     """单条添加 {id, text}，id 任意、不要求连续"""
    #     embs = self.encoder.encode([text]).cpu().numpy().astype("float32")  # [1, dim]
    #     if embs.shape != (1, self.dim):
    #         raise ValueError(f"Encoder output shape {embs.shape} != (1, {self.dim})")
    #     self.store.add(_id, embs[0])

    def get_embedding(self, _id: Any) -> np.ndarray | None:
        """通过单个 id 取向量"""
        return self.store.get(_id)

    def get_embeddings(self, ids: List[Any]) -> np.ndarray:
        """通过一组 id 取向量矩阵"""
        return self.store.get_batch(ids)

    def __len__(self):
        return len(self.store)

    
def cal_metric(ground_truth_api_vectors, api_vectors):
    """
    计算 api_vector 与 api_vectors 中每个向量的余弦相似度最后取avg
    :param ground_truth_api_vectors: ground_truth_api_vectors
    :param api_vectors: api_vectors
    :return: 每个向量的余弦相似度最后取avg
    """
    cosine_similarities = []
    for ground_truth_api_vector in ground_truth_api_vectors:
        current_cosine_similarities = []
        for api_vector in api_vectors:
            cosine_similarity = 1 - cosine(ground_truth_api_vector, api_vector)
            current_cosine_similarities.append(cosine_similarity)
        cosine_similarities.append(np.mean(current_cosine_similarities))
    return np.mean(cosine_similarities)

if __name__ == "__main__":


    # 准备一些数据，id 是任意标识符，不要求连续、不要求整数：
    items = [
        {"id": "logistics.suivi_colis.latest", "text": "Get the latest tracking status of a parcel."},
        {"id": "tool_42", "text": "Search events by city and date."},
        {"id": 9999, "text": "Retrieve all tracking history for a shipment."},
    ]

    store = EmbeddingVectorStore(
        embedding_name="Qwen3_Embedding", 
        model_name="Qwen3_Embedding", 
        model_path="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B",
        dtype="float32")

    store.build_from_id_texts(items, id_key="id", text_key="text", batch_size=32)

    print("Total vectors:", len(store))  # 3

    # 单个 id 取向量
    v1 = store.get_embedding("logistics.suivi_colis.latest")
    print(type(v1))
    print("v1 shape:", v1.shape)

    # 一组 id 取向量
    batch_vecs = store.get_embeddings(["tool_42", 9999])
    print("batch_vecs shape:", batch_vecs.shape)

    #calc cosine similarity
    print("cosine similarity:", 1 - cosine(v1, batch_vecs[0]))
    print("cosine similarity:", 1 - cosine(v1, v1))