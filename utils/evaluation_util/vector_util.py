from typing import Any, Dict, List, Optional, Iterable
import numpy as np
import torch
import utils.retriever_util
from utils.retriever_util.embeddingFactory import EmbeddingFactory
from scipy.spatial.distance import cosine
from tqdm import tqdm
import pickle
import os


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
        if emb.shape != (self.dim,):
            raise ValueError(f"Embedding shape {emb.shape} != ({self.dim},)")
        if _id in self.id2idx:
            # 如果希望覆盖，可以在这里找到原 idx 覆盖 _emb_list[idx]
            # 当前逻辑是忽略
            return

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
        for item in items:
            self.add(item[id_key], item[emb_key])

    def build_matrix(self):
        self._ensure_matrix()

    def get(self, _id: Any) -> Optional[np.ndarray]:
        idx = self.id2idx.get(_id, None)
        if idx is None:
            return None
        self._ensure_matrix()
        return self._emb_matrix[idx]

    def get_batch(self, ids: List[Any]) -> np.ndarray:
        self._ensure_matrix()
        idxs = [self.id2idx[_id] for _id in ids if _id in self.id2idx]
        if not idxs:
            return np.zeros((0, self.dim), dtype=self.dtype)
        return self._emb_matrix[idxs]

    def __len__(self):
        return len(self.id2idx)

    # ========= 新增：保存到本地 =========
    def save(self, path: str):
        """
        将向量和 id 映射保存到本地文件（pickle 格式）。
        """
        self._ensure_matrix()
        data = {
            "dim": self.dim,
            "dtype": self.dtype,
            "id2idx": self.id2idx,
            "emb_matrix": self._emb_matrix,   # 直接存矩阵，比存 list 更快加载
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        """
        从本地文件加载 VectorStore。
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        dim = data["dim"]
        dtype = data.get("dtype", "float32")
        store = cls(dim=dim, dtype=dtype)
        store.id2idx = data["id2idx"]
        store._emb_matrix = data["emb_matrix"].astype(dtype)

        # 如果你之后想继续 add，可以反向拆成 _emb_list
        store._emb_list = [v for v in store._emb_matrix]
        return store


class EmbeddingVectorStore:
    """
    - 输入 {id, text}；用 embedding_encoder 编出向量；
    - 存到内部的 VectorStore；之后可以通过 id / id 列表取回向量。
    """

    def __init__(
        self,
        embedding_name: str,
        model_name: str,
        model_path: str,
        dtype: str = "float32",
    ):
        EmbeddingFactory.auto_import_all(utils.retriever_util)
        embedding_kwargs = {
            "model_name": model_name,
            "model_path": model_path,
            "batch_size": 2,
            "singleton": True
        }
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
        batch_ids: List[Any] = []
        batch_texts: List[str] = []

        def flush_batch():
            if not batch_ids:
                return
            with torch.no_grad():
                tensor_embs = self.encoder.encode(batch_texts)  # [B, dim]
                embs = tensor_embs.cpu().numpy().astype("float32")
                torch.cuda.empty_cache()
                del tensor_embs
            if embs.shape[1] != self.dim:
                raise ValueError(
                    f"Encoder output dim {embs.shape[1]} != expected {self.dim}"
                )
            for _id, emb in zip(batch_ids, embs):
                self.store.add(_id, emb)
            batch_ids.clear()
            batch_texts.clear()
            del embs
            torch.cuda.empty_cache()

        for item in tqdm(items, desc="Building vector store"):
            _id = item[id_key]
            text = item[text_key]
            batch_ids.append(_id)
            batch_texts.append(text)
            if len(batch_ids) >= batch_size:
                flush_batch()

        flush_batch()
        self.store.build_matrix()

    def get_embedding(self, _id: Any) -> Optional[np.ndarray]:
        return self.store.get(_id)

    def get_embeddings(self, ids: List[Any]) -> np.ndarray:
        return self.store.get_batch(ids)

    def __len__(self):
        return len(self.store)

    # ========= 新增：保存 / 加载 整个 EmbeddingVectorStore =========
    def save(self, path: str):
        """
        保存向量仓库到本地。
        只保存向量和 id 映射，不保存 encoder 模型本身（模型可通过参数重建）。
        """
        # path 可以是目录，也可以是具体文件
        if os.path.isdir(path):
            vs_path = os.path.join(path, "vector_store.pkl")
        else:
            vs_path = path

        self.store.save(vs_path)

    @classmethod
    def load(
        cls,
        embedding_name: str,
        model_name: str,
        model_path: str,
        path: str,
        dtype: str = "float32",
    ) -> "EmbeddingVectorStore":
        """
        重新实例化一个 EmbeddingVectorStore，并从文件加载向量。
        embedding_name/model_name/model_path 用于重建 encoder；
        path 为保存的 vector_store.pkl 路径。
        """
        obj = cls(
            embedding_name=embedding_name,
            model_name=model_name,
            model_path=model_path,
            dtype=dtype,
        )
        if os.path.isdir(path):
            vs_path = os.path.join(path, "vector_store.pkl")
        else:
            vs_path = path
        obj.store = VectorStore.load(vs_path)
        obj.dim = obj.store.dim
        return obj





def cal_metric(ground_truth_api_vectors, api_vectors, top_l: int = 3):
    """
    ground_truth_api_vectors: np.ndarray, shape = (N_gold, dim)
    api_vectors: np.ndarray, shape = (N_neg, dim)  # 干扰项集合 D

    返回字典（都是在 gold 维度上再取平均的结果）:
    {
        "max_confuse": float,   # 每个 gold 的“最毒干扰项”，再对所有 gold 取平均
        "topL_confuse": float,  # 每个 gold 的“Top-L 混淆度”，再对所有 gold 取平均
        "mean_confuse": float,  # 每个 gold 的“所有干扰项平均混淆度”，再对所有 gold 取平均
    }
    """
    # 没有 gold 或没有干扰项，直接返回 0
    if len(ground_truth_api_vectors) == 0 or len(api_vectors) == 0:
        return {
            "max_confuse": 0.0,
            "topL_confuse": 0.0,
            "mean_confuse": 0.0,
        }

    per_gold_max = []      # 对每个 gold：最毒干扰项
    per_gold_topL = []     # 对每个 gold：Top-L 混淆度
    per_gold_mean = []     # 对每个 gold：平均混淆度

    for g_vec in ground_truth_api_vectors:
        # 当前这个 gold 和所有干扰项的相似度
        sims = []
        for d_vec in api_vectors:
            s = 1 - cosine(g_vec, d_vec)
            sims.append(s)

        sims = np.array(sims, dtype=np.float32)

        # 1) 这个 gold 的最毒干扰项
        g_max = float(sims.max())

        # 2) 这个 gold 的 Top-L 混淆度
        sorted_sims = np.sort(sims)[::-1]  # 从大到小
        L = min(top_l, len(sorted_sims))
        g_topL = float(sorted_sims[:L].mean()) if L > 0 else 0.0

        # 3) 这个 gold 的所有干扰项平均混淆度
        g_mean = float(sims.mean())

        per_gold_max.append(g_max)
        per_gold_topL.append(g_topL)
        per_gold_mean.append(g_mean)

    # 最后对所有 gold 再取平均，得到整体指标
    #max_confusability = float(np.mean(per_gold_max))
    #topL_confusability = float(np.mean(per_gold_topL))
    #mean_confusability = float(np.mean(per_gold_mean))

    return {
        "per_gold_max": per_gold_max,
        "per_gold_topL": per_gold_topL,
        "per_gold_mean": per_gold_mean,
    }


if __name__ == "__main__":

    items = [
        {"id": "logistics.suivi_colis.latest", "text": "Get the latest tracking status of a parcel."},
        {"id": "tool_42", "text": "Search events by city and date."},
        {"id": 9999, "text": "Retrieve all tracking history for a shipment."},
    ]

    store = EmbeddingVectorStore(
        embedding_name="Qwen3_Embedding",
        model_name="Qwen3_Embedding",
        model_path="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B",
        dtype="float32"
    )

    # 第一次构建并保存
    store.build_from_id_texts(items, id_key="id", text_key="text", batch_size=32)
    store.save("/ossfs/workspace/hy65/dzl/code/toolPlaner/evaluation/eval_confus_score/stabletoolbench/test_vector_store.pkl")

    print("Total vectors:", len(store))  # 3

    v1 = store.get_embedding("logistics.suivi_colis.latest")
    print(type(v1))
    print("v1 shape:", v1.shape)

    batch_vecs = store.get_embeddings(["tool_42", 9999])
    print("batch_vecs shape:", batch_vecs.shape)

    print("cosine similarity:", 1 - cosine(v1, batch_vecs[0]))
    print("cosine similarity:", 1 - cosine(v1, v1))

    # 之后想复用时，直接加载，不用重新 encode
    store2 = EmbeddingVectorStore.load(
        embedding_name="Qwen3_Embedding",
        model_name="Qwen3_Embedding",
        model_path="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B",
        path="/ossfs/workspace/hy65/dzl/code/toolPlaner/evaluation/eval_confus_score/stabletoolbench/test_vector_store.pkl",
    )
    print("Loaded total vectors:", len(store2))
    v1_loaded = store2.get_embedding("logistics.suivi_colis.latest")
    print("loaded v1 equal:", np.allclose(v1, v1_loaded))