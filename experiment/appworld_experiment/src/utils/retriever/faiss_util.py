import os
import faiss
import numpy as np
import pickle
import torch

class FaissUtil:
    def __init__(self, dim, index_name="faiss.index", index_dir="./faiss_index", save_embedding_name=None):
        self.dim = dim
        self.index_name = index_name
        if save_embedding_name is not None:
            self.index_dir =  os.path.join(index_dir, save_embedding_name)
        else:
            self.index_dir = index_dir
        
        self.index_path = os.path.join(self.index_dir, index_name)
        print(f"Faiss index path: {self.index_path}")
        self.text_path = self.index_path + ".pkl"
        self.index = None
        self.haved_state = False
        self.texts = []
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)

        if os.path.exists(self.index_path):
            self.load()
            self.haved_state = True
            print(f"Faiss index loaded from {self.index_path}")
            print(f"Number of vectors in index: {self.index.ntotal}")
        else:
            self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors, texts: list[str]):
        
        if not isinstance(vectors, np.ndarray):
            if torch.is_tensor(vectors):      #判断如果在gpu上移动到cpu
                vectors = vectors.detach().cpu().numpy().astype('float32')
            elif isinstance(vectors, list):  # 其它类型（如list）
                vectors = np.array(vectors).astype('float32')
            else:
                vectors = np.array(vectors).astype('float32')

        #过滤重复texts in self.texts
        keep_indices = [i for i, text in enumerate(texts) if text not in self.texts]
        if not keep_indices:
            print(" >>> no new texts to add.")
            return

        filtered_vectors = vectors[keep_indices]
        filtered_texts = [texts[i] for i in keep_indices]
        print(f"Adding {len(filtered_texts)} new texts to Faiss index.")
        self.index.add(filtered_vectors)
        self.texts.extend(filtered_texts)

    def save(self):
        if os.path.exists(self.index_path):
            print(f"Faiss index already exists at {self.index_path}, not overwriting.")
        else:
            faiss.write_index(self.index, self.index_path)
            with open(self.text_path, "wb") as f:
                pickle.dump(self.texts, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.text_path):
            with open(self.text_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.texts = []

    def search(self, query, topk: int):
        '''
         query: np.ndarray, shape should be (dim,)
        '''
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy().astype('float32')
        elif not isinstance(query, np.ndarray):
            query = np.array(query).astype('float32')
        if query.ndim == 1:
            query = query.reshape(1, -1)
        if query.dtype != np.float32:
            query = query.astype('float32')
        SimVaule, I = self.index.search(query, topk)
        results = [self.texts[i] if i < len(self.texts) else "" for i in I[0]]
        res_list= []
        # print("Indices:", I)
        # print("Similarity Values:", SimVaule)
        # print("texts:", results)
        for idx in range(len(I[0])):
            res_list.append({
            "indices": I[0][idx].item(),
            "scores": SimVaule[0][idx].item(),
            "texts": results[idx]
        })

        return res_list

    def __len__(self):
        return self.index.ntotal