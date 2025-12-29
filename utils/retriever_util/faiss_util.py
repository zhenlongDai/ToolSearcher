import os
import faiss
import numpy as np
import pickle
import torch
from tqdm import tqdm

class FaissUtil:
    def __init__(self, dim, index_name="faiss.index", index_dir="./faiss_index", save_embedding_name=None, use_faiss_gpu = False):
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

        if use_faiss_gpu:
            print(">>> use faiss gpu")
            # ngpus = faiss.get_num_gpus()
            # print(f"ngpus = {ngpus}")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
        print(">>> init sucess")
        
    def add(self, vectors, texts: list[str]):
        
        if not isinstance(vectors, np.ndarray):
            if torch.is_tensor(vectors):      #判断如果在gpu上移动到cpu
                new_vectors = vectors.detach().cpu().numpy().astype('float32')
                del vectors
                torch.cuda.empty_cache()
                
            elif isinstance(vectors, list):  # 其它类型（如list）
                new_vectors = np.array(vectors).astype('float32')
            else:
                new_vectors = np.array(vectors).astype('float32')
        else:
            new_vectors = vectors
        #过滤重复texts in self.texts
        keep_indices = [i for i, text in enumerate(texts) if text not in self.texts]
        if not keep_indices:
            print(" >>> no new texts to add.")
            return

        filtered_vectors = new_vectors[keep_indices]
        filtered_texts = [texts[i] for i in keep_indices]
        print(f"Adding {len(filtered_texts)} new texts to Faiss index.")
        self.index.add(filtered_vectors)
        self.texts.extend(filtered_texts)

    def search(self, query, topk: int):
        '''
         query: np.ndarray, shape should be (dim,)
        '''
        if isinstance(query, torch.Tensor):
            new_query = query.detach().cpu().numpy().astype('float32')
            del query
            torch.cuda.empty_cache()
        elif not isinstance(query, np.ndarray):
            new_query = np.array(query).astype('float32')
        else:
            new_query = query
        if new_query.ndim == 1:
            new_query = new_query.reshape(1, -1)
        if new_query.dtype != np.float32:
            new_query = new_query.astype('float32')
            
        SimVaule, I = self.index.search(new_query, topk)
        text_results = [self.texts[i] if i < len(self.texts) else "" for i in I[0]]
        res_list= []
        for idx in range(len(I[0])):
            res_list.append({
            "indix": I[0][idx].item(),
            "score": SimVaule[0][idx].item(),
            "text": text_results[idx]
        })

        return res_list

    def batch_search(self, query_list: np.ndarray, topk: int = None, batch_size = 2):
        '''
        Batch search for multiple queries
        Args:
            query_list: np.ndarray or list, shape should be (num_queries, dim)
            topk: int, number of top results to return
            batch_size: int, number of queries to process at once
        Returns:
            list of lists, each sublist contains topk results for one query
        '''
        # Convert input to numpy array
        if isinstance(query_list, torch.Tensor):
            new_query_list = query_list.detach().cpu().numpy().astype('float32')
            del query_list
            torch.cuda.empty_cache()
        elif not isinstance(query_list, np.ndarray):
            new_query_list = np.array(query_list).astype('float32')
        else:
            new_query_list = query_list
        # Ensure correct shape
        if new_query_list.ndim == 1:
            new_query_list = new_query_list.reshape(1, -1)
        if new_query_list.dtype != np.float32:
            new_query_list = new_query_list.astype('float32')
        
        num_queries = new_query_list.shape[0]
           
        all_results = []
        
        # Process in batches with progress bar
        for start_idx in tqdm(range(0, num_queries, batch_size), desc="Batch searching"):
            end_idx = min(start_idx + batch_size, num_queries)
            batch_queries = new_query_list[start_idx:end_idx]
            
            # Perform batch search
            SimValues, Indices = self.index.search(batch_queries, topk)
            
            # Process results for each query in the batch
            for batch_idx in range(len(batch_queries)):
                current_result = []
                #print(len(Indices[batch_idx]))
                for idx in range(len(Indices[batch_idx])):
                    text_result = self.texts[Indices[batch_idx][idx]] if Indices[batch_idx][idx] < len(self.texts) else ""
                    current_result.append({
                        "indix": Indices[batch_idx][idx].item(),
                        "score": SimValues[batch_idx][idx].item(),
                        "text": text_result
                    })
                all_results.append(current_result)
        
        return all_results
    
    def __len__(self):
        return self.index.ntotal
    
    def save(self):
        if os.path.exists(self.index_path):
            print(f"Faiss index already exists at {self.index_path}, not overwriting.")
        else:
            index_to_save = self.index
    
            if hasattr(faiss, 'GpuIndex') and isinstance(self.index, faiss.GpuIndex):
                print("Converting GPU index to CPU for saving...")
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            elif hasattr(self.index, 'at'):  # GpuIndexProxy (多GPU情况)
                print("Converting multi-GPU index to CPU for saving...")
                index_to_save = faiss.index_gpu_to_cpu(self.index)
                
            faiss.write_index(index_to_save, self.index_path)
            with open(self.text_path, "wb") as f:
                pickle.dump(self.texts, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.text_path):
            with open(self.text_path, "rb") as f:
                self.texts = pickle.load(f)
        else:
            self.texts = []
