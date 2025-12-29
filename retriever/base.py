import copy

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_model_name: str = "Qwen3_Embedding", 
        retrieval_model_path: str = "/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B", 
        retrieval_topk: int = 5,
        index_dir: str = "./data/retrieval_dataset/toolbench_faiss_indexes",
        save_embedding_name: str = "Qwen3-Embedding-0.6B",
        index_name: str = "toolbench_index",
        use_faiss_gpu: bool = True,
        retrieval_api_docs_dataset_name: str = "toolbench",
        retrieval_api_docs_dataset_path: str = "path", 
        retrieval_batch_size: int = 128,
        add_retireval_without_category: bool= True,
        debug_mode: bool = False
    ):
        self.retrieval_model_name = retrieval_model_name
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_topk = retrieval_topk
        self.index_dir = index_dir
        self.save_embedding_name = save_embedding_name
        self.index_name = index_name # final path is {index_dir}/{save_embedding_name}/{index_name}.index
        self.use_faiss_gpu = use_faiss_gpu
        self.retrieval_api_docs_dataset_name = retrieval_api_docs_dataset_name
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_api_docs_dataset_path = retrieval_api_docs_dataset_path
        self.add_retireval_without_category = add_retireval_without_category
        self.debug_mode = debug_mode
        
class SingleCategoryInformation:
    def __init__(
    self,
    embedding_kwargs: dict = None,
    faiss_kwargs: dict = None,
    category: str = None,
    faiss_index_exist: bool = False,
    API_docs: list[str] = None
    ):
        # 深拷贝，防止嵌套对象被共享
        self.embedding_kwargs = copy.deepcopy(embedding_kwargs) if embedding_kwargs else {}
        self.faiss_kwargs = copy.deepcopy(faiss_kwargs) if faiss_kwargs else {}
        self.category = category
        self.faiss_index_exist = faiss_index_exist
        self.API_docs = copy.deepcopy(API_docs) if API_docs else []

    def __repr__(self):
        return (f"SingleCategoryInformation(\n"
                f"  category={self.category},\n"
                f"  faiss_index_exist={self.faiss_index_exist},\n"
                f"  embedding_kwargs={self.embedding_kwargs},\n"
                f"  faiss_kwargs={self.faiss_kwargs},\n"
                f"  API_docs_count={len(self.API_docs) if self.API_docs else 0}\n"
                f")")
    
