from src.utils.retriever.faiss_util import FaissUtil
import src.utils.retriever
from src.utils.retriever.embeddingFactory import EmbeddingFactory

class FaissEmbeddingRetriever:
    def __init__(self, embedding_name, embedding_kwargs, faiss_kwargs):
        """
        embedding_name: EmbeddingFactory中注册的embedding名称
        embedding_kwargs: dict, embedding初始化参数(如model_name, model_path等)
        faiss_kwargs: dict, FaissUtil初始化参数(如dim, index_name, index_dir等)
        """
        EmbeddingFactory.auto_import_all(src.utils.retriever)
        self.embedding_model = EmbeddingFactory.create(embedding_name, **embedding_kwargs)
        faiss_kwargs['dim'] = self.embedding_model.get_embedding_dimension()
        self.faiss_util = FaissUtil(**faiss_kwargs)
        #self.topk = faiss_kwargs.get("topk", 30)

    def add_texts(self, texts: list[str]):
        vectors = self.embedding_model.get_sentence_embeddings(texts)
        self.faiss_util.add(vectors, texts)

    def search(self, query: str, topk: int):
        query_vec = self.embedding_model.get_query_embedding(query)
        #query_vec = query_vec.detach().cpu().numpy().astype('float32')
        results = self.faiss_util.search(query_vec, topk)
        return results

    def save_embedding_dataset(self): # after adding all texts 
        self.faiss_util.save()

    def __len__(self):
        return len(self.faiss_util)