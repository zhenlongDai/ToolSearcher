from utils.retriever_util.retriever import FaissEmbeddingRetriever
from retriever.toolbench_retreiver import construct_toolbench_category_infos
from retriever.base import Config
from utils.toolbench_util.format_util import get_target_category
    
    
class MultiCategoryRetriever:
    def __init__(self, retriever_config: Config):
        self.retrievers = {}
        self.retrieval_api_docs_dataset_name = retriever_config.retrieval_api_docs_dataset_name
        self.retrieval_api_docs_dataset_path = retriever_config.retrieval_api_docs_dataset_path
        if self.retrieval_api_docs_dataset_name == "toolbench":
            CategoryInfo_list = construct_toolbench_category_infos(retriever_config) 
            
        for CategoryInfo in CategoryInfo_list:
            category = CategoryInfo.category
            #input(CategoryInfo)
            self.retrievers[category] = FaissEmbeddingRetriever(
                embedding_name = CategoryInfo.embedding_kwargs['model_name'],
                embedding_kwargs = CategoryInfo.embedding_kwargs,
                faiss_kwargs = CategoryInfo.faiss_kwargs
            )
            if not CategoryInfo.faiss_index_exist: 
                self.retrievers[category].add_texts(CategoryInfo.API_docs)
                self.retrievers[category].save_embedding_dataset()
            
    def search(self, query, topk, category):
        if category is None:
            category = "all"
        elif self.retrieval_api_docs_dataset_name == "toolbench":
                category = get_target_category(category)

        if category not in self.retrievers:
            return [f"Category: [{category}] is not within the search scope"], [0]
        else:
            
            search_results = self.retrievers[category].search(query, topk)
            results = []
            scores = []
            for search_result in search_results:
                results.append(search_result['text'])
                scores.append(search_result['score'])
            
            return results, scores