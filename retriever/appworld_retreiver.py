from retriever.base import Config,SingleCategoryInformation
import os
from utils.json_util import load_data_from_json
from utils.appworld_util.api_doc_util import constrcut_appworld_api_doc
from tqdm import tqdm
from typing import List, Union
from utils.appworld_util.format_util import categories
# class SingleCategoryInformation:
#     embedding_kwargs: dict = None
#     faiss_kwargs: dict = None
#     category: str = None 
#     faiss_index_exist: bool = False
#     API_docs: list[dic] = None 


def construct_appworld_category_infos(config: Config) -> list[SingleCategoryInformation]:
    embedding_kwargs =  {
        "model_name": config.retrieval_model_name,
        "model_path": config.retrieval_model_path,
        "singleton": True
    }
    
    faiss_kwargs = {
        "index_dir": config.index_dir,
        "save_embedding_name": config.save_embedding_name,
        "index_name": None,
        "use_faiss_gpu": config.use_faiss_gpu
    }
    
    
    appworld_api_doc_file_path = config.retrieval_api_docs_dataset_path
    result_list = []
    all_standand_API_docs = []
    API_docs_list = load_data_from_json(appworld_api_doc_file_path)
    #1.获取类别
    category_names = categories

    
    faiss_index_path = os.path.join(config.index_dir, config.save_embedding_name, f"{config.index_name}_all")
    all_faiss_index_exist = os.path.exists(faiss_index_path)

    #2.获取对应类别的json文件中的api对象
    print("category_names: ", category_names)
    for category_name in tqdm(category_names, desc="construct appworld infos"):
        faiss_kwargs['index_name'] = f"{config.index_name}_{category_name}"
        faiss_index_path = os.path.join(config.index_dir, config.save_embedding_name, faiss_kwargs['index_name'])
        faiss_index_exist = os.path.exists(faiss_index_path)
        
        print(f"category_name: {category_name}, faiss_index_exist: {faiss_index_exist}")
        if not all_faiss_index_exist:
            standand_API_docs = []
            category_api_list = []
            for item in API_docs_list:
                if 'category_name' not in item:
                    print(f"category_name not in item: {item}")
                    input()
                if item['category_name'] == category_name:
                    category_api_list.append(item)

            #category_api_list = [item for item in API_docs_dict if item['category_name'] == category_name]
            print(f"category_name: {category_name}, category_api_list: {len(category_api_list)}")
            for API_doc in category_api_list:
                standand_API_doc = constrcut_appworld_api_doc(API_doc)
                standand_API_docs.append(str(standand_API_doc))

           
            if config.add_retireval_without_category:
                all_standand_API_docs.extend(standand_API_docs)
        else:
            standand_API_docs = []
            
        result_list.append(SingleCategoryInformation(embedding_kwargs, faiss_kwargs, category_name, faiss_index_exist, standand_API_docs))
    
    
    
    if config.add_retireval_without_category:
        faiss_kwargs['index_name'] = f"{config.index_name}_all"
        result_list.append(SingleCategoryInformation(embedding_kwargs, faiss_kwargs, "all", all_faiss_index_exist, all_standand_API_docs)) 
        
        
    return result_list

