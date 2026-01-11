from retriever.base import Config,SingleCategoryInformation
from utils.file_util import get_all_folder_name, get_all_json_file_name
import os
from utils.json_util import load_data_from_json
from utils.toolbench_util.api_doc_util import constrcut_toolbench_api_docs, get_api_docs
from tqdm import tqdm
from utils.toolbench_util.format_util import get_target_category
from typing import List, Union
from transformers import AutoTokenizer

# class SingleCategoryInformation:
#     embedding_kwargs: dict = None
#     faiss_kwargs: dict = None
#     category: str = None 
#     faiss_index_exist: bool = False
#     API_docs: list[dic] = None 

def merge_similar_category_info(infolist: list[SingleCategoryInformation], index_name) -> list[SingleCategoryInformation]: 
    new_info_map = {}
    for info in infolist:
        new_category = get_target_category(info.category)
        print("origin category", info.category)
        print("new category", new_category)

        if new_category in new_info_map:
            new_info_map[new_category].API_docs.extend(info.API_docs)
        else:
            info.category = new_category
            info.faiss_kwargs['index_name'] = f"{index_name}_{new_category}"
            faiss_index_path = os.path.join(info.faiss_kwargs['index_dir'], info.faiss_kwargs['save_embedding_name'], info.faiss_kwargs['index_name'])
            info.faiss_index_exist = os.path.exists(faiss_index_path)
            new_info_map[new_category] = info
    new_info_list = list(new_info_map.values())
    return new_info_list

def truncate_api_docs(
    standard_API_docs: Union[str, List[str]],
    retrieval_model_path: str,
    max_len: int = 300
) -> Union[str, List[str]]:
    """
    按 tokenizer 的 token 长度截断 API 文档内容。
    参数：
        standard_API_docs: 字符串或字符串列表
        retrieval_model_path: 用于加载 tokenizer 的模型路径/名称
        max_len: 最大 token 长度（含 special tokens）
    返回：
        截断后的字符串或字符串列表（与输入类型相同）
    """
    tokenizer = AutoTokenizer.from_pretrained(retrieval_model_path)
    def _truncate_one(text: str) -> str:
        # 编码为 token id（不返回 tensor，避免维度问题）
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_len,
            return_attention_mask=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]  # List[int]
        # 直接 decode 截断后的 token
        truncated_text = tokenizer.decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return truncated_text
    if isinstance(standard_API_docs, str):
        return _truncate_one(standard_API_docs)
    else:
        return [_truncate_one(doc) for doc in standard_API_docs]

def construct_toolbench_category_infos(config: Config) -> list[SingleCategoryInformation]:
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
    
    
    toolbench_tools_dir = config.retrieval_api_docs_dataset_path
    result_list = []
    all_standand_API_docs = []
    #1.获取指定目录下所有文件夹名类别
    category_names = get_all_folder_name(toolbench_tools_dir)
    if config.debug_mode: category_names = category_names[:5]
    
    
    faiss_index_path = os.path.join(config.index_dir, config.save_embedding_name, f"{config.index_name}_all")
    all_faiss_index_exist = os.path.exists(faiss_index_path)
    #2.获取对应类别的json文件中的api对象
    for category_name in tqdm(category_names, desc="construct toolbench_category infos"):
        faiss_kwargs['index_name'] = f"{config.index_name}_{category_name}"
        faiss_index_path = os.path.join(config.index_dir, config.save_embedding_name, faiss_kwargs['index_name'])
        faiss_index_exist = os.path.exists(faiss_index_path)
        
        category_tools_path = os.path.join(toolbench_tools_dir, category_name)
        if not all_faiss_index_exist:
            API_docs = get_api_docs(category_name, category_tools_path)
            standand_API_docs = constrcut_toolbench_api_docs(API_docs)
           
            standand_API_docs = [str(api_doc) for api_doc in standand_API_docs]
            standand_API_docs = truncate_api_docs(standand_API_docs, config.retrieval_model_path)

            if config.add_retireval_without_category:
                all_standand_API_docs.extend(standand_API_docs)
        else:
            standand_API_docs = []
            
        result_list.append(SingleCategoryInformation(embedding_kwargs, faiss_kwargs, category_name, faiss_index_exist, standand_API_docs))
    
    result_list = merge_similar_category_info(result_list, config.index_name)
    
    if config.add_retireval_without_category:
        faiss_kwargs['index_name'] = f"{config.index_name}_all"
        result_list.append(SingleCategoryInformation(embedding_kwargs, faiss_kwargs, "all", all_faiss_index_exist, all_standand_API_docs)) 
        
        
    return result_list

#def check_faiss_status_by_count()