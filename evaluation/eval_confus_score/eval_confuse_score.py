import argparse
import pandas as pd
from utils.json_util import load_list_from_json 
from tqdm import tqdm 
from utils.evaluation_util.metric_util import cal_f1_recall_precision, cal_f1_recall_precision_from_seach_apis, ndcg_at_k_sklearn
from utils.evaluation_util.metric_util import is_match

from utils.toolbench_util.api_doc_util import constrcut_toolbench_api_docs, get_api_docs, get_standardize_api_name
from utils.file_util import get_all_folder_name
from utils.json_util import read_parquet_to_list
from utils.evaluation_util.vector_util import EmbeddingVectorStore, cal_metric
import os
import numpy as np
from utils.evaluation_util.format_util import parse_tool_list
from utils.evaluation_util.format_util import parse_tool_apiname_lists_from_retrieval_content

def get_api_docs_from_file(toolbench_tools_dir):
    #1.获取指定目录下所有文件夹名类别
    category_names = get_all_folder_name(toolbench_tools_dir)
    toolbench_API_docs_dic = {}
    #2.获取对应类别的json文件中的api对象
    #category_names = category_names[:3]
    for category_name in tqdm(category_names, desc="construct toolbench_category infos"):
        #if category_name!="Mapping":continue

        category_tools_path = os.path.join(toolbench_tools_dir, category_name)
        API_docs = get_api_docs(category_name, category_tools_path)
        standand_API_docs = constrcut_toolbench_api_docs(API_docs)
    
        for str_api_doc, api_doc in zip(standand_API_docs, API_docs):
            tool_api_name = get_standardize_api_name(api_doc)
            toolbench_API_docs_dic[tool_api_name] = str_api_doc 
            #print(tool_api_name)
    #input()
    return toolbench_API_docs_dic


def filter_apis(ground_truth: list, search_apis: list):
    """
    - ground_truth 去重
    - search_apis 去重
    - 把在 ground_truth 中出现的元素从 search_apis 里去掉
    顺序不重要
    """
    gt_set = set(ground_truth)
    sa_set = set(search_apis)

    # 从 search_apis 中去掉在 ground_truth 里的元素
    new_sa_set = sa_set - gt_set   # 或者 sa_set.difference_update(gt_set)
    have_gt_set = gt_set & sa_set
    no_select_count = len(gt_set - have_gt_set)
    return list(new_sa_set), list(have_gt_set), no_select_count

def cal_confuse_score(ground_truth, search_apis, store):
    search_apis, have_ground_truth, no_select_count = filter_apis(ground_truth, search_apis)
    if len(search_apis) == 0:
        return {
            "max_confuse": 1.0,
            "topL_confuse": 1.0,
            "mean_confuse": 1.0,
        }
    ground_truth_api_vectors = store.get_embeddings(have_ground_truth)
    search_apis_vectors = store.get_embeddings(search_apis)
    if len(ground_truth_api_vectors) == 0:
        return {
            "max_confuse": 1.0,
            "topL_confuse": 1.0,
            "mean_confuse": 1.0,
        }
    else:
        score_items = cal_metric(ground_truth_api_vectors, search_apis_vectors, top_l=5)
        per_gold_max = score_items["per_gold_max"]
        per_gold_topL = score_items["per_gold_topL"]
        per_gold_mean = score_items["per_gold_mean"]

        miss_n = no_select_count
        
        if miss_n > 0:
            fill_value = 1.0
            per_gold_max.extend([fill_value] * miss_n)
            per_gold_topL.extend([fill_value] * miss_n)
            per_gold_mean.extend([fill_value] * miss_n)

        max_confusability = float(np.mean(per_gold_max))
        topL_confusability = float(np.mean(per_gold_topL))
        mean_confusability = float(np.mean(per_gold_mean))

    return {
        "max_confuse": max_confusability,
        "topL_confuse": topL_confusability,
        "mean_confuse": mean_confusability,
    }

def get_search_apis(search_apis_list):
    search_apis = []
    if isinstance(search_apis_list, list) and len(search_apis_list) > 0 and isinstance(search_apis_list[0], list):
        for search_apis_item in search_apis_list:
            search_apis.extend(search_apis_item)
    else:
        search_apis = list(search_apis_list)
    return search_apis

def eval_stabletoolbench_metric(file_path, toolbench_tools_dir, save_vector_file_path, groundtruth_file_path, retrieved_file_path):
   
    if os.path.exists(save_vector_file_path):
        store = EmbeddingVectorStore.load(
            embedding_name="Qwen3_Embedding",
            model_name="Qwen3_Embedding",
            model_path="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B",
            path=save_vector_file_path,
        )
        print("load data")
    else:
        toolbench_API_docs_dic = get_api_docs_from_file(toolbench_tools_dir)
        api_data_list = []
        for key, value in toolbench_API_docs_dic.items():
            api_data_list.append({"id": key, "text":value})
            #print({"id": key, "text":value})
            #input()
        store = EmbeddingVectorStore(
            embedding_name="Qwen3_Embedding", 
            model_name="Qwen3_Embedding", 
            model_path="/ossfs/workspace/hy65/dzl/model/retriever/Qwen3-Embedding-0.6B",
            dtype="float32")
        store.build_from_id_texts(api_data_list, id_key="id", text_key="text", batch_size=32)
        store.save(save_vector_file_path)
    print("Total vectors:", len(store))  # 3
    
    test_data_list = load_list_from_json(file_path)
    results = []
    
    groundtruth_dic = get_groundtruth_dic(groundtruth_file_path)
    retrieve_content_dic = get_retrieved_content_dic(retrieved_file_path)
    for data in tqdm(test_data_list):
        data_source = data['data_source']
        # if 'ground_truth' in data:
        #     ground_truth = data['ground_truth']
        # else:
        ground_truth = groundtruth_dic[data['index']]
        if 'selected_apis'  in data:
            selected_apis = data['selected_apis']
            search_apis = get_search_apis(data['search_apis'])
        elif 'generated_text' in data:
            selected_apis = parse_tool_list(data['generated_text'])
            search_apis = parse_tool_apiname_lists_from_retrieval_content(retrieve_content_dic[data['index']])
           
    
        s_f1, s_recall, s_precision = cal_f1_recall_precision(ground_truth, selected_apis)
        
        
        
        search_f1, search_recall, search_precision = cal_f1_recall_precision_from_seach_apis(ground_truth, search_apis)
        

        try:
            confuse_score_item = cal_confuse_score(ground_truth, search_apis, store)
        except Exception as e:
            print("cal_metric 出错!!!：", e)
            raise e
        #print(confuse_score)
        match = is_match(ground_truth, selected_apis)
        res = {
            'selected_f1': s_f1,
            'selected_recall': s_recall,
            'selected_precision': s_precision,
            'search_recall': search_recall,
            'search_precision': search_precision,
            'max_confuse': confuse_score_item['max_confuse'],
            'topL_confuse': confuse_score_item['topL_confuse'],
            'mean_confuse': confuse_score_item['mean_confuse'],
            'match': match
        }
        results.append(res)

    if results:
        print("start")
        avg = {key: sum(r[key] for r in results) / len(results) for key in results[0]}

        for k, v in avg.items():
            print(f"{k}: {v:.4f}")

def get_groundtruth_dic(groundtruth_file_path):
    groundtruth_dic = {}
    groundtruth_list = read_parquet_to_list(groundtruth_file_path)
    retrieve_content_dic = {}
    for data in groundtruth_list:
        groundtruth_dic[data['extra_info']['index']] = data['reward_model']['ground_truth']
    return groundtruth_dic

def get_retrieved_content_dic(retrieved_file_path):
    retrieve_content_dic = {}
    retrieved_file_list = read_parquet_to_list(retrieved_file_path)
    for data in retrieved_file_list:
         retrieve_content_dic[data['extra_info']['index']] = data['messages'][1]['content']
    return retrieve_content_dic

def main():
    parser = argparse.ArgumentParser(description='eval the results of stabletoolbench.')
    parser.add_argument('--predict_file_path', type=str, help='Path to the json file')
    parser.add_argument('--toolbench_tools_dir', type=str, default='')
    parser.add_argument('--save_vector_file_path', type=str, default='')
    parser.add_argument('--groundtruth_file_path', type=str, help='Path to the parquet file')
    parser.add_argument('--retrieved_file_path', type=str, help='Path to the parquet file')
    args = parser.parse_args()
    eval_stabletoolbench_metric(
        args.predict_file_path, 
        args.toolbench_tools_dir, 
        args.save_vector_file_path,
        args.groundtruth_file_path,
        args.retrieved_file_path
        )


   
if __name__ == '__main__':
    main()