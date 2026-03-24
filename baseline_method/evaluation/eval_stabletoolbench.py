import argparse
import pandas as pd
from utils.json_util import load_list_from_json, read_parquet_to_list
from tqdm import tqdm 
from utils.evaluation_util.metric_util import cal_f1_recall_precision, cal_f1_recall_precision_from_seach_apis, ndcg_at_k_sklearn
from utils.evaluation_util.metric_util import is_match
from utils.evaluation_util.format_util import parse_tool_list
from verl.utils.toolplan.search_process_util import parse_tools_from_retrieval_content

def get_groundtruth_dic(groundtruth_file_path):
    groundtruth_dic = {}
    groundtruth_list = read_parquet_to_list(groundtruth_file_path)
    retrieve_content_dic = {}
    for data in groundtruth_list:
        groundtruth_dic[data['extra_info']['index']] = data['reward_model']['ground_truth']
        retrieve_content_dic[data['extra_info']['index']] = data['messages'][1]['content']
    return groundtruth_dic, retrieve_content_dic

def eval_stabletoolbench_result(file_path, groundtruth_file_path, Filter, filtering_file_path, specific_domain=None, k_list = [1,5,10,20]):
    print(f"Filter:{Filter}")
    if Filter:
        print("start filter")
        test_data_list = filter_low_quality_samples(file_path, filtering_file_path)
    else:
        test_data_list = load_list_from_json(file_path)
    
    groundtruth_dic, retrieve_content_dic = get_groundtruth_dic(groundtruth_file_path)

    results = []

    for data in tqdm(test_data_list):
        data_source = data['data_source']
        if specific_domain is not None:
            if specific_domain not in data_source: continue

        ground_truth = groundtruth_dic[data['index']]
        selected_apis = parse_tool_list(data['generated_text'])
        s_f1, s_recall, s_precision = cal_f1_recall_precision(ground_truth, selected_apis)

        
        search_apis_dic_list = parse_tools_from_retrieval_content(retrieve_content_dic[data['index']])
        search_apis = []
        for search_item in search_apis_dic_list:
            search_apis.append(f"{search_item['category_name']}.{search_item['tool_name']}.{search_item['api_name']}")
    
        search_f1, search_recall, search_precision = cal_f1_recall_precision_from_seach_apis(ground_truth, search_apis)
        # if s_recall > 0:
        #     print(f"search_apis:{len(search_apis)}")
        #     print(f"search_apis:{(search_apis)}")
        #     print(f"ground_truth:{(ground_truth)}")
        #     print(f"selected_apis:{(selected_apis)}")
        #     print(search_recall)
        #     input("press enter to continue")
            
        match = is_match(ground_truth, selected_apis)
        ndcg_list = [ndcg_at_k_sklearn(ground_truth, selected_apis, k) for k in k_list]
        res = {
            'selected_f1': s_f1,
            'selected_recall': s_recall,
            'selected_precision': s_precision,
            'search_recall': search_recall,
            'search_precision': search_precision,
            'match': match
        }
        for idx, k in enumerate(k_list):
            res[f'ndcg@{k}'] = ndcg_list[idx]
        results.append(res)

    if results:
        print("start")
        avg = {key: sum(r[key] for r in results)/len(results) for key in results[0]}
        print(avg)  

def filter_low_quality_samples(file_path, filtering_file_path):
    test_data_list = load_list_from_json(file_path)
    filtering_data_list = load_list_from_json(filtering_file_path)
    available_id_dic = {}
    for data in filtering_data_list:
        if data['level'] != 0 and data['level'] != 1:
            available_id_dic[data['index']] = True
        else:
            available_id_dic[data['index']] = False
    filtering_data_list = [data for data in test_data_list if available_id_dic[data['index']]]
    print(len(filtering_data_list))
    return filtering_data_list

def main():
    parser = argparse.ArgumentParser(description='eval the results of stabletoolbench.')
    parser.add_argument('--file_path', type=str, help='Path to the json file')
    parser.add_argument('--groundtruth_file_path', type=str, help='Path to the json file')
    parser.add_argument('--Filter', type=bool, default=False, help='Wether to filter the samples of low quaility')
    parser.add_argument('--filtering_file_path', type=str, help='Path to the json file')
    args = parser.parse_args()
    eval_stabletoolbench_result(args.file_path, args.groundtruth_file_path, args.Filter, args.filtering_file_path)

if __name__ == '__main__':
    main()