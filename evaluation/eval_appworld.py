import argparse
import pandas as pd
from utils.json_util import load_list_from_json 
from tqdm import tqdm 
from utils.evaluation_util.metric_util import cal_f1_recall_precision, cal_f1_recall_precision_from_seach_apis
from utils.evaluation_util.metric_util import is_match

from utils.json_util import read_parquet_to_list
from utils.evaluation_util.format_util import parse_tool_list
from utils.evaluation_util.format_util import parse_tool_apiname_lists_from_retrieval_content


def get_core_apis(selected_apis: list):
    core_apis = []
    stable_apis = []
    for api in selected_apis:
        if 'login' in api or 'supervisor' in api:
            stable_apis.append(api)
            continue
        core_apis.append(api)
    return core_apis, stable_apis

def get_overall_apis(selected_apis):
    overall_apis = []
    app_name_set = set()
    for api in selected_apis:
        app_name = api.split('.', 1)[0]
        if app_name != "supervisor":
            app_name_set.add(app_name)

    for app_name in app_name_set:
        overall_apis.append(f"{app_name}.login")
    overall_apis.extend(selected_apis)
    overall_apis.extend(["supervisor.complete_task","supervisor.show_account_passwords","supervisor.show_profile"])
    overall_apis = list(set(overall_apis))

    return overall_apis

def filter_core_apis(selected_apis):
    overall_apis = []

    for api in selected_apis:
        app_name = api.split('.', 1)[0]
        if app_name == "supervisor" or app_name =="docs":
            continue
        elif 'login' in api:
            continue
        else:
            overall_apis.append(api)

    return overall_apis

def get_search_apis(search_apis_list):
    search_apis = []
    if isinstance(search_apis_list, list) and len(search_apis_list) > 0 and isinstance(search_apis_list[0], list):
        for search_apis_item in search_apis_list:
            search_apis.extend(search_apis_item)
    else:
        search_apis = list(search_apis_list)
    final_search_apis = []
    for api in search_apis:
        api_name = api.split('.', 1)[-1]
        final_search_apis.append(api_name)
    return final_search_apis

def is_set_match(ground_truth, predict_apis):
    core_predict_apis = set(ground_truth) & set(predict_apis) 
    if core_predict_apis == set(ground_truth):
        return True
    return False

def eval_appworld_metric(file_path, groundtruth_file_path, retrieved_file_path):
   
    test_data_list = load_list_from_json(file_path)
    results = []
    
    groundtruth_dic = get_groundtruth_dic(groundtruth_file_path)
    if retrieved_file_path !='':
        retrieve_content_dic = get_retrieved_content_dic(retrieved_file_path)
    else:
        retrieve_content_dic = {}
        
    for data in tqdm(test_data_list):
        #print(data)
        #input()
        data_source = data['data_source']
        ground_truth = groundtruth_dic[data['index']]['selected_apis']
        core_ground_truth = groundtruth_dic[data['index']]['core_apis']
        if 'selected_apis'  in data:
            selected_apis = data['selected_apis']
            if "search_apis" in data:
                search_apis = get_search_apis(data['search_apis'])
                search_apis = list(set(search_apis))
                
        elif 'generated_text' in data:
            selected_apis = parse_tool_list(data['generated_text'])
            if retrieved_file_path !='':
                search_apis = parse_tool_apiname_lists_from_retrieval_content(retrieve_content_dic[data['index']])
                search_apis = list(set(search_apis))
            elif "search_apis" in data:
                search_apis = get_search_apis(data['search_apis'])
                search_apis = list(set(search_apis))

        #1. caluate the rate of core apis
        cs_f1, cs_recall, cs_precision = cal_f1_recall_precision(core_ground_truth, selected_apis)
        #if "search_apis" in data:
        if len(search_apis) == 0:
            search_apis = []
        elif len(search_apis[0].split('.')) == 3:
            search_apis = [api.split('.', 1)[-1] for api in search_apis]
        
        search_f1, search_recall, search_precision = cal_f1_recall_precision_from_seach_apis(core_ground_truth, search_apis)
        #else:
        #    search_f1, search_recall, search_precision = 0.0,0.0,0.0
        core_selected_apis = filter_core_apis(selected_apis)
        core_match = is_set_match(core_ground_truth, core_selected_apis)
        # print(core_ground_truth)
        # print(core_selected_apis)
        # print(selected_apis)
        # input()
        #2. calculate the rate based on selected appname (add login)
        _selected_apis = get_overall_apis(selected_apis)
        s_f1, s_recall, s_precision = cal_f1_recall_precision(ground_truth, _selected_apis)
        match = is_match(ground_truth, _selected_apis)
        #print(ground_truth)
        #print(selected_apis)
        #print(_selected_apis)
        #input()
        
        res = {
            'core_selected_f1': cs_f1,
            'core_selected_recall': cs_recall,
            'core_selected_precision': cs_precision,
            'selected_f1': s_f1,
            'selected_recall': s_recall,
            'selected_precision': s_precision,
            'search_recall': search_recall,
            'search_precision': search_precision,
            'match': match,
            'core_match': core_match
        }
        results.append(res)

    if results:
        print("start")
        avg = {key: sum(r[key] for r in results) / len(results) for key in results[0]}

        for k, v in avg.items():
            print(f"{k}: {v:.4f}")

def get_groundtruth_dic(groundtruth_file_path):
    groundtruth_dic = {}
    groundtruth_list = load_list_from_json(groundtruth_file_path)
    for data in groundtruth_list:
        #input(data)
        groundtruth_dic[data['index']] = {
            'selected_apis': data['required_apis'],
            'core_apis': data['core_apis'],
        }
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
    parser.add_argument('--groundtruth_file_path', type=str, help='Path to the parquet file')
    parser.add_argument('--retrieved_file_path', type=str, default="", help='Path to the parquet file')
    args = parser.parse_args()
    eval_appworld_metric(
        args.predict_file_path, 
        args.groundtruth_file_path,
        args.retrieved_file_path
        )


   
if __name__ == '__main__':
    main()