import argparse
import pandas as pd
from utils.json_util import load_list_from_json,read_parquet_to_list
from tqdm import tqdm 
from utils.evaluation_util.metric_util import cal_f1_recall_precision, cal_f1_recall_precision_from_seach_apis, ndcg_at_k_sklearn
from utils.evaluation_util.metric_util import is_match
import pandas as pd


def eval_stabletoolbench_turns(file_path, ground_truth_dic, specific_domain=None, k_list = [1,5,10,20]):
    test_data_list = load_list_from_json(file_path)
    results = []
    max_turns = 0
    for data in tqdm(test_data_list):
        data_source = data['data_source']
        if specific_domain is not None:
            if specific_domain not in data_source: continue

        ground_truth = ground_truth_dic[data['index']]
        
        selected_apis = data['selected_apis']
        s_f1, s_recall, s_precision = cal_f1_recall_precision(ground_truth, selected_apis)
        
        if 'search_apis' in data:
            max_turns = max(max_turns, len(data['search_apis']))
            search_apis = data['search_apis']
            _, search_recall, _ = cal_f1_recall_precision_from_seach_apis(ground_truth, search_apis)
        else:
            search_recall = 0.0
            
        match = is_match(ground_truth, selected_apis)
        ndcg_list = [ndcg_at_k_sklearn(ground_truth, selected_apis, k) for k in k_list]
        res = {
            'selected_f1': s_f1,
            'selected_recall': s_recall,
            'selected_precision': s_precision,
            'search_recall': search_recall,
            'match': match
        }
        for idx, k in enumerate(k_list):
            res[f'ndcg@{k}'] = ndcg_list[idx]
        results.append(res)

    if results:
        print("start")
        avg = {key: sum(r[key] for r in results)/len(results) for key in results[0]}
        print(avg)  
        print(f"max_turns:{max_turns}")

    #print(f"all_s_f1: {all_s_f1/len(test_data_list)}, all_s_recall: {all_s_recall/len(test_data_list)}, all_s_precision: {all_s_precision/len(test_data_list)}")
    #print(f"all_search_recall: {all_search_recall/len(test_data_list)}")

def get_ground_truth(file_path):
    # 读取 parquet 文件为 DataFrame
    data_list = read_parquet_to_list(file_path)
    # 查看前几行内容
    ground_truth_dic = {}
    for data in data_list:
        #print(data)
        index = data['extra_info']['index']
        ground_truth = data['reward_model']['ground_truth']
        #print(data['extra_info']['question'])
        ground_truth_dic[index] = ground_truth
    return ground_truth_dic

def main():
    parser = argparse.ArgumentParser(description='eval the results of stabletoolbench.')
    parser.add_argument('--file_path', type=str, help='Path to the json file')
    args = parser.parse_args()
    ground_truth_dic = get_ground_truth("/ossfs/workspace/hy65/dzl/code/toolPlaner/data/stabletoolbench_dataset/tool_selection.parquet")
    eval_stabletoolbench_turns(args.file_path, ground_truth_dic)

if __name__ == '__main__':
    main()