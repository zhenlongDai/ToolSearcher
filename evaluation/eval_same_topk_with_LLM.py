import argparse
import pandas as pd
from utils.json_util import load_list_from_json,read_parquet_to_list
from tqdm import tqdm 
from utils.evaluation_util.metric_util import cal_f1_recall_precision, cal_f1_recall_precision_from_seach_apis, ndcg_at_k_sklearn
from utils.evaluation_util.metric_util import is_match
import pandas as pd


def cal_results(test_data_list, ground_truth_dic, specific_domain, k_list, flag = False):
    results = []

    for data in tqdm(test_data_list):
        data_source = data['data_source']
        if specific_domain is not None:
            if specific_domain not in data_source: continue

        ground_truth = ground_truth_dic[data['index']]
        
        selected_apis = data['selected_apis']
        s_f1, s_recall, s_precision = cal_f1_recall_precision(ground_truth, selected_apis)
        
        if 'search_apis' in data:
            search_apis = data['search_apis']
            search_F1, search_recall, search_precision = cal_f1_recall_precision_from_seach_apis(ground_truth, search_apis)
        else:
            search_recall = s_recall
            search_precision = s_precision
            
        match = is_match(ground_truth, selected_apis)
        # if match == 1 and flag:
        #     print(data['index'])
        #     print(ground_truth)
        #     print(selected_apis)
        #     input()
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

#{'G1_category', 'G1_tool', 'G2_category', 'G1_instruction', 'G2_instruction', 'G3_instruction'}
def eval_stabletoolbench_result(LLM_file_path, retriever_file_path, ground_truth_dic, specific_domain=None, k_list = [1,5,10,20]):
    print(f"specific_domain: {specific_domain}")
    test_data_list = load_list_from_json(LLM_file_path)
    retriever_data_list = load_list_from_json(retriever_file_path)
    retrieval_topk_dic = {}
    data_source_set = set()
    for test_data in test_data_list:
        data_source_set.add(test_data['data_source'])
        index = test_data['index']
        search_apis = []
        for search_api_list in test_data['search_apis']:
            search_apis.extend(search_api_list)
        retrieval_topk_dic[index] = len(search_apis)
    # print(data_source_set)
    # input()
    print("LLM>>>>")
    cal_results(test_data_list, ground_truth_dic, specific_domain, k_list)
    
    for retriever_data in retriever_data_list:
        index = retriever_data['index']
        topk = retrieval_topk_dic[index]
        retriever_data['selected_apis'] = retriever_data['selected_apis'][:topk]
    
    print("retriever>>>>")
    cal_results(retriever_data_list, ground_truth_dic, specific_domain, k_list, True)
   

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
    parser.add_argument('--LLM_file_path', type=str, default="/ossfs/workspace/hy65/dzl/code/toolPlaner/experiment_results/multiturn_search_wrong/baseline_turns_15.json", help='Path to the json file')
    parser.add_argument('--retriever_file_path', type=str, default="/ossfs/workspace/hy65/dzl/code/toolPlaner/experiment_results/baseline/Qwen3-Embedding-0.6B_top80.json", help='Path to the json file')
    args = parser.parse_args()
    ground_truth_dic = get_ground_truth("/ossfs/workspace/hy65/dzl/code/toolPlaner/data/stabletoolbench_dataset/tool_selection.parquet")
    eval_stabletoolbench_result(args.LLM_file_path, args.retriever_file_path, ground_truth_dic)

if __name__ == '__main__':
    main()