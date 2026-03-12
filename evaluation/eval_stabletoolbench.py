import argparse
import pandas as pd
from utils.json_util import load_list_from_json 
from tqdm import tqdm 
from utils.evaluation_util.metric_util import cal_f1_recall_precision, cal_f1_recall_precision_from_seach_apis, ndcg_at_k_sklearn
from utils.evaluation_util.metric_util import is_match


def eval_stabletoolbench_result(file_path, specific_domain=None, k_list = [1,5,10,20]):
    test_data_list = load_list_from_json(file_path)
    results = []

    for data in tqdm(test_data_list):
        data_source = data['data_source']
        if specific_domain is not None:
            if specific_domain not in data_source: continue

        ground_truth = data['ground_truth']
        selected_apis = data['selected_apis']
        s_f1, s_recall, s_precision = cal_f1_recall_precision(ground_truth, selected_apis)

        if 'search_apis' in data:
            search_apis = data['search_apis']
            search_f1, search_recall, search_precision = cal_f1_recall_precision_from_seach_apis(ground_truth, search_apis)
        else:
            search_recall = s_recall
            search_precision = s_precision
        
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



def main():
    parser = argparse.ArgumentParser(description='eval the results of stabletoolbench.')
    parser.add_argument('--file_path', type=str, help='Path to the json file')
    args = parser.parse_args()
    eval_stabletoolbench_result(args.file_path)

if __name__ == '__main__':
    main()