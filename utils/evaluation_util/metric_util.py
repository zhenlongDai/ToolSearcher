from sklearn.metrics import ndcg_score

def is_match(ground_list, predict_list):
    return set(ground_list) == set(predict_list)

def cal_f1_recall_precision_from_seach_apis(ground_list, search_apis_list):
    predict_list = []
    if isinstance(search_apis_list, list) and len(search_apis_list) > 0 and isinstance(search_apis_list[0], list):
        for search_apis in search_apis_list:
            predict_list.extend(search_apis)
    else:
        predict_list = list(search_apis_list)
        
    return cal_f1_recall_precision(ground_list, predict_list)
    
def cal_f1_recall_precision(ground_list, predict_list):
    ground_set = set(ground_list)
    predict_set = set(predict_list)
    true_positive = len(ground_set & predict_set)
    precision = true_positive / len(predict_set) if predict_set else 0
    recall = true_positive / len(ground_set) if ground_set else 0
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1, recall, precision



# def ndcg_at_k_sklearn(ground_list, predict_list, k):
#     # 如果长度不足k，补足到k
#     fixed_predict_list = predict_list[:k] + [f'pad_{i}' for i in range(k - len(predict_list))]
#     y_true = [[1 if item in ground_list else 0 for item in fixed_predict_list]]
#     y_score = [[1.0 for _ in fixed_predict_list]]  # 这里可保留同分
#     return ndcg_score(y_true, y_score, k=k)


def ndcg_at_k_sklearn(ground_list, predict_list, k):
    if k == 1:
        if len(predict_list) == 0: return 0.0
        
        fixed_predict_list = predict_list[:k]
        y_true = [[1 if item in ground_list else 0 for item in fixed_predict_list]]
        y_score = [[1.0 for _ in fixed_predict_list]]
        if y_true[0][0] == 0: return 0.0
        else: return 1.0
    else:
        fixed_predict_list = predict_list[:k] + [f'pad_{i}' for i in range(k - len(predict_list[:k]))]
        y_true = [[1 if item in ground_list else 0 for item in fixed_predict_list]]
        y_score = [[1.0 for _ in fixed_predict_list]]
    try:
        return ndcg_score(y_true, y_score, k=k)
    except Exception as e:
        print("\nNDCG计算异常！详细信息如下：")
        print(f"ground_list: {ground_list}")
        print(f"predict_list: {predict_list}")
        print(f"fixed_predict_list: {fixed_predict_list}")
        print(f"y_true: {y_true}")
        print(f"y_score: {y_score}")
        print(f"k: {k}")
        print(f"异常: {e}\n")
        return None  # 或 raise
if __name__ == "__main__":
    print(ndcg_at_k_sklearn(['a','b','c'], ['a'], 1))