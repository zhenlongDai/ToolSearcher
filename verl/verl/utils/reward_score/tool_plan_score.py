import re
import random

def get_tool_list_str_and_format_score(response_str):
    answer_pattern = r"<tool_list>(.*?)</tool_list>"
    match = re.finditer(answer_pattern, response_str, re.DOTALL)
    matches = list(match)
    format_score = 0.0
    tool_list_str = None
    # If there are 0  matches, return None
    if len(matches) < 1:
        tool_list_str = None 
    elif len(matches) == 1:
        format_score = 0.1
        tool_list_str = matches[-1].group(1).strip()
    else:
        format_score = 0.0
        tool_list_str = matches[-1].group(1).strip()
    # If there are 2 or more matches, return the last one
    return tool_list_str, format_score

def parse_tool_list(tool_list_str: str):
  if tool_list_str == None:
    return []
  
  return [
      item.strip()
      for item in tool_list_str.split(",")
      if item.strip()
  ]

def compute_id_match(pred, target):
    pred = set(pred)
    gold = set(target)
    tp = len(pred & gold)      # 交集
    fp = len(pred - gold)      # 预测有但 gold 没有
    fn = len(gold - pred)      # gold 有但预测没有
    return tp, fp, fn

def compute_F1_score(pred, target):
  tp, fp, fn = compute_id_match(pred, target)
  precision = tp / (tp + fp) if tp + fp > 0 else 0.0
  recall    = tp / (tp + fn) if tp + fn > 0 else 0.0
  f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
#   print("----------")
#   print(precision)
#   print(recall)
#   print(f1)
#   print("----------")
  return f1

def compute_tool_list_score(solution_str, ground_truth):
  """Extract the equation from the solution string."""
  # Remove everything before the first "Assistant:"
  # if "Assistant:" in solution_str:
  #     solution_str = solution_str.split("Assistant:", 1)[1]
  # elif "<|im_start|>assistant" in solution_str:
  #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
  # else:
  #     return None
  # solution_str = solution_str.split('\n')[-1]
  tool_list_str, format_score =  get_tool_list_str_and_format_score(solution_str)
  tool_list = parse_tool_list(tool_list_str)
  tool_list_F1_score = compute_F1_score(tool_list, ground_truth)
  return tool_list_F1_score + format_score, tool_list
    


def compute_score(solution_str, ground_truth :list[str], apis_in_search_process:set):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
    """
    score, answer = compute_tool_list_score(solution_str=solution_str, ground_truth = ground_truth)
    #caluate the ratio of answer in apis_in_search_process
    # apis_in_search_process mybe is empty
    if len(answer) > 0 and len(apis_in_search_process) > 0:
        selection_from_search_ratio = len(set(answer) & apis_in_search_process) / len(apis_in_search_process)
    else:
        selection_from_search_ratio = 0.0

    if len(answer) > 0:
        selection_from_gt_ratio = len(set(answer) & set(ground_truth)) / len(ground_truth)
    else:
        selection_from_gt_ratio = 0.0   

    #do_print = random.randint(1, 64) == 1

    # #if do_print:
    # print("--------------------------------")
    # print(f"Golden answers: {ground_truth}")
    # if answer is not None:
    #     print(f"Extracted answer is not None: {answer}")
    # else:
    #     print("Extracted answer: None!")
    # print(f"Solution string: {solution_str}")
    # print(f"score: {score}")

    return { "tool_selection_score": score, 
             "selection_from_search_ratio": selection_from_search_ratio, 
             "selection_from_gt_ratio": selection_from_gt_ratio
            }