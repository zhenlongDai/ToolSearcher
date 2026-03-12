import json
from utils.toolbench_util.format_util import standardize_category, standardize
train = "/ossfs/workspace/hy65/dzl/data/tool_data/ToolBench_data/data/retrieval/G1/train.json"
from utils.json_util import load_list_from_json,read_parquet_to_list
from tqdm import tqdm 

def test_same_tool_name():
  # 读取整个 json 文件
  with open(train, 'r', encoding='utf-8') as f:
    data = json.load(f)

  tool_name_set = set()
  # 打印前几条
  for i, item in enumerate(data):
    for api_content in item['api_list']:
      tool_name = standardize_category(api_content['category_name']) + "."+ standardize(api_content['tool_name'])#  + "."+ standardize(api_content['api_name']) 
      tool_name_set.add(tool_name)

  # 如需查看数据规模
  print("总样本数:", len(data))

  data_list = read_parquet_to_list("./data/stabletoolbench_dataset/tool_selection.parquet")
  #print(len(data_list))
  data_result_list = []
  same_count = 0
  for data in tqdm(data_list):
      ground_truth =  data['reward_model']['ground_truth']
      flag = 0
      for api_name in ground_truth:
        new_api_name = api_name.rsplit('.', 1)[0]
        if new_api_name in tool_name_set:
          flag = 1
      same_count += flag
     
  print("here")
  print(same_count)
  print("here")
  
if __name__ == "__main__":
  test_same_tool_name()