import re
from verl.utils.toolplan.show_message import _structure_single_dialogue

class search_process:
    def __init__(self, tool_calls_arguments: dict, retrieval_api_names: list[dict]):
        self.tool_calls_arguments = tool_calls_arguments
        self.retrieval_api_names = retrieval_api_names


def parse_tools_from_retrieval_content(retrieval_content):
    """
    从 retrieval_content 中提取每段的 category_name, tool_name, api_name。
    
    参数
    ----
    retrieval_content : str 或 dict
        - 如果是 dict，假定有键 "result"，内容为多行文本；
        - 如果是 str，直接视为多行文本。
    
    返回
    ----
    List[Dict[str, str]]，每个元素形如：
        {
            "category_name": "...",
            "tool_name": "...",
            "api_name": "..."
        }
    """
    # 1. 取出原始文本
    if isinstance(retrieval_content, dict):
        text = retrieval_content.get("result", "")
    else:
        text = retrieval_content
    
    text = text.replace("\\n", "\n")

    # 2. 按 doc 分段：doc 1:, doc 2:, ...（仅用于切块，不保留 doc_id）
    doc_pattern = re.compile(r"doc\s+\d+:\s*(.*?)(?=doc\s+\d+:|$)", re.DOTALL | re.IGNORECASE)
    
    results: List[Dict[str, str]] = []
    
    for match in doc_pattern.finditer(text):
        doc_block = match.group(1)
        # 3. 抽取 category_name/tool_name/api_name
        cat_match = re.search(r"category_name:\s*([^\n\r]+)", doc_block, re.IGNORECASE)
        tool_match = re.search(r"tool_name:\s*([^\n\r]+)", doc_block, re.IGNORECASE)
        api_match = re.search(r"api_name:\s*([^\n\r]+)", doc_block, re.IGNORECASE)
        
        category_name = cat_match.group(1).strip() if cat_match else ""
        tool_name = tool_match.group(1).strip() if tool_match else ""
        api_name = api_match.group(1).strip() if api_match else ""
       
        if category_name and tool_name and api_name:
            results.append(
                {
                    "category_name": category_name,
                    "tool_name": tool_name,
                    "api_name": api_name,
                }
            )
    return results



def print_single_data(single_data):
    dialogue_view = _structure_single_dialogue(single_data)
    for turn_view in dialogue_view.turns:
        print(f"turn_view.role: {turn_view.role}")
        print(f"turn_view.content: {turn_view.content}")
        if turn_view.tool_calls:
            for tool_call in turn_view.tool_calls:
                print(  f"turn_view.tool_calls.name: {tool_call.name}")
                print(f"turn_view.tool_calls.arguments: {tool_call.arguments}")
        print("==============")   
    print(len(dialogue_view.turns))
# main fuction
if __name__ == "__main__":
   tool_response_str='''
    {"result": "doc 1:\ncategory_name: Data\ntool_name: fake_users\napi_name: get_user_by_gender\napi_description: '[tool]:fake users is a Api that give you fake users [api]:get user by gender'\nrequired_parameters:\n- name: gender\n  type: STRING\n  description: ''\n  default: male\ntemplate_response:\n  results:\n  - gender: str\n    name:\n      title: str\n      first: str\n      last: str\n    location:\n      street:\n        number: int\n        name: str\n      city: str\n      state: str\n      country: str\n      postcode: int\n      coordinates:\n        latitude: str\n        longitude: str\n      timezone:\n        offset: str\n        description: str\n    email: str\n    login:\n      uuid: str\n      username: str\n      password: str\n      salt: str\n      md5: str\n      sha1: str\n      sha256: str\n    dob:\n      date: str\n      age: int\n    registered:\n      date: str\n      age: int\n    phone: str\n    cell: str\n    id:\n      name: str\n      value: str\n    picture:\n      large: str\n      medium: str\n      thumbnail: str\n    nat: str\n    _list_length: 1\n  info:\n    seed: str\n    results: int\n   \ndoc 2:\ncategory_name: Data\ntool_name: fake_users\napi_name: user\napi_description: '[tool]:fake users is a Api that give you fake users [api]:get one user'\ntemplate_response:\n  results:\n  - gender: str\n    name:\n      title: str\n      first: str\n      last: str\n    location:\n      street:\n        number: int\n        name: str\n      city: str\n      state: str\n      country: str\n      postcode: int\n      coordinates:\n        latitude: str\n        longitude: str\n      timezone:\n        offset: str\n        description: str\n    email: str\n    login:\n      uuid: str\n      username: str\n      password: str\n      salt: str\n      md5: str\n      sha1: str\n      sha256: str\n    dob:\n      date: str\n      age: int\n    registered:\n      date: str\n      age: int\n    phone: str\n    cell: str\n    id:\n      name: str\n      value: NoneType\n    picture:\n      large: str\n      medium: str\n      thumbnail: str\n    nat: str\n    _list_length: 1\n  info:\n    seed: str\n    results: int\n    page: int\n    version: str\n\ndoc 3:\ncategory_name: Data\ntool_name: seeding_data\napi_name: users\napi_description: '[tool]:Completely APIs that helps web developers and web designers generate fake data in a fast and easy way. [api]:Get 100 users with en_US locale and gender male'\nrequired_parameters:\n- name: _quantity\n  type: NUMBER\n  description: ''\n  default: 100\n- name: _gender\n  type: STRING\n  description: ''\n  default: male\nmethod:\n  type: object\n  properties:\n    get:\n     ...
   '''
   print(parse_tools_from_retrieval_content(tool_response_str))
