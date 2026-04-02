from utils.json_util import dump_yaml


def constrcut_appworld_api_doc(ori_api_doc):
    api_doc = ori_api_doc.copy()
    new_api_doc = {}

    category_name = api_doc.pop('category_name')
    tool_name = api_doc.pop('app_name')
    api_name = api_doc.pop('api_name')

    new_api_doc['category_name'] = category_name
    new_api_doc['tool_name'] = tool_name
    new_api_doc['api_name'] = api_name
    new_api_doc['api_name'] = api_name
    for key,value in api_doc.items():
        new_api_doc[key] = value

    new_api_doc = dump_yaml(new_api_doc)
    # print(api_doc)
    # print(repr(new_api_doc))
    # input("press")
    return new_api_doc  