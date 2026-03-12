import re

categories = [
    'Advertising', 
    'Artificial_Intelligence_Machine_Learning', 
    'Business', 
    'Business_Software', 
    'Commerce', 'Communication', 
    'Cryptography', 
    'Cybersecurity', 
    'Data', 
    'Database', 
    'Devices', 
    'Education', 
    'Email', 
    'Energy', 
    'Entertainment', 
    'Events', 
    'Finance', 
    'Financial', 
    'Food', 
    'Gaming', 
    'Health_and_Fitness', 
    'Jobs', 
    'Location', 
    'Logistics', 
    'Mapping', 
    'Media', 
    'Medical', 
    'Monitoring', 
    'Movies', 
    'Music', 
    'News_Media', 
    'Other', 
    'Payments', 
    'Reward', 
    'SMS', 
    'Science', 
    'Search', 
    'Social', 
    'Sports', 
    'Storage', 
    'Text_Analysis', 
    'Tools', 
    'Translation', 
    'Transportation', 
    'Travel', 
    'Video_Images', 
    'Visual_Recognition', 
    'Weather', 
    'eCommerce'
]

prompt_categories = [
    'Advertising', 
    'Artificial_Intelligence_Machine_Learning', 
    'Communication', 
    'Cryptography', 
    'Cybersecurity', 
    'Data', 
    'Database', 
    'Devices', 
    'Education', 
    'Email', 
    'Energy', 
    'Entertainment', 
    'Events', 
    'Food', 
    'Gaming', 
    'Health_and_Fitness', 
    'Jobs', 
    'Location', 
    'Logistics', 
    'Business_Business_Software', 
    'Video_Images_Media', 
    'eCommerce_Commerce', 
    'Finance_Financial_Payments', 
    'Mapping', 
    'Media', 
    'Medical', 
    'Monitoring', 
    'Movies', 
    'Music', 
    'Reward', 
    'SMS', 
    'Science', 
    'Search', 
    'Social', 
    'Sports', 
    'Storage', 
    'Text_Analysis', 
    'Tools', 
    'Translation', 
    'Transportation', 
    'Travel', 
    'Visual_Recognition', 
    'Weather',
] # remove 'Other'

def get_toolbench_prompt_categories():
    return prompt_categories

def get_toolbench_tool_categories():
    return categories

def get_target_category(category):
    merge_modal_list = ["News_Media", "Video_Images"]
    merge_Finance_list = ["Finance", "Financial", "Payments"]
    merge_business_list = ["Business", "Business_Software"]
    merge_commerce_list = ["eCommerce", "Commerce"]
    if category in merge_modal_list:
        return "Video_Images_Media"
    elif category in merge_Finance_list:
        return "Finance_Financial_Payments"
    elif category in merge_business_list:
        return "Business_Business_Software"
    elif category in merge_commerce_list:
        return "eCommerce_Commerce"
    else:
        return standardize_category(category)
    

def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name

# without using this function for procssing data
def prepare_tool_info(info):
    category = info['category']
    standard_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in standard_category or "," in standard_category:
        standard_category = standard_category.replace(" ", "_").replace(",", "_")
    standard_category = standard_category.replace("__", "_")
    
    tool_name = info['tool_name']
    api_name = change_name(standardize(info['api_name'])).split(f"_for_{tool_name}")[0]
    if not tool_name.endswith(f"_for_{standard_category}"):
        tool_name = standardize(info['tool_name'])
        tool_name += f"_for_{standard_category}"
    else:
        tmp_tool_name = standardize(tool_name.replace(f"_for_{standard_category}", ""))
       
    return tool_name, standard_category, api_name