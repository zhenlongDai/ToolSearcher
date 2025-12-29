from utils.file_util import get_all_folder_name, get_all_json_file_name

def test_get_all_folder_name(path):
    folder_name_list = get_all_folder_name(path)
    print(folder_name_list)
    print(len(folder_name_list))
   #['Advertising', 'Artificial_Intelligence_Machine_Learning', 'Business', 'Business_Software', 'Commerce', 'Communication', 'Cryptography', 'Customized', 'Cybersecurity', 'Data', 'Database', 'Devices', 'Education', 'Email', 'Energy', 'Entertainment', 'Events', 'Finance', 'Financial', 'Food', 'Gaming', 'Health_and_Fitness', 'Jobs', 'Location', 'Logistics', 'Mapping', 'Media', 'Medical', 'Monitoring', 'Movies', 'Music', 'News_Media', 'Other', 'Payments', 'Reward', 'SMS', 'Science', 'Search', 'Social', 'Sports', 'Storage', 'Text_Analysis', 'Tools', 'Translation', 'Transportation', 'Travel', 'Video_Images', 'Visual_Recognition', 'Weather', 'eCommerce']
   # 50

def test_get_all_json_file_name(path):
    folder_name_list = get_all_json_file_name(path)
    print(folder_name_list)
    print(len(folder_name_list))
if __name__ == '__main__':

    #test_get_all_folder_name('/ossfs/workspace/hy65/dzl/code/toolwork/StableToolBench/tools')
    
    test_get_all_json_file_name('/ossfs/workspace/hy65/dzl/code/toolwork/StableToolBench/tools/Advertising')