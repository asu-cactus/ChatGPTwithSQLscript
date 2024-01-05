import os
import json

def get_test_info(json_file_path, len_id_target_id):
    # Read the JSON file once
    with open(json_file_path, 'r') as file:
        data_list = json.load(file)
        # Create a dictionary for faster lookups
        data_dict = {item["Source Data Name"]: item for item in data_list}

    # Constructing the path to the specific subfolder
    sub_folder_name = f"length{len_id_target_id}"
    main_folder_name = os.path.abspath("github-pipelines")
    sub_folder_path = os.path.join(main_folder_name, sub_folder_name)

    # Counting files starting with 'test' in this subfolder
    file_count = sum(1 for _, _, files in os.walk(sub_folder_path) for file in files if file.startswith('test'))

    # Find and store the required data
    source_data_name_list = []
    source_data_schema_list = []
    source_samples_list = []
    target_data_name, target_data_schema, target_samples = None, None, None

    for i in range(file_count):
        source_data_name_to_find = f"Source{len_id_target_id}_{i}"
        data = data_dict.get(source_data_name_to_find)

        if data:
            # Extract the relevant information from the JSON data
            if target_data_name is None:  # Assuming all target data names and schemas are the same
                target_data_name = data["Target Data Name"]
                target_data_schema = data["Target Data Schema"]
                target_samples = data["Target Data Sample"]

            source_data_name_list.append(data["Source Data Name"])
            source_data_schema_list.append(data["Source Data Schema"])
            source_samples_list.append(data["3 Samples of Source Data"])

    return target_data_name, target_data_schema, target_samples, file_count, source_data_name_list, source_data_schema_list, source_samples_list

import json

def get_test_cases_ids(json_file_path, len_id, max_len_id, target_id, max_target_id):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data_list = json.load(file)

    # Find the item with the specified Source Data Name
    ids = [item["Target Data Name"] for item in data_list]

    # Adjust the filter criteria
    filtered_ids = []
    for id in ids:
        parts = id[6:].split('_')
        if len(parts) >= 2:
            len_part = int(parts[0])
            target_part = int(parts[1])
            if len_id <= len_part <= max_len_id and target_id <= target_part <= max_target_id:
                filtered_ids.append(id)

    #bad_ids = ['Target1_16']
    past_at_least_once = ['Target1_0', 'Target1_3', 'Target1_4', 'Target1_6', 'Target1_11', 'Target1_20',
                          'Target1_21', 'Target1_29', 'Target1_34', 'Target1_35', 'Target1_38',
                          'Target1_55', 'Target1_60', 'Target1_65', 'Target1_67', 'Target1_71',
                          'Target1_72', 'Target1_80', 'Target1_84']
    finish_issue = ['Target1_27', 'Target1_46', 'Target1_97'] # 27, 46, 97
    further_finish_issue = ['Target1_97']
    with_dirty_rows = ['Target1_40', 'Target1_59']
    with_similarity_issue = ['Target1_10']#, 'Target1_23', 'Target1_62', 'Target1_86', 'Target1_97']
    with_syntax_issue = ['Target1_44', 'Target1_78', 'Target1_88', 'Target1_89', 'Target1_90', 'Target1_91', 'Target1_92', 'Target1_93']

    print('Total number of test cases:', len(filtered_ids))
    print('Number of test cases that have been tested at least once:', len(past_at_least_once))
    #print('Number of bad test cases:', len(bad_ids))
    #filtered_ids = [id for id in filtered_ids if id not in bad_ids]
    filtered_ids = [id for id in filtered_ids if id not in past_at_least_once]
    print('Number of Test cases after filtering:', len(filtered_ids))
    return ['Target1_99']
