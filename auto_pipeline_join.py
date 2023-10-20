import os
import pandas as pd
import json
from datetime import datetime
from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed)
from join_util import convert_target_names,access_auto_pipeline_dataset,read_csv_target
from gpt import chat_with_gpt
from join import validation
import logging
logging.basicConfig(filename='auto_pipeline_join.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_datetime(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError("Type not serializable")

def convert_excel_to_json(excel_file_path, json_file_path):
    # Read the Excel file
    xls = pd.ExcelFile(excel_file_path)

    # Specify the columns to include in the JSON file
    columns_to_include = [
        "Folder Name",
        "Target Data Name",
        "Target Data Schema",
        "Source Data Name",
        "Source Data Schema",
        "Target Data Description"]

    # Read the specified columns from the first sheet
    data_to_convert = pd.read_excel(xls, sheet_name='Sheet2', usecols=columns_to_include)

    # Fill missing values in the specified columns by forward filling
    columns_to_fill = ["Target Data Name"]
    data_to_convert[columns_to_fill] = data_to_convert[columns_to_fill].ffill()

    # Create a dictionary to store the groupings
    json_data_grouped = {}

    # Iterate through each row and append to the corresponding list in the dictionary
    for _, row in data_to_convert.iterrows():
        target_data_name = row["Target Data Name"]
        row_dict = row.to_dict()

        if target_data_name not in json_data_grouped:
            json_data_grouped[target_data_name] = []

        json_data_grouped[target_data_name].append(row_dict)

    # Open the JSON file for writing
    with open(json_file_path, 'w') as json_file:
        json_file.write(json.dumps(json_data_grouped, indent=4, default=convert_datetime))

    print(f"JSON file has been saved to {json_file_path}")

def create_sample_i(samples):
    sample_i = {f"sample_{i}": sample for i, sample in enumerate(samples)}
    return sample_i

def generate_prompt_auto_pipeline(no_of_source_tables,source_names,target_name,source_data_schema,
                                  target_data_schema,target_data_description,test_0_path,test_1_path,
                                  sub_folder,template_option):
    target_name = target_name[0] 
    no_of_source_tables = no_of_source_tables
    source_data_schema = source_data_schema
    target_data_schema = target_data_schema[0]
    target_data_description = target_data_description[0]
    if no_of_source_tables == 1:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create the source table with following {source_names} with only the given attributes: {source_data_schema}. 
        Please delete the table before creating it if the table exists. 
        Second, the table should have data from csv files at given path {test_0_path}.
        Also treat empty value as NULL and You can use the HEADER option in the COPY command to skip the first row (which contains column names)
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the first table exists.
        Finally, insert all rows from the source table into only one {target_name}, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Hint-1: {target_data_description}
        Also COPY the SQL result into a {sub_folder}{target_name}_result.csv" file.
        Please don't remove the any table, because we need it for validation.
        Please quote the returned SQL script between "```sql\n" and "\n```".
        """
    elif no_of_source_tables > 1:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create the {no_of_source_tables} tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the first table exists. 
        Second, these {no_of_source_tables} tables should have data from csv files at given path {test_0_path} and {test_1_path} respectively.
        Also treat empty value as NULL and You can use the HEADER option in the COPY command to skip the first row (which contains column names)
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the first table exists.
        Finally, insert all rows from the {no_of_source_tables} tables into only one {target_name}, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Hint-1: {target_data_description}
        Also COPY the SQL result into a {sub_folder}{target_name}_result.csv" file.
        Please don't remove the any table, because we need it for validation.
        Please quote the returned SQL script between "```sql\n" and "\n```".
        """
    else:
        print("choose different template option")
    return prompt

def gpt_auto_pipeline(json_file_path, target_data_name_to_find):
    with open(json_file_path, 'r') as file:
        data_list = json.load(file)
    target_data_names = []
    source_data_names = []
    source_data_schema = []
    target_data_schema = []
    target_data_description = []
    for target_key, target_values in data_list.items():
        for target_value in target_values:
            if target_value.get("Target Data Name") == target_data_name_to_find:
                target_data_names.append(target_value.get("Target Data Name"))
                source_data_names.append(target_value.get("Source Data Name"))
                source_data_schema.append(target_value.get("Source Data Schema"))
                target_data_schema.append(target_value.get("Target Data Schema"))
                target_data_description.append(target_value.get("Target Data Description"))
    return target_data_names, source_data_names,source_data_schema, target_data_schema, target_data_description

def main(*args):
    (json_file_path, template_option, target_id, max_target_id,length_id) = args
    conn = create_connection()

    while target_id <= max_target_id:
        target_data_name_to_find = "Target" + str(length_id) + "_" + str(target_id)
        # Get JSON data for prompt
        target_data_names, source_data_names,source_data_schema, target_data_schema, target_data_description = gpt_auto_pipeline(json_file_path,target_data_name_to_find)
        find_target_name_folder = convert_target_names(target_data_names[0])
        main_folder_name,sub_folder, test_0_path, test_1_path, target_path = access_auto_pipeline_dataset(find_target_name_folder) 
        logging.info(f"target_data_name, Source_data_names: {target_data_names[0]}, {source_data_names}")
        logging.info(f"number of sources: {len(source_data_names)}")
        no_of_source_tables = len(source_data_names)
        logging.info(f"source data schema: {source_data_schema}")
        logging.info(f"target data schema:{target_data_schema}")
        # Create a list to store similarity scores of each iteration
        all_similarity_scores = []

        # Iterative Prompt Optimization and Validation
        iteration_count = 0
        validation_table_created = False
        accuracy_list = []
        # Run the experiment
        chatgpt_prompt = generate_prompt_auto_pipeline(no_of_source_tables,source_data_names,
                                                           target_data_names,source_data_schema, target_data_schema,
                                                           target_data_description,test_0_path,test_1_path,sub_folder,template_option)
  
        logging.info(f"final prompt: {chatgpt_prompt}")

        #chatgpt_output = generate_prompt_auto_pipeline()
        logging.info(f"Prompt for GPT Response:{chatgpt_prompt}")
        gpt_output = chat_with_gpt(chatgpt_prompt)
        logging.info(f"gold gpt sql: {gpt_output}")
        # Execute the SQL script on the specified table
        sql_result = execute_sql(conn, gpt_output)
        logging.info(f"SQL Result: {sql_result}")
        logging.info(f"target path: {target_path}")
        logging.info(f"{main_folder_name}{sub_folder}{target_data_name_to_find}_result.csv")
        if (validation_table_created == False):
            gold_target_csv = read_csv_target(target_path)
            validation_table_created = True
        # Validate the ChatGPT generated SQL script
        logging.info(f"validation_table_created: {validation_table_created}")
        case_accuracy, is_correct, similarity_scores, validation_error = validation(sql_result,gold_target_csv)
        accuracy_list.append(case_accuracy)
        all_similarity_scores.append(similarity_scores)
        logging.info(f"is_correct and similarity_score: {is_correct} {similarity_scores}")

        if is_correct:
            log_experiment_success(target_data_names, target_data_name_to_find, iteration_count)
        else:
            log_experiment_failed(target_data_names, target_data_name_to_find, iteration_count, all_similarity_scores,accuracy_list)

        target_id = target_id + 1

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    # Path to the Excel file
    # excel_file_path = 'auto-pipeline-small.xlsx'
    excel_file_path = 'auto-pipeline.xlsx'

    # Path to save the JSON file
    json_file_path = 'auto-pipeline.json'

    # Call the function to perform the conversion
    convert_excel_to_json(excel_file_path, json_file_path)
    template_option = 2 
    #length{length_id}_{target_id} is length1_2
    length_id = 1 
    target_id, max_target_id = 1, 1
    source_id, max_source_id = 0 , 2
    print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id)
    main(json_file_path, template_option, target_id, max_target_id,length_id)
