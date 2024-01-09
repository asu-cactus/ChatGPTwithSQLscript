import os
import pandas as pd
import json
from datetime import datetime
from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed)
from join_util import convert_target_names,access_auto_pipeline_dataset,read_csv_target
from gpt import chat_with_gpt, gpt4_sql_script
from join import validation
import logging
from io import StringIO

logging.basicConfig(filename='auto_pipeline_join_valid_test_l1.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        "Target Data Description",
        "3 Samples of Source Data"]

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

def generate_prompt_auto_pipeline(no_of_source_tables,source_names,target_name,source_data_schema,target_data_schema,target_data_description,
                                  samples,test_0_path,test_1_path,test_2_path,test_3_path,test_4_path, test_5_path, test_6_path, test_7_path,test_8_path,
                                  sub_folder,template_option):
    target_name = target_name[0] 
    no_of_source_tables = no_of_source_tables
    source_data_schema = source_data_schema
    target_data_schema = target_data_schema[0]
    target_data_description = target_data_description[0]
    sample_i = create_sample_i(samples)
    sample_0 = sample_i.get("sample_0")
    sample_1 = sample_i.get("sample_1")
    sample_2 = sample_i.get("sample_2")
    sample_3 = sample_i.get("sample_3")
    sample_4 = sample_i.get("sample_4")
    sample_5 = sample_i.get("sample_5")
    sample_6 = sample_i.get("sample_6")
    sample_7 = sample_i.get("sample_7")
    sample_8 = sample_i.get("sample_8")

    if no_of_source_tables == 1 and template_option == 4:  #without target description
        prompt = f"""
        You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create {no_of_source_tables} source table with following {source_names} with only the given attributes: {source_data_schema}. 
        Please delete the table before creating it if the source table exists.
        Second, insert the entire csv file with headers from given paths {test_0_path} the {no_of_source_tables} source table respectively (treat empty value as NULL).
        Source table samples are as follows {sample_0}.
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, insert all rows from the source table into only one {target_name}, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation. 
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```".
        """
    elif no_of_source_tables == 2 and template_option == 4: #without target description
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source table exists.
        Second, insert the entire csv file with headers from given paths {test_0_path} and {test_1_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1}.
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 3 and template_option == 4: #with target description
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source table exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path} and {test_2_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_3}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 4 and template_option == 4:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source tables exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path}, {test_2_path} and {test_3_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_2} and Fourth table samples are as follows {sample_3}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 5 and template_option == 4:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source tables exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path}, {test_2_path}, {test_3_path} and {test_4_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_2} and Fourth table samples are as follows {sample_3} and Fifth table samples are as follows {sample_4}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 6 and template_option == 4:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source tables exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path}, {test_2_path}, {test_3_path}, {test_4_path} and {test_5_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_2} and Fourth table samples are as follows {sample_3} and Fifth table samples are as follows {sample_4} and Sixth table samples are as follows {sample_5}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 7 and template_option == 4:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source tables exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path}, {test_2_path}, {test_3_path}, {test_4_path}, {test_5_path} and {test_6_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_2} and Fourth table samples are as follows {sample_3} and Fifth table samples are as follows {sample_4} and Sixth table samples are as follows {sample_5}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 8 and template_option == 4:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source tables exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path}, {test_2_path}, {test_3_path}, {test_4_path}, {test_5_path}, {test_6_path} and {test_7_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_2} and Fourth table samples are as follows {sample_3} and Fifth table samples are as follows {sample_4} and Sixth table samples are as follows {sample_5} and Seventh table samples are as follows {sample_6} and Eight table samples are as follows {sample_7}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
        Please quote the returned SQL script between "```sql\n" and "\n```". 
        """
    elif no_of_source_tables == 9 and template_option == 4:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the {no_of_source_tables} source table to be consistent with the format of the target table {target_id}. 
        First, you must create  {no_of_source_tables} source tables with following {source_names} with only the given attributes respectively: {source_data_schema}. 
        Please delete the table before creating it if the source tables exists.
        Second, insert the entire csv file with headers from given paths {test_0_path}, {test_1_path}, {test_2_path}, {test_3_path}, {test_4_path}, {test_5_path}, {test_6_path}, {test_7_path} and {test_8_path} into the {no_of_source_tables} tables respectively (treat empty value as NULL).
        First table samples are as follows {sample_0} and Second table samples are as follows {sample_1} and Third table samples are as follows {sample_2} and Fourth table samples are as follows {sample_3} and Fifth table samples are as follows {sample_4} and Sixth table samples are as follows {sample_5} and Seventh table samples are as follows {sample_6} and Eight table samples are as follows {sample_7} and Ninth table samples are as follows {sample_8}
        Third, you must create a target table named {target_name} with only the given attributes: {target_data_schema}. 
        Please delete the table before creating it if the target table exists.
        Finally, the script convert these {no_of_source_tables} source tables into only one {target_name} target table, note that the selection clause in the insert statement should ignore attributes that are not needed.
        Also COPY the SQL result into a f"{sub_folder}{target_name}_result.csv" file. Use client-side facility such as psql's copy.
        Please don't remove the any table, because we need it for validation.
        Only generate one SQL statement for each given instruction. Do not give repetitive SQL.
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
    samples = []
    for target_key, target_values in data_list.items():
        for target_value in target_values:
            if target_value.get("Target Data Name") == target_data_name_to_find:
                target_data_names.append(target_value.get("Target Data Name"))
                source_data_names.append(target_value.get("Source Data Name"))
                source_data_schema.append(target_value.get("Source Data Schema"))
                target_data_schema.append(target_value.get("Target Data Schema"))
                target_data_description.append(target_value.get("Target Data Description"))
                samples.append(target_value.get("3 Samples of Source Data"))
                #ground_truth_sql_result.append(target_value.get("Ground Truth SQL"))
    return target_data_names, source_data_names,source_data_schema, target_data_schema, target_data_description,samples


def main(*args):
    (json_file_path, template_option, target_id, max_target_id,length_id) = args
    conn = create_connection()

    while target_id <= max_target_id:
        target_data_name_to_find = "Target" + str(length_id) + "_" + str(target_id)
        logging.info(f"target to find {target_data_name_to_find} and {json_file_path}")
        # Get JSON data for prompt
        target_data_names, source_data_names,source_data_schema, target_data_schema, target_data_description,samples = gpt_auto_pipeline(json_file_path,target_data_name_to_find)
        logging.info(f"here returned {target_data_names, source_data_names,source_data_schema, target_data_schema, target_data_description,samples}")
        no_of_source_tables = len(source_data_names)
        logging.info(f"target_data_names from initial function {target_data_names}")
        find_target_name_folder = convert_target_names(target_data_names[0])
        main_folder_name,sub_folder, test_0_path, test_1_path, test_2_path,test_3_path, test_4_path, test_5_path, test_6_path, test_7_path,test_8_path, target_path = access_auto_pipeline_dataset(find_target_name_folder) 
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
        chatgpt_prompt = generate_prompt_auto_pipeline(no_of_source_tables,source_data_names,target_data_names,source_data_schema, target_data_schema,target_data_description,samples,
                                                       test_0_path,test_1_path,test_2_path,test_3_path,test_4_path, test_5_path, test_6_path, test_7_path,test_8_path, sub_folder,template_option)
  
        logging.info(f"final prompt: {chatgpt_prompt}")
        #gpt_output = chat_with_gpt(chatgpt_prompt)
        total_tokens=10000
        gpt_output = gpt4_sql_script(chatgpt_prompt, total_tokens)
        logging.info(f"gold gpt sql: {gpt_output}")
        # Execute the SQL script on the specified table
        sql_result = execute_sql(conn, gpt_output)
        if "Error:" in sql_result:
            logging.info(f"\n Error in the previous response: {sql_result}")
            chatgpt_prompt += f"\n Error in the previous response: {sql_result}"
            print(f"ERROR: {sql_result}")
            accuracy_list.append(0.0)
            break
                
        logging.info(f"sql_result --> {type(sql_result)}: {sql_result}")
        sql_result_df = pd.DataFrame(sql_result)
        logging.info(f"SQL Result DF Final: {sql_result_df}")
        
        logging.info(f"target path: {target_path}")
        logging.info(f"{sub_folder}{target_data_name_to_find}_result.csv")
        target_result_csv = f"{sub_folder}{target_data_name_to_find}_result.csv"
        logging.info(f"target_result_csv: {target_result_csv}")
        target_result_csv_pd = pd.read_csv(target_result_csv)
        target_result_csv_pd_df = pd.DataFrame(target_result_csv_pd)
        target_result_csv_pd_df_sort = target_result_csv_pd_df.sort_values(by=list(target_result_csv_pd_df.columns))
        logging.info(f"target_result_csv_pd_df_sort: {target_result_csv_pd_df_sort}")
        if (validation_table_created == False):
            gold_target_csv = read_csv_target(target_path)
            gold_target_csv_pd = pd.read_csv(target_path)
            gold_target_csv_df = pd.DataFrame(gold_target_csv_pd)
            logging.info(f"gold_target_csv_df {gold_target_csv_df}")
            gold_target_csv_df_sort = gold_target_csv_df.sort_values(by=list(gold_target_csv_df.columns))
            logging.info(f"gold_target_csv_df_sort {gold_target_csv_df_sort}")
            validation_table_created = True
        # Validate the ChatGPT generated SQL script
        logging.info(f"validation_table_created: {validation_table_created}")
        case_accuracy, is_correct, similarity_scores, validation_error = validation(target_result_csv_pd_df_sort,gold_target_csv_df_sort)
        accuracy_list.append(case_accuracy)
        all_similarity_scores.append(similarity_scores)
        logging.info(f"{target_data_name_to_find} is_correct and similarity_score: {is_correct} {similarity_scores}")

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
    excel_file_path = 'auto-pipeline-100.xlsx'

    # Path to save the JSON file
    json_file_path = 'auto-pipeline-100.json'

    # Call the function to perform the conversion
    convert_excel_to_json(excel_file_path, json_file_path)
    template_option = 4 
    #length{length_id}_{target_id} is length1_2
    length_id = 1
    target_id, max_target_id = 6,6
    source_id, max_source_id = 0 , 8
    print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id)
    logging.info(f"*********** Starting template option and target_id: {template_option},{target_id}****************")
    print(f"*Starting template option and target_id: {template_option},Target{length_id}_{target_id}")
    main(json_file_path, template_option, target_id, max_target_id,length_id)
