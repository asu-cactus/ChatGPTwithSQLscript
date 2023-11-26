import pandas as pd
import json
from datetime import datetime
from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity)
from gpt import generate_prompt, chat_with_gpt

def convert_datetime(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError("Type not serializable")


def convert_excel_to_json(excel_file_path, json_file_path):
    # Read the Excel file
    xls = pd.ExcelFile(excel_file_path)

    # Specify the columns to include in the JSON file
    columns_to_include = [
        "Target Data Name",
        "Target Data Description",
        "Source Data Name",
        "Source Data Description",
        "5 Samples of Source Data",
        "Ground Truth SQL",
        "ChatGPT Response"
    ]

    # Read the specified columns from the first sheet
    data_to_convert = pd.read_excel(xls, sheet_name='Sheet7', usecols=columns_to_include)

    # Fill missing values in the specified columns by forward filling
    columns_to_fill = ["Target Data Name", "Target Data Description"]
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


def validation(sql_result, ground_truth, tolerance=1e-10):
    validation_error = ""

    if len(sql_result) != len(ground_truth):
        return 0.0, False, ['first length'], validation_error
    elif len(sql_result) != len(ground_truth) or len(sql_result.columns) != len(ground_truth.columns):
        validation_error = "Different number of rows or columns in the results and ground truth."
        return 0.0, False, ["missmatch"], validation_error

    # Initialize a list to store similarity scores
    similarity_scores = []
    fully_matched_columns_num = 0  # Counter for columns that matched perfectly

    res = True
    # Iterate through columns and compare
    for col_index in range(len(sql_result.columns)):
        sql_column = sql_result.iloc[:, col_index].apply(lambda x: str(x) if pd.notna(x) else '0.0').tolist()
        truth_column = ground_truth.iloc[:, col_index].apply(lambda x: str(x) if pd.notna(x) else '0.0').tolist()

        # Determine whether columns can be treated as numerical
        is_sql_numeric = sql_result.iloc[:, col_index].apply(lambda x: isinstance(x, (int, float))).all()
        is_truth_numeric = ground_truth.iloc[:, col_index].apply(lambda x: isinstance(x, (int, float))).all()

        if is_sql_numeric and is_truth_numeric:
            similarity_type = "numerical"
        else:
            # [for x in [is_sql_numeric, is_truth_numeric] ]
            similarity_type = "jaccard"

        similarity_score = calculate_similarity(sql_column, truth_column, similarity_type=similarity_type,
                                                threshold=tolerance)

        if similarity_score == 1:
            fully_matched_columns_num += 1
        elif similarity_score < 1:
            validation_error += f"the {col_index}-th column does not match; "
            res = False

        # Append the similarity score for this column
        similarity_scores.append(similarity_score)

    case_accuracy = fully_matched_columns_num / len(sql_result.columns)
    print("Similarity scores for this iteration:", similarity_scores)

    # Returning both the result of strict validation, the similarity scores, and the global accuracy
    return case_accuracy, res, similarity_scores, validation_error

def gpt(json_file_path, source_data_name_to_find):
    # Read the JSON file
    with open(json_file_path, 'r') as file:
        data_list = json.load(file)

    # Find the item with the specified Source Data Name
    data = None
    for item in data_list:
        if item["Source Data Name"] == source_data_name_to_find:
            data = item
            break
    if data is None:
        return f"No data found for Source Data Name: {source_data_name_to_find}"

    # Extract the relevant information from the JSON data
    target_data_name = data["Target Data Name"]
    source_data_name = data["Source Data Name"]
    samples = data["5 Samples of Source Data"]
    target_data_description = data["Target Data Description"]
    source_data_description = data["Source Data Description"]
    gpt_response = data["ChatGPT Response"]
    ground_truth = data["Ground Truth SQL"]
    return gpt_response, ground_truth,target_data_name


def main(*args):
    (json_file_path, template_option, target_id, max_target_id) = args
    conn = create_connection()

    while target_id <= max_target_id:
        # Source Data Name to find
        target_data_name_to_find = "TargetJ" + "_" +str(target_id)
        print(target_data_name_to_find)
        # Generate the prompt for the chatGPT model
        gpt_response,ground_truth_query,target_name = gpt(json_file_path, template_option,target_data_name_to_find)

        # Create a list to store similarity scores of each iteration
        all_similarity_scores = []

        # Iterative Prompt Optimization and Validation
        iteration_count = 0
        validation_table_created = False
        ground_truth_sql_result = None
        accuracy_list = []
        # Run the experiment
        gpt_output = gpt_response
        print("SQL Script Extracted from GPT Response:")
        print(gpt_output)
        # Execute the SQL script on the specified table
        sql_result = execute_sql(conn, gpt_output)
        print("SQL Result:")
        print(sql_result)
        # SQL script returned by ChatGPT is executed correctly
        if (validation_table_created == False):
            ground_truth_sql_result = execute_sql(conn, ground_truth_query)
            validation_table_created = True

        print("\nGround Truth SQL Query:")
        print(ground_truth_query)
        print("\nGround Truth SQL Query Result:")
        print(ground_truth_sql_result)

        # Validate the ChatGPT generated SQL script
        case_accuracy, is_correct, similarity_scores, validation_error = validation(sql_result,
                                                                                            ground_truth_sql_result)
        accuracy_list.append(case_accuracy)
        all_similarity_scores.append(similarity_scores)
        print(is_correct)

        if is_correct:
            log_experiment_success(target_name, target_data_name_to_find, iteration_count)
        else:
            log_experiment_failed(target_name, target_data_name_to_find, iteration_count, validation_error)

        target_id = target_id + 1

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    # Path to the Excel file
    excel_file_path = 'D:/SQL/chatgpt.xlsx'

    # Path to save the JSON file
    json_file_path = 'D:/SQL/chatgpt.json'

    # Call the function to perform the conversion
    convert_excel_to_json(excel_file_path, json_file_path)
    json_file_path = 'chatgpt.json'
    template_option = 1
    target_id, max_target_id =21 , 22
    print_experiment_settings(template_option, target_id, max_target_id)
    main(json_file_path, template_option, target_id, max_target_id)
