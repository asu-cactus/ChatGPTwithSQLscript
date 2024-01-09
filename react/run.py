from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity)
from parse_json import get_test_info, get_test_cases_ids
from agent import Agent
import os
import csv
import pandas as pd


def convert_if_number(s):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return s

def are_elements_equal(elem1, elem2, tolerance=1e-8):
    elem1 = '' if elem1 is None else elem1
    elem2 = '' if elem2 is None else elem2
    elem1, elem2 = convert_if_number(elem1), convert_if_number(elem2)
    if isinstance(elem1, float) and isinstance(elem2, float):
        return abs(elem1 - elem2) < tolerance
    elif isinstance(elem1, str) and isinstance(elem2, str):
        return elem1.strip().lower() == elem2.strip().lower()
    else:
        return elem1 == elem2

def numerical_similarity(num1, num2, threshold=1e-8):
    return abs(num1 - num2) < threshold

def compare_columns(column_a, column_b, similarity_type="jaccard", threshold=1e-8):
    if similarity_type == "numerical":
        scores = [numerical_similarity(float(val1), float(val2), threshold) for val1, val2 in zip(column_a, column_b)]
        return sum(scores) / len(scores)
    elif similarity_type == "jaccard":
        intersection = len(set(column_a) & set(column_b))
        union = len(set(column_a) | set(column_b))
        return intersection / union if union else 0

# Convert list of tuples to sorted DataFrame
def convert_to_sorted_df(lst):
    column_names = [f"Column{i+1}" for i in range(len(lst[0]))]
    df = pd.DataFrame(lst, columns=column_names)

    # Sort by all columns
    sorted_df = df.sort_values(by=list(df.columns))
    return sorted_df

def is_column_numerical(column):
    # Expanded check for numerical types including unsigned integers
    return column.dtype.kind in 'fiu' or pd.to_numeric(column, errors='coerce').notna().all()


def is_column_numerical(column):
    try:
        # Attempt to convert the column to a numeric type
        pd.to_numeric(column, errors='raise')
        return True
    except ValueError:
        # If ValueError is raised, the conversion failed, indicating non-numerical values
        return False

# Main Comparison Function
def compare_lists_matching(list1, list2):
    df1 = convert_to_sorted_df(list1)
    df2 = convert_to_sorted_df(list2)

    if len(df1.columns) == 0 or len(df2.columns) == 0:
        return 0, False, [], ["Mismatch - No columns in one or both DataFrames"]

    if len(df1) != len(df2):
        return 0, False, [], [f"Mismatch - DataFrames lengths differ ({len(df1)} vs {len(df2)})"]

    similarities = []
    all_mismatches = []

    for col in df1.columns:
        column_a = df1[col].tolist()
        column_b = df2[col].tolist()

        # Determine the similarity type (numerical or Jaccard)

        column_a = df1[col].tolist()
        column_b = df2[col].tolist()

        # Determine if the column is numerical
        is_numerical = is_column_numerical(df1[col])
        similarity_type = "numerical" if is_numerical else "jaccard"

        column_similarity = compare_columns(column_a, column_b, similarity_type)
        similarities.append(column_similarity)

        if column_similarity < 1:
            mismatches = [(i, column_a[i], column_b[i]) for i in range(len(column_a)) if not are_elements_equal(column_a[i], column_b[i])]
            all_mismatches.append((col, mismatches))

    average_similarity = sum(similarities) / len(df1.columns)
    res = average_similarity == 1

    return average_similarity, res, similarities, all_mismatches

def main(
    max_len_id,
    len_id,
    max_target_id,
    target_id,
    max_source_id,
    source_id=0,
    oneshot_source_id=0,
    max_iterations=2,
    json_file_path='./data/chatgpt.json',
    clarify_on=False,
):
    conn = create_connection()
    print("Postgres connection established.")
    print(f"target_id: {target_id}, max_target_id: {max_target_id}, source_id: {source_id}, max_source_id: {max_source_id}")
    test_cases_list =  get_test_cases_ids(json_file_path, len_id, max_len_id, target_id, max_target_id)
    print(f"test_cases_list: {test_cases_list}")
    for test_case in test_cases_list:
        len_id_target_id = test_case[6:]
        # Get the information of the target and source data
        target_data_name, target_data_schema, target_samples, file_count, source_data_name_list, source_data_schema_list, source_samples_list = get_test_info(
            json_file_path, len_id_target_id)

        # Create a list to store similarity scores of each iteration
        all_similarity_scores = []

        # Iterative Prompt Optimization and Validation
        iteration_count = 0
        validation_table_created = False
        ground_truth_sql_result = None
        accuracy_list = []
        sql_errors = ['']
        # Run the experiment
        while True:
            iteration_count += 1
            # check if reached to max number of iterations
            if iteration_count > max_iterations:
                log_experiment_failed(target_data_name, source_data_name_list, iteration_count,
                                        all_similarity_scores, accuracy_list)
                all_similarity_scores = []
                break
            print(f"*** itr {iteration_count} ***")

            sub_folder_name = f"length{len_id_target_id}"
            #main_folder_name = os.path.abspath("github-pipelines")
            main_folder_name = os.path.abspath("/tmp/github-pipelines")  # Changed to point to /tmp for mac
            target_path = os.path.join(main_folder_name, sub_folder_name, f"target.csv")
            test_0_path = os.path.join(main_folder_name, sub_folder_name, f"test_0.csv")
            result_path = os.path.join(main_folder_name, sub_folder_name, '')
            source_name = f"Source{len_id_target_id}_{source_id}"
            target_name = f"Target{len_id_target_id}"

            # interact with gpt
            #　Create agnet
            agent = Agent(source_name, target_name, test_0_path, source_data_schema_list, target_data_schema, source_samples_list, target_samples, result_path, clarify_on=clarify_on)
            agent.prompt = agent.prompt + f"\n\nFix the following Error: {sql_errors[-1]}\n" if sql_errors[-1] != '' else agent.prompt
            #　Run agent
            gpt_output = agent.run()[0]
            print("SQL Script Extracted from GPT Response:")
            print(gpt_output)

            # Execute the SQL script on the specified table
            sql_result = execute_sql(conn, gpt_output)
            #print(f"SQL Result: {sql_result}")
            if "Error:" in sql_result:
                print( f"\n iter{iteration_count} Error in the previous response: {sql_result}")
                with open('log/all_similarity_scores.log', 'a+') as file:
                    file.write(f"{target_data_name} <- {source_data_name_list}")
                    file.write(f"\t\t\t\t[Failed]\n\tError in the previous response: {sql_result}\n")
                accuracy_list.append(0.0)
                #break
                sql_errors.append(sql_result)
                continue

            ground_truth_table = []
            # SQL script returned by ChatGPT is executed correctly
            if (validation_table_created == False):
                with open(target_path, 'r', encoding="utf-8") as file:
                    reader = csv.reader(file)
                    header = next(reader)
                    for row in reader:
                        ground_truth_table.append(tuple(row))
                validation_table_created = True

            # Validate the ChatGPT generated SQL script
            case_accuracy, is_correct, similarity_scores, validation_error = compare_lists_matching(sql_result,
                                                                                        ground_truth_table)

            accuracy_list.append(case_accuracy)
            all_similarity_scores.append(similarity_scores)
            print(is_correct)

            if is_correct:
                log_experiment_success(target_data_name, source_data_name_list, iteration_count)
                all_similarity_scores = []
                break
            else:
                # to be revised, phase 2
                print(f"The returned SQL script can run, but the execution result of the SQL is wrong: {validation_error}. Please try again.")

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    
    template_option = 1
    max_len_id, len_id = 1, 1
    target_id, max_target_id = 1, 99
    source_id, max_source_id = 0, 0
    print_experiment_settings(len_id, max_len_id, target_id, max_target_id, clarify_on=False)
    oneshot_source_id = 0 # Set to 0 to disable oneshot
    main(max_len_id, len_id, max_target_id, target_id, max_source_id, source_id=0, oneshot_source_id=0, max_iterations=5, json_file_path='data/chatgpt.json', clarify_on=False, )
