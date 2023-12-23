from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity)
from parse_json import get_test_info, get_test_cases_ids
from agent import Agent
import os
import csv

import numpy as np
from scipy.optimize import linear_sum_assignment

def are_numerically_close(num1, num2, tolerance=1e-8):
    return abs(num1 - num2) < tolerance

def are_strings_similar(str1, str2):
    return str1.strip().lower() == str2.strip().lower()

def convert_if_number(s):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return s

def are_elements_equal(elem1, elem2, tolerance=1e-8):
    if elem1 is None or elem2 is None:
        return elem1 is None and elem2 is None
    elem1, elem2 = convert_if_number(elem1), convert_if_number(elem2)

    if isinstance(elem1, float) and isinstance(elem2, float):
        return abs(elem1 - elem2) < tolerance
    elif isinstance(elem1, str) and isinstance(elem2, str):
        return elem1.strip().lower() == elem2.strip().lower()
    else:
        return elem1 == elem2

def tuple_similarity(t1, t2):
    if len(t1) != len(t2):
        return 0, ["Mismatch - Tuple lengths differ"]

    matching_elements = 0
    mismatches = []
    for index, (e1, e2) in enumerate(zip(t1, t2)):
        if are_elements_equal(e1, e2):
            matching_elements += 1
        else:
            mismatches.append(f"Column {index + 1}: {e1} vs {e2}")

    similarity = matching_elements / len(t1) if t1 else 0
    return similarity, mismatches

def calculate_column_similarities(matched_tuples, list1, list2):
    num_columns = len(list1[0])
    column_similarities = [0] * num_columns

    for r, c in matched_tuples:
        for i in range(num_columns):
            if are_elements_equal(list1[r][i], list2[c][i]):
                column_similarities[i] += 1

    return [sim / len(list1) for sim in column_similarities]

def compare_lists_matching(list1, list2):
    if not list1 or not list2 or len(list1) != len(list2):
        return 0, False, ["missmatch"], ["Mismatch - Lists lengths differ"]

    # Create a similarity matrix
    similarity_matrix = np.zeros((len(list1), len(list2)))
    mismatch_details = {}

    for i, t1 in enumerate(list1):
        for j, t2 in enumerate(list2):
            similarity, mismatches = tuple_similarity(t1, t2)
            similarity_matrix[i][j] = similarity
            if mismatches:
                mismatch_details[(i, j)] = mismatches

    # Find the optimal matching using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
    total_similarity = similarity_matrix[row_ind, col_ind].sum()
    matched_tuples = list(zip(row_ind, col_ind))

    # Calculate column-wise similarities based on matched tuples
    column_similarities = calculate_column_similarities(matched_tuples, list1, list2)

    # Collect mismatch information
    all_mismatches = []
    for r, c in matched_tuples:
        if (r, c) in mismatch_details:
            all_mismatches.extend(mismatch_details[(r, c)])

    average_similarity = total_similarity / len(list1)
    res = average_similarity == 1

    return average_similarity, res, column_similarities, all_mismatches

def validation(sql_result, ground_truth, tolerance=1e-10):
    validation_error = ""

    if len(sql_result) != len(ground_truth) or len(sql_result[0]) != len(ground_truth[0]):
        validation_error = "Different number of rows or columns in the results and ground truth."
        return 0.0, False, ["missmatch"], validation_error

    # Initialize a list to store similarity scores
    similarity_scores = []
    fully_matched_columns_num = 0  # Counter for columns that matched perfectly

    res = True
    # Iterate through columns and compare
    for col_index in range(len(sql_result[0])):
        sql_column = [str(row[col_index]) if row[col_index] is not None else '0.0' for row in sql_result]
        truth_column = [str(row[col_index]) if row[col_index] is not None else '0.0' for row in ground_truth]

        # Determine whether columns can be treated as numerical
        is_sql_numeric = True
        is_truth_numeric = True

        try:
            float(sql_column[0])  # Try converting the first value
        except (ValueError, TypeError):
            is_sql_numeric = False

        try:
            float(truth_column[0])
        except (ValueError, TypeError):
            is_truth_numeric = False

        if is_sql_numeric and is_truth_numeric:
            similarity_type = "numerical"
        else:
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

    case_accuracy = fully_matched_columns_num / len(sql_result[0])
    print("Similarity scores for this iteration:", similarity_scores)

    # Returning both the result of strict validation, the similarity scores, and the global accuracy
    return case_accuracy, res, similarity_scores, validation_error

def main(
    max_len_id,
    len_id,
    max_target_id,
    target_id,
    max_source_id,
    source_id=0,
    oneshot_source_id=0,
    max_iterations=1,
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
            main_folder_name = os.path.abspath("github-pipelines")
            target_path = os.path.join(main_folder_name, sub_folder_name, f"target.csv")
            test_0_path = os.path.join(main_folder_name, sub_folder_name, f"test_0.csv")
            result_path = os.path.join(main_folder_name, sub_folder_name, '')
            source_name = f"Source{len_id_target_id}_{source_id}"
            target_name = f"Target{len_id_target_id}"

            # interact with gpt
            #　Create agnet
            agent = Agent(source_name, target_name, test_0_path, source_data_schema_list, target_data_schema, source_samples_list, target_samples, result_path, clarify_on=clarify_on)

            gpt_output = agent.run()[0]
            print("SQL Script Extracted from GPT Response:")
            print(gpt_output)

            # Execute the SQL script on the specified table
            sql_result = execute_sql(conn, gpt_output)
            #print(f"SQL Result: {sql_result}")
            if "Error:" in sql_result:
                print( f"\n Error in the previous response: {sql_result}")
                with open('log/all_similarity_scores.log', 'a+') as file:
                    file.write(f"{target_data_name} <- {source_data_name_list}")
                    file.write(f"\t\t\t\t[Failed]\n\tError in the previous response: {sql_result}\n")
                accuracy_list.append(0.0)
                break

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
    main(max_len_id, len_id, max_target_id, target_id, max_source_id, source_id=0, oneshot_source_id=0, max_iterations=1, json_file_path='data/chatgpt.json', clarify_on=False, )
