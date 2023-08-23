from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity)
from gpt import generate_prompt, chat_with_gpt


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


# 5 Samples of Source Data: {sammples}
# main script
def main(*args):
    (json_file_path, template_option, target_id, max_target_id, source_id, max_source_id) = args
    conn = create_connection()
    while target_id <= max_target_id:
        while source_id <= max_source_id:
            # Source Data Name to find
            source_data_name_to_find = "Source" + str(target_id) + "_" + str(source_id)
            source_id = source_id + 1
            print(source_data_name_to_find)
            # Generate the prompt for the chatGPT model
            if template_option in [1, 2, 3, 4]:
                prompt, ground_truth_query, target_data_name = generate_prompt(json_file_path, template_option,
                                                                               source_data_name_to_find)
            else:
                print("Invalid template option.")
                return

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
                if iteration_count > 5:
                    log_experiment_failed(target_data_name, source_data_name_to_find, iteration_count,
                                            all_similarity_scores, accuracy_list)
                    all_similarity_scores = []
                    break
                print("*** itr " + str(iteration_count) + "***")
                # interact with gpt
                gpt_output = chat_with_gpt(prompt)
                print("SQL Script Extracted from GPT Response:")
                print(gpt_output)
                if "Error:" in gpt_output:
                    prompt += " GPT Error: " + gpt_output
                    continue

                # Execute the SQL script on the specified table
                sql_result = execute_sql(conn, gpt_output)
                print("SQL Result:")
                print(sql_result)
                if "Error:" in sql_result:
                    prompt += "\n Error in the previous response:"
                    prompt += sql_result
                    print(prompt)
                    accuracy_list.append(0.0)
                    continue

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
                    log_experiment_success(target_data_name, source_data_name_to_find, iteration_count)
                    all_similarity_scores = []
                    break
                else:
                    # to be revised
                    prompt += "The returned SQL script can run, but the execution result of the SQL is wrong: " + str(
                        validation_error) + " Please try again."

                    print(prompt + "\n")
                    continue
        target_id = target_id + 1

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    json_file_path = 'chatgpt.json'
    template_option = 4
    target_id, max_target_id = 2, 2
    source_id, max_source_id = 10, 10
    print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id)
    main(json_file_path, template_option, target_id, max_target_id, source_id, max_source_id)