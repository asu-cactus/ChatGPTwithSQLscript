from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity)
from gpt import generate_prompt, chat_with_gpt


def validation(sql_result, ground_truth, tolerance=1e-10):
    validation_error = ""

    if len(sql_result) != len(ground_truth) or len(sql_result[0]) != len(ground_truth[0]):
        validation_error = "Different number of rows or columns in the results and ground truth."
        return 0.0, False, ["mismatch"], validation_error

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
        except:
            is_sql_numeric = False

        try:
            float(truth_column[0])
        except:
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
def main(
    template_option, 
    target_id, 
    max_target_id, 
    source_id, 
    max_source_id,
    oneshot_source_id=0,
    max_iterations=5,
    json_file_path='chatgpt.json',
):
    conn = create_connection()
    print("Connection established.")
    print(f"target_id: {target_id}, max_target_id: {max_target_id}, source_id: {source_id}, max_source_id: {max_source_id}")
    while target_id <= max_target_id:
        while source_id <= max_source_id:
            # Source Data Name to find
            source_data_name_to_find = "Source" + str(target_id) + "_" + str(source_id)
            source_id = source_id + 1
            print(source_data_name_to_find)
            
            # Generate the prompt for the chatGPT model
            if oneshot_source_id == 0:
                prompt, ground_truth_query, target_data_name = generate_prompt(json_file_path, template_option,
                                                                               source_data_name_to_find)
            else:
                oneshot_data_name_to_find = f"Source{target_id}_{oneshot_source_id}"
                prompt, ground_truth_query, target_data_name = generate_prompt(json_file_path, template_option,
                                                                               source_data_name_to_find, oneshot_data_name_to_find)
           

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
                    log_experiment_failed(target_data_name, source_data_name_to_find, iteration_count,
                                            all_similarity_scores, accuracy_list)
                    all_similarity_scores = []
                    break
                print(f"*** itr {iteration_count} ***")
                # interact with gpt
                gpt_output = chat_with_gpt(prompt)
                print("SQL Script Extracted from GPT Response:")
                print(gpt_output)

                # Execute the SQL script on the specified table
                sql_result = execute_sql(conn, gpt_output)
                print(f"SQL Result: {sql_result}")
                if "Error:" in sql_result:
                    prompt += f"\n Error in the previous response: {sql_result}"
                    print(prompt)
                    accuracy_list.append(0.0)
                    continue

                # SQL script returned by ChatGPT is executed correctly
                if (validation_table_created == False):
                    ground_truth_sql_result = execute_sql(conn, ground_truth_query)
                    validation_table_created = True

                print(f"\nGround Truth SQL Query: {ground_truth_query}")
                print(f"\nGround Truth SQL Query Result: {ground_truth_sql_result}")

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
                    prompt += f"The returned SQL script can run, but the execution result of the SQL is wrong: {validation_error}. Please try again."
                    print(prompt + "\n")
                    continue
        target_id = target_id + 1

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    
    template_option = 1
    target_id, max_target_id = 26, 26
    source_id, max_source_id = 1, 1
    print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id)
    oneshot_source_id = 0 # Set to 0 to disable oneshot
    main(template_option, target_id, max_target_id, source_id, max_source_id, oneshot_source_id=oneshot_source_id)
