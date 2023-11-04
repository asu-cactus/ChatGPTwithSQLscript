from util import (create_connection, execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity,extract_source_table)
from gpt import generate_prompt, chat_with_gpt
from quality_control import reverse_quality,validation,schema_quality,differential_quality,information_case,fd_quality,mapping_quality


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
                prompt, ground_truth_query, target_data_name = generate_prompt(json_file_path, template_option,0,
                                                                               source_data_name_to_find)
                initial_prompt = prompt
            else:
                oneshot_data_name_to_find = f"Source{target_id}_{oneshot_source_id}"
                prompt, ground_truth_query, target_data_name = generate_prompt(json_file_path, template_option,0,
                                                                               source_data_name_to_find, oneshot_data_name_to_find)
                initial_prompt = prompt
           

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
                gpt_output = chat_with_gpt(prompt,True)
                print("SQL Script Extracted from GPT Response:")
                print(gpt_output)

                # Execute the SQL script on the specified table
                sql_result = execute_sql(conn, gpt_output)
                print(f"SQL Result: {sql_result}")
                if "Error:" in sql_result:
                    prompt = initial_prompt + f"\n Error in the previous response: {sql_result}"
                    print(prompt)
                    accuracy_list.append(0.0)
                    continue

                # # SQL script returned by ChatGPT is executed correctly
                # if (validation_table_created == False):
                #     ground_truth_sql_result = execute_sql(conn, ground_truth_query)
                #     validation_table_created = True
                #
                # print(f"\nGround Truth SQL Query: {ground_truth_query}")
                # print(f"\nGround Truth SQL Query Result: {ground_truth_sql_result}")

                # # Validate the ChatGPT generated SQL script
                # case_accuracy, is_correct, similarity_scores, validation_error = validation(sql_result,
                #                                                                             ground_truth_sql_result)
                # accuracy_list.append(case_accuracy)
                # all_similarity_scores.append(similarity_scores)
                # print(is_correct)

                differential_score, differential_feedback = differential_quality(gpt_output, conn,
                                                                                 source_data_name_to_find,
                                                                                 target_data_name)
                threshold = 0.9
                schema_score,schema_feedback = schema_quality(gpt_output, source_data_name_to_find, target_data_name,json_file_path)
                if schema_score < 1:
                    prompt = initial_prompt + f"The returned SQL script can run, but the execution result of the SQL is wrong. Please try again."+schema_feedback
                    print(prompt + "\n")
                    continue

                mapping_score,mapping_feedback = mapping_quality(gpt_output,source_data_name_to_find, target_data_name)
                if mapping_score < threshold:
                    prompt = initial_prompt + f"The returned SQL script can run, but the execution result of the SQL is wrong. Please try again."+mapping_feedback
                    print(prompt + "\n")
                    continue
                #fd_score,fd_feedback = fd_quality(conn, gpt_output, source_data_name_to_find, target_data_name)
                # if fd_score < threshold:
                #     prompt = initial_prompt + f"The returned SQL script can run, but the execution result of the SQL is wrong. Please try again."+fd_feedback
                #     print(prompt + "\n")
                #     continue

                information_loss = information_case(mapping_feedback,gpt_output,source_data_name_to_find, target_data_name)
                if information_loss == 0:
                    reverse_score = reverse_quality(json_file_path, gpt_output, sql_result, source_data_name_to_find,
                                                    accuracy_list, conn, all_similarity_scores)
                    if reverse_score > threshold:
                        log_experiment_success(target_data_name, source_data_name_to_find, iteration_count)
                        all_similarity_scores = []
                        break
                    else:
                        prompt = initial_prompt + f"The returned SQL script can run, but the execution result of the SQL is wrong. Please try again."
                        print(prompt + "\n")
                        continue
                elif information_loss == 1:
                    differential_score,differential_feedback = differential_quality(gpt_output, conn, source_data_name_to_find,
                                                              target_data_name)
                    if differential_score > threshold:
                        log_experiment_success(target_data_name, source_data_name_to_find, iteration_count)
                        all_similarity_scores = []
                        break
                    else:
                        prompt = initial_prompt + f"The returned SQL script can run, but the execution result of the SQL is wrong. Please try again."+differential_feedback
                        print(prompt + "\n")
                        continue
                else:
                    log_experiment_success(target_data_name, source_data_name_to_find, iteration_count)
                    all_similarity_scores = []
                    break



        target_id = target_id + 1

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    
    template_option = 3
    target_id, max_target_id = 1, 3
    source_id, max_source_id = 8, 8
    print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id)
    oneshot_source_id = 0 # Set to 0 to disable oneshot
    main(template_option, target_id, max_target_id, source_id, max_source_id, oneshot_source_id=oneshot_source_id)
