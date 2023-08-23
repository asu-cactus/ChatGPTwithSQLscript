import openai
import pymysql
from pymysql.err import OperationalError
import difflib
import numpy as np
from pymysql.err import ProgrammingError
import psycopg2
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import csv
import re
from config import OPENAI_API_KEY
from decimal import Decimal

# openai key -- from config import OPENAI_API_KEY
# in config.py put OPENAI_API_KEY='your_key'
openai.api_key = OPENAI_API_KEY

def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def create_connection():
    """ create a database connection to the PostgreSQL database """
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="localhost",  # e.g., "localhost"
            port="5432"  # e.g., "5432"
        )
        return conn
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return None


def extract_last_insert_table_name(query):
    """
    Extracts the table name from the last INSERT INTO clause in the given SQL query.
    """
    matches = re.findall(r"INSERT\s+INTO\s+(\w+)", query, re.IGNORECASE)
    if matches:
        return matches[-1]
    return None


def execute_sql(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN;")
        cursor.execute(query)
        # Assuming you want to commit after every SQL execution for simplicity
        conn.commit()

        # Check if the operation is not a SELECT statement
        if not query.strip().upper().startswith("SELECT"):
            target_table = extract_last_insert_table_name(query)
            if target_table:
                # Fetch results from the last inserted table
                cursor.execute(f"SELECT * FROM {target_table};")
                result = cursor.fetchall()
            else:
                result = "Table name not identified from last INSERT INTO query."
        else:
            result = cursor.fetchall()

        return result
    except psycopg2.Error as e:
        conn.rollback()  # Rollback the transaction on error
        return f"Error: {e.pgerror}"


def create_table(conn, create_statement):
    print(create_statement)
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN;")
        cursor.execute(create_statement)
        # Assuming you want to commit after every SQL execution for simplicity
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()  # Rollback the transaction on error
        return f"Error: {e.pgerror}"


# interact with chatGPT model
def chat_with_gpt(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10000,
        )
        complete_response_message = response.choices[0]['message']['content']

        #print("Complete Response:")
        #print(complete_response_message)

        sub1 = "```sql"
        sub2 = "```"
        sql_script = ''.join(complete_response_message.split(sub1)[1].split(sub2)[0])

        #print("SQL Script Extracted from GPT Response:")
        #print(sql_script)

        return sql_script
    except Exception as e:
        return str(e)


def calculate_similarity(column_a, column_b, similarity_type="numerical", threshold=1e-10):
    if similarity_type == "cosine":
        vectorizer = CountVectorizer().fit_transform(column_a + column_b)
        cosine_sim = cosine_similarity(vectorizer[:len(column_a)], vectorizer[len(column_a):])
        return cosine_sim[0, 0]
    elif similarity_type == "numerical":
        # Compare each pair of values in the columns and return the average similarity
        scores = [numerical_similarity(val1, val2, threshold) for val1, val2 in zip(column_a, column_b)]
        return sum(scores) / len(scores)
    else:  # Jaccard similarity
        intersection = len(set(column_a) & set(column_b))
        union = len(set(column_a) | set(column_b))
        return intersection / union if union != 0 else 0

def numerical_similarity(value1, value2, threshold=1e-10):
    """
    Calculate similarity for numerical values.
    Returns 1.0 if the difference is below the threshold, 0.0 otherwise.
    value2 is ground truth
    """
    try:
        if value1 is None and value2 is None:
            return 1.0
        if value1 is None and value2 == 0:
            return 1.0
        diff = abs(float(value1) - float(value2))
        if diff <= threshold:
            return 1.0
    except ValueError:
        return 0.0
    return 0.0


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

def generate_prompt(json_file_path, template_option, source_data_name_to_find):
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
    target_data_schema = data["Target Data Schema"]
    source_data_name = data["Source Data Name"]
    source_data_schema = data["Source Data Schema"]
    samples = data["5 Samples of Source Data"]
    target_data_description = data["Target Data Description"]
    source_data_description = data["Source Data Description"]
    schema_change_hints = data["Schema Change Hints"]
    notes = data["Remark or Note"]
    ground_truth = data["Ground Truth SQL"]

    # Generate the prompt based on the template option
    if template_option == 1:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you must create the first table named {source_data_name} with only the given attributes: {source_data_schema}. Please delete the table before creating it if the first table exists. 

        Second, insert the following row(s) into the first table:

        {samples}

        Third, you must create a second table named {target_data_name} with only the given attributes: {target_data_schema}. Please delete the table before creating it if the first table exists.

        Finally, insert all rows from the first table into the second table, note that the selection clause in the insert statement should ignore attributes that are not needed.

        Please don't remove the first table, because we need it for validation.

        Please quote the returned SQL script between "```sql\n" and "\n```". 

        {target_data_description}"""

        print(prompt)
        print("Ground Truth SQL Query:")
        print(ground_truth)

    elif template_option == 2:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you must create the first table named {source_data_name} with only the given attributes: {source_data_schema}. Please delete the table before creating it if the first table exists. 

        Second, insert the following row(s) into the first table and please don't remove any values:

        {samples}

        Third, you must create a second table named {target_data_name} with only the given attributes: {target_data_schema}. Please delete the table before creating it if the first table exists.

        Finally, insert all rows from the first table into the second table, note that the selection clause in the insert statement should ignore attributes that are not needed.

        Please don't remove the first table, because we need it for validation.

        Please quote the returned SQL script between "```sql\n" and "\n```".

        Some explanation for the first table: {source_data_description}

        Some explanation for the second table: {target_data_description}

        """
    elif template_option == 3:
        prompt = f"""You are a SQL developer. Please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you must create the first table named {source_data_name} with only the given attributes: {source_data_schema}. Please delete the table before creating it if the first table exists. 

        Second, insert the following row(s) into the first table and please don't remove any values:

        {samples}

        Third, you must create a second table named {target_data_name} with only the given attributes: {target_data_schema}. Please delete the table before creating it if the first table exists.

        Finally, insert all rows from the first table into the second table, note that the selection clause in the insert statement should ignore attributes that are not needed.

        Please don't remove the first table, because we need it for validation.

        Please quote the returned SQL script between "```sql\n" and "\n```".

        Some explanation for the first table: {source_data_description}

        Some explanation for the second table: {target_data_description}

        Some hints for the schema changes from the first table to the second table: {schema_change_hints}

        """
    elif template_option == 4:
        prompt = f"""You are a skilled Postgres SQL developer. Let's perform some tasks:
        1. Creating the {source_data_name} Table:
        - Check if a table named {source_data_name} exists. If it does, delete it.
        - Create a new table named {source_data_name}. This table should have exact attributes from the following 
        schema: {source_data_schema}.
        - Note:{source_data_description}
        2. Populating the {source_data_name} Table:
        - Insert the provided rows {samples} into the {source_data_name} table.
        3. Creating the {target_data_name} Table:
        - Check if a table named {target_data_name} exists. If it does, delete it.
        - Create a new table named {target_data_name}. This table should have exact attributes from the following 
        schema:{target_data_schema}.
        - Important: {target_data_description}
        4. Transforming Data from {source_data_name} to {target_data_name}:
        - Write a SQL transformation query to insert all rows from the {source_data_name} table to the {target_data_name} table.
        - Transformation hints: {schema_change_hints}
        Please don't remove the {source_data_name} table, because we need it for validation.
        Please quote the returned SQL script to perform these tasks between "```sql\n" and "\n```".
        """

    print(prompt)
    print("Ground Truth SQL Query:")
    print(ground_truth)

    return prompt, ground_truth, target_data_name

def print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id):
    with open('all_similarity_scores.log', 'a+') as file:
        file.write("Starting with template" + str(template_option)+" ...\n")
        file.write("Scope: target ")
        if target_id == max_target_id:
            file.write("is " + str(target_id))
        else:
            file.write("in [" + str(target_id) + ", " + str(max_target_id) + "]")
        file.write(", source ")
        if source_id == max_source_id:
            file.write("is " + str(source_id))
        else:
            file.write("in [" + str(source_id) + ", " + str(max_source_id) + "]")
        file.write("\n")

# 5 Samples of Source Data: {sammples}
# main script
def main(template_option):
    conn = create_connection()

    json_file_path = 'chatgpt.json'
    target_id = 2
    max_target_id = 2
    source_id = 3
    max_source_id = 3

    # Log the starting of set of experiments
    print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id)

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
                    print("[FAILED] Maximum iterations reached without correct result.")
                    with open('all_similarity_scores.log', 'a+') as file:
                        file.write(target_data_name)
                        file.write("<- ")
                        file.write(source_data_name_to_find)
                        file.write("\t\t\t\t[Failed]\n\tPlease check the similarity scores:\n")
                        for count, iteration_scores in enumerate(all_similarity_scores):
                            file.write("\t\t iter-")
                            file.write(str(count+1))
                            file.write(": ")
                            if iteration_scores[0] == "missmatch":
                                file.write("miss-match: # of rows in result and ground truth\n")
                            else:
                                file.write(", ".join(map(str, iteration_scores)) + "\n")
                        print(accuracy_list)
                        file.write(f"\t\t\t\tCase accuracy: {max(accuracy_list):.2f}\n")
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
                case_accuracy, is_correct, similarity_scores, validation_error = validation(sql_result, ground_truth_sql_result)
                accuracy_list.append(case_accuracy)
                all_similarity_scores.append(similarity_scores)
                print(is_correct)

                if is_correct:
                    print("[Success] Successful SQL execution with correct result.")
                    with open('all_similarity_scores.log', 'a+') as file:
                        file.write(target_data_name)
                        file.write("<- ")
                        file.write(source_data_name_to_find)
                        file.write(" with iter-")
                        file.write(str(iteration_count))
                        file.write("\t\t[Success]\n")
                        # Append the global accuracy to the end
                        #file.write(f", Global accuracy: {case_accuracy:.2f}\n")
                    all_similarity_scores = []
                    break
                else:
                    # to be revised
                    prompt += "The returned SQL script can run, but the execution result of the SQL is wrong: " + str(
                        validation_error) + " Please try again."

                    print(prompt+"\n")
                    continue
        target_id = target_id + 1

    print("All similarity scores saved to all_similarity_scores.log.")
    conn.close()


if __name__ == "__main__":
    template_option = int(input("Choose template option (1/2/3/4): "))
    main(template_option)
