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

openai.api_key = 'xxx'

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
            password="021111",
            host="localhost",  # e.g., "localhost"
            port="5432"   # e.g., "5432"
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

# interact with chatGPT model
def chat_with_gpt(prompt):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=2000,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return str(e)

def calculate_similarity(column_a, column_b, similarity_type="jaccard"):
    if similarity_type == "cosine":
        vectorizer = CountVectorizer().fit_transform(column_a + column_b)
        cosine_sim = cosine_similarity(vectorizer[:len(column_a)], vectorizer[len(column_a):])
        return cosine_sim[0, 0]
    else:  # Jaccard similarity
        intersection = len(set(column_a) & set(column_b))
        union = len(set(column_a) | set(column_b))
        return intersection / union if union != 0 else 0

"""def verify(sql_result, ground_truth):
    if len(sql_result) != len(ground_truth):
        print("Different number of rows in the results and ground truth.")
        return False

    for row_index, (sql_row, truth_row) in enumerate(zip(sql_result, ground_truth)):
        sql_row_array = np.array(sql_row, dtype=object)
        truth_row_array = np.array(truth_row, dtype=object)

        for col_index in range(len(sql_row_array)):
            try:
                sql_val = float(sql_row_array[col_index])
                truth_val = float(truth_row_array[col_index])

                if not np.isclose(sql_val, truth_val):
                    print(f"Row {row_index}, Column {col_index} does not match. SQL Result: {sql_val}, Ground Truth: {truth_val}")
                    return False
            except ValueError:
                # This means that the conversion to float failed, so we skip this column
                continue

    print("Verification passed. The transformation is correct.")
    return True
"""
def validation(sql_result, ground_truth):
    if len(sql_result) != len(ground_truth) or len(sql_result[0]) != len(ground_truth[0]):
        print("Different number of rows or columns in the results and ground truth.")
        return False, []

    # Initialize a list to store similarity scores
    similarity_scores = []

    # Iterate through columns and compare
    for col_index in range(len(sql_result[0])):
        sql_column = [str(row[col_index]) for row in sql_result]
        truth_column = [str(row[col_index]) for row in ground_truth]

        # You can change this to "cosine" to calculate cosine similarity
        similarity_score = calculate_similarity(sql_column, truth_column, similarity_type="jaccard")

        # Append the similarity score for this column
        similarity_scores.append(similarity_score)

    print("Similarity scores for this iteration:", similarity_scores)

    # Returning both the result of strict validation and the similarity scores
    return strict_validation(sql_result, ground_truth), similarity_scores


def strict_validation(sql_result, ground_truth):
    if len(sql_result) != len(ground_truth):
        print("Different number of rows in the results and ground truth.")
        return False

    for row_index, (sql_row, truth_row) in enumerate(zip(sql_result, ground_truth)):
        if sql_row != truth_row:
            print(f"Row {row_index} does not match. SQL Result: {sql_row}, Ground Truth: {truth_row}")
            return False

    print("Verification passed. The transformation is correct.")
    return True

def generate_prompt(json_file_path, template_option,source_data_name_to_find):
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
        prompt = f"""You are a SQL developer. I want you to finish a task, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named SourceSchema {source_data_schema}
        Insert 5 rows into the source table.
        Second, you have to make a second table for the given attributes, named TargetSchema: {target_data_schema}
        Insert all rows from the first table into the second table."""
        print(prompt)
    elif template_option == 2:
        prompt = f"""You are a SQL developer. I want you to finish a task, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named SourceSchema {source_data_schema}. {source_data_description}
        Insert 5 rows into the source table.
        Second, you have to make a second table for the given attributes, named TargetSchema: {target_data_schema}. {target_data_description}
        Schema Change Hints: {schema_change_hints}
        Insert all rows from the first table into the second table."""
    elif template_option == 3:
        prompt = f"""You are a SQL developer. I want you to finish a task, using the || operator for string concatenation, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named {source_data_name} {source_data_schema}. {source_data_description}
        examples of source data: {samples}
        Insert 1 rows into the source table.
        Second, you have to make a second table for the given attributes, named TargetSchema: {target_data_schema}. {target_data_description}
        Schema Change Hints: {schema_change_hints}
        Notes: {notes}
        Insert all rows from the first table into the second table."""

    return prompt,ground_truth

#5 Samples of Source Data: {sammples}
# main script
def main(template_option):
    conn = create_connection()


    json_file_path = 'D:/SQL/ChatGPT Benchmark Datasets.json'
    # Source Data Name to find
    source_data_name_to_find = 'Orange and Rockland'
    # Generate the prompt for the chatGPT model
    if template_option == 1 or template_option == 2 or template_option == 3:
       prompt, ground_truth_query = generate_prompt(json_file_path, template_option, source_data_name_to_find)
    else:
        print("Invalid template option.")
        return
    # Create a list to store similarity scores of each iteration
    all_similarity_scores = []
    iteration_count = 0
    truth_result = execute_sql(conn, ground_truth_query)
    print("TRUTH Result:")
    print(truth_result)
    while True:
        iteration_count += 1
        if iteration_count > 5:
            print("Maximum iterations reached without correct result.")
            break

        gpt_output = chat_with_gpt(prompt)
        print(gpt_output)


        if "Error:" in gpt_output:
            prompt += " GPT Error: " + gpt_output
            break

        # Execute the SQL query on the specified table
        sql_result = execute_sql(conn, gpt_output)
        print("SQL Result:")
        print(sql_result)
        if "Error:" in sql_result:
            prompt += " SQL Error: " + sql_result + " Please try again and show me the SQL query only."
            continue

        is_correct, similarity_scores = validation(sql_result, truth_result)
        all_similarity_scores.append(similarity_scores)
        print(is_correct)

        if is_correct:
            print("Successful SQL execution with correct result.")
            break
        else:
            prompt = "The SQL query can run,  but the result is like the following, which is wrong: " + str(sql_result) + " Please try again and show me the SQL query only."
            continue

    with open('all_similarity_scores.txt', 'w') as file:
        for iteration_scores in all_similarity_scores:
            file.write(", ".join(map(str, iteration_scores)) + "\n")

    print("All similarity scores saved to all_similarity_scores.txt.")
    conn.close()


if __name__ == "__main__":
    template_option = int(input("Choose template option (1/2/3): "))
    main(template_option)
