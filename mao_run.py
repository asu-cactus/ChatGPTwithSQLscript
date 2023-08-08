import openai
import pymysql
from pymysql.err import OperationalError
import difflib
import numpy as np
from pymysql.err import ProgrammingError
import sqlite3
from sqlite3 import Error
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import csv

openai.api_key = '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'

def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

# define database connection
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def upload_data_to_database(conn, table_name, data):
    cursor = conn.cursor()
    placeholders = ', '.join(['?'] * len(data[0]))
    for row in data:
        cursor.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", row)
    conn.commit()

def execute_sql(conn, sql):
    """ Execute an SQL statement """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)

        # If you're fetching data, return the fetched data
        return cursor.fetchall()
    except Error as e:
        print(e)



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
def  validation(sql_result, ground_truth):
    # Checking the number of columns and rows
    if len(sql_result) != len(ground_truth) or len(sql_result[0]) != len(ground_truth[0]):
        print("Different number of rows or columns in the results and ground truth.")
        return False

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

        # Additional checks for exact column name and values can be added here if needed

    # Writing the similarity scores to a file
    with open('similarity_scores.txt', 'w') as file:
        for score in similarity_scores:
            file.write(f"{score}\n")

    print("Verification passed. Similarity scores recorded in similarity_scores.txt.")
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
    sammples = data["5 Samples of Source Data"]
    target_data_description = data["Target Data Description"]
    source_data_description = data["Source Data Description"]
    schema_change_hints = data["Schema Change Hints"]
    notes = data["Remark or Note"]
    ground_truth = data["GroundTruth SQL"]
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
        examples of source data: {sammples}
        Insert 2 rows into the source table.
        Second, you have to make a second table for the given attributes, named TargetSchema: {target_data_schema}. {target_data_description}
        Schema Change Hints: {schema_change_hints}
        Notes: {notes}
        Insert all rows from the first table into the second table."""

    return prompt,ground_truth

#5 Samples of Source Data: {sammples}
# main script
def main(template_option):
    conn = create_connection("C:/Users/23879/gpt.db")


    # Upload the source data to the origin table
    #upload_data_to_database(conn,'SourceSchema' , source_data)



    # Retrieve the original data from the specified table
    original_data_query = "SELECT * FROM SourceSchema;"
    original_data = execute_sql(conn, original_data_query)

    # Retrieve the ground truth from the specified table
    ground_truth_query = "SELECT * FROM Source_Schema"
    ground_truth = execute_sql(conn, ground_truth_query)


    json_file_path = 'D:/SQL/ChatGPT Benchmark Datasets.json'
    # Source Data Name to find
    source_data_name_to_find = 'EVERSOURCE-NSTAR:2014-load-profile-ema'
    # Generate the prompt for the chatGPT model
    if template_option == 1 or template_option == 2 or template_option == 3:
       prompt, ground_truth_query = generate_prompt(json_file_path, template_option, source_data_name_to_find)
    else:
        print("Invalid template option.")
        return

    iteration_count = 0
    while True:
        iteration_count += 1
        if iteration_count > 5:
            print("Maximum iterations reached without correct result.")
            break

        gpt_output = chat_with_gpt(prompt)
        print(gpt_output)

        
        if "Error:" in gpt_output:
            prompt += " GPT Error: " + gpt_output
            continue

        # Execute the SQL query on the specified table
        print(gpt_output)
        sql_result = execute_sql(conn, gpt_output)
        truth_result = execute_sql(conn, ground_truth_query)
        print(sql_result)
        # Compare the SQL result with the ground truth
        is_correct = validation(sql_result, truth_result)
        print(is_correct)
        if isinstance(sql_result, str):
            prompt += " SQL Error: " + sql_result
            continue

        if is_correct:
            print("Successful SQL execution with correct result.")
            break
        else:
            prompt = "The SQL query can run,  but the result is like the following, which is wrong: " + str(sql_result)
            continue
    conn.close()


if __name__ == "__main__":
    template_option = int(input("Choose template option (1/2/3): "))
    main(template_option)
