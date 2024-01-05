import os
import logging
import pandas as pd
import re
import csv

def convert_target_names(target_names_str):
    target_names = target_names_str.split(',')
    converted_names = []

    for target_name in target_names:
        match = re.match(r'^Target(\d+)_(\d+)$', target_name.strip())
        if match:
            number1, number2 = match.groups()
            converted_name = f"length{number1}_{number2}"
            converted_names.append(converted_name)
        else:
            converted_names.append(target_name)

    converted_names_str = ', '.join(converted_names)
    return converted_names_str

def access_auto_pipeline_dataset(sub_folder_name):
    main_folder_name = "github-pipelines"
    main_folder_name = os.path.abspath(main_folder_name)
    sub_folder = f"{main_folder_name}\{sub_folder_name}\\"
    test_0 = f"{sub_folder}test_0.csv"
    test_1 = f"{sub_folder}test_1.csv"
    target = f"{sub_folder}target.csv"
    return main_folder_name,sub_folder, test_0, test_1, target

def read_csv_target(target):
    gold_target = []
    logging.info(f"Final target path{target}")
    with open(target, 'r',encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                gold_target.append(tuple(row))
    return gold_target


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import psycopg2
import csv
import re


def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data


def create_connection():
    """ create a database connection to the PostgreSQL database """
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="postgres",
        host="localhost",  # e.g., "localhost"
        port="5432"  # e.g., "5432"
    )
    return conn


def extract_last_insert_table_name(query):
    """
    Extracts the table name from the last INSERT INTO clause in the given SQL query.
    """
    matches = re.findall(r"INSERT\s+INTO\s+(\"[^\"]+\"|\w+)", query, re.IGNORECASE)
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


# def create_table(conn, create_statement):
#     print(create_statement)
#     cursor = conn.cursor()
#     try:
#         cursor.execute("BEGIN;")
#         cursor.execute(create_statement)
#         # Assuming you want to commit after every SQL execution for simplicity
#         conn.commit()
#     except psycopg2.Error as e:
#         conn.rollback()  # Rollback the transaction on error
#         return f"Error: {e.pgerror}"


def print_experiment_settings(len_id, max_len_id, target_id, max_target_id, clarify_on):
    with open('log/all_similarity_scores.log', 'a+') as file:
        file.write(f"{'[Clarify On]' if clarify_on else '[Clarify Off]'}\n")
        file.write("Scope: length ")
        if len_id == max_len_id:
            file.write(f"is {len_id}")
        else:
            file.write(f"in [{len_id}, {max_len_id}]")
        file.write(", target ")
        if target_id == max_target_id:
            file.write(f"is {target_id}")
        else:
            file.write(f"in [{target_id}, {max_target_id}]")
        file.write("\n")


def log_experiment_failed(target_data_name, source_data_name_to_find, iteration_count, all_similarity_scores,
                          accuracy_list):
    print("[FAILED] Maximum iterations reached without correct result.")
    with open('log/all_similarity_scores.log', 'a+') as file:
        file.write(f"{target_data_name} <- {source_data_name_to_find}")
        file.write("\t\t\t\t[Failed]\n\tPlease check the similarity scores:\n")
        for count, iteration_scores in enumerate(all_similarity_scores):
            file.write(f"\t\t iter-{count + 1}: ")
            if iteration_scores[0] == "mismatch":
                file.write("mis-match: # of rows in result and ground truth\n")
            else:
                file.write(", ".join(map(str, iteration_scores)) + "\n")
        print(accuracy_list)
        file.write(f"\t\t\t\tCase accuracy: {max(accuracy_list):.2f}\n")


def log_experiment_success(target_data_name, source_data_name_to_find, iteration_count):
    print("[Success] Successful SQL execution with correct result.")
    with open('log/all_similarity_scores.log', 'a+') as file:
        file.write(f"{target_data_name} <- {source_data_name_to_find} with iter-{iteration_count}\t\t[Success]\n")
        # Append the global accuracy to the end
        # file.write(f", Global accuracy: {case_accuracy:.2f}\n")


def numerical_similarity(value1, value2, threshold=1e-10):
    """ Calculate numerical similarity between two values. """
    if value1 in (0.0, None) and value2 in (0.0, None):
        return 1.0
    return 1.0 if abs(float(value1) - float(value2)) <= threshold else 0.0


def calculate_similarity(column_a, column_b, similarity_type="numerical", threshold=1e-10):
    """ Calculate similarity between two columns based on specified similarity type. """
    if similarity_type == "numerical":
        scores = [numerical_similarity(val1, val2, threshold) for val1, val2 in zip(column_a, column_b)]
        return sum(scores) / len(scores)
    elif similarity_type == "jaccard":
        intersection = len(set(column_a) & set(column_b))
        union = len(set(column_a) | set(column_b))
        return intersection / union if union else 0
    else:  # Not used in the current version
        vectorizer = CountVectorizer().fit_transform(column_a + column_b)
        return cosine_similarity(vectorizer[:len(column_a)], vectorizer[len(column_a):])[0, 0]


