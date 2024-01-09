from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import psycopg2
import csv
import re
import logging
import pandas as pd
import datetime

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
        password="1234",
        host="localhost",  # e.g., "localhost"
        port="5432"  # e.g., "5432"
    )
    return conn

def extract_target_table_name(sql_query):
    # Match the target table name after the third "CREATE TABLE" statement
    create_matches = re.findall(r'CREATE\s+TABLE\s+(\w+)', sql_query, re.IGNORECASE)

    if len(create_matches) >= 3:
        return create_matches[2]  # Return the third occurrence of CREATE TABLE

    # Match the target table name after the "INSERT INTO" statement
    insert_match = re.search(r'(CREATE\s+TABLE\s+\w+.*?){3}|INSERT\s+INTO\s+(\w+)', sql_query, re.IGNORECASE)

    if insert_match:
        # If the match is from the "CREATE TABLE" group, return the captured group (table name) after CREATE TABLE
        if insert_match.group(1):
            create_table_match = re.search(r'CREATE\s+TABLE\s+(\w+)', insert_match.group(1), re.IGNORECASE)
            if create_table_match:
                return create_table_match.group(1)

        # If the match is from the "INSERT INTO" group, return the captured group (table name) after INSERT INTO
        elif insert_match.group(2):
            return insert_match.group(2)    

def execute_sql(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN;")
        cursor.execute(query)
        # Assuming you want to commit after every SQL execution for simplicity
        conn.commit()

        # Check if the operation is not a SELECT statement
        if not query.strip().upper().startswith("SELECT"):
            logging.info(f"query to be parsed {query}")
            target_table = extract_target_table_name(query)
            if target_table:
                # Fetch results from the last inserted table
                logging.info(f"final name of target table {target_table}")
                cursor.execute(f"SELECT * FROM {target_table};")
                result = cursor.fetchall()
                #logging.info(f"good target table result {result}")
            else:
                result = "Table name not identified from last INSERT INTO query."
                logging.info(f"bad target table result {result}")
        else:
            result = cursor.fetchall()

        return result
    except psycopg2.Error as e:
        conn.rollback()  # Rollback the transaction on error
        return f"Error: {e.pgerror}"

def print_experiment_settings(template_option, target_id, max_target_id, source_id, max_source_id):
    with open('all_similarity_scores.log', 'a+') as file:
        file.write("Starting with template" + str(template_option) + " ...\n")
        file.write("Scope: target ")
        if target_id == max_target_id:
            file.write(f"is {target_id}")
        else:
            file.write(f"in [{target_id}, {max_target_id}]")
        file.write(", source ")
        if source_id == max_source_id:
            file.write(f"is {source_id}")
        else:
            file.write(f"in [{source_id}, {max_source_id}]")
        file.write("\n")


def log_experiment_failed(target_data_name, source_data_name_to_find, iteration_count, all_similarity_scores,
                            accuracy_list):
    print("[FAILED] Maximum iterations reached without correct result.")
    with open('all_similarity_scores.log', 'a+') as file:
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
    with open('all_similarity_scores.log', 'a+') as file:
        file.write(f"{target_data_name} <- {source_data_name_to_find} with iter-{iteration_count}\t\t[Success]\n")
        # Append the global accuracy to the end
        # file.write(f", Global accuracy: {case_accuracy:.2f}\n")

def numerical_similarity(value1, value2, threshold=1e-10):
    """ Calculate numerical similarity between two values. """
    if pd.isna(value1) and pd.isna(value2):
        return 1.0

    try:
        if isinstance(value1, pd.Series):
            value1 = value1.iloc[0]  # Extract the first element if it's a Series

        if isinstance(value2, pd.Series):
            value2 = value2.iloc[0]  # Extract the first element if it's a Series

        numeric_value1 = pd.to_numeric(value1, errors='coerce')
        numeric_value2 = pd.to_numeric(value2, errors='coerce')

        if pd.api.types.is_integer_dtype(numeric_value1) or pd.api.types.is_float_dtype(numeric_value1):
            float_value1 = float(numeric_value1)
        else:
            return 0.0

        if pd.api.types.is_integer_dtype(numeric_value2) or pd.api.types.is_float_dtype(numeric_value2):
            float_value2 = float(numeric_value2)
        else:
            return 0.0

        return 1.0 if abs(float_value1 - float_value2) <= threshold else 0.0
    except (ValueError, TypeError):
        return 0.0

def object_similarity(obj1, obj2, threshold=1e-10):
    """ Calculate similarity between two objects. """
    if obj1 == obj2:
        return 1.0

    try:
        # Attempt to convert objects to float
        float_value1 = float(obj1)
        float_value2 = float(obj2)

        # Check numerical similarity
        return 1.0 if abs(float_value1 - float_value2) <= threshold else 0.0

    except (ValueError, TypeError):
        # If conversion to float fails, check if the objects are equal
        return 1.0 if obj1 == obj2 else 0.0
    
def calculate_similarity(column_a, column_b, similarity_type="numerical", threshold=1e-10):
    """ Calculate similarity between two columns based on specified similarity type. """
    if similarity_type == "numerical":
        scores = [numerical_similarity(val1, val2, threshold) for val1, val2 in zip(column_a, column_b)]
        return sum(scores) / len(scores)
    elif similarity_type == "jaccard":
        intersection = len(set(column_a) & set(column_b))
        union = len(set(column_a) | set(column_b))
        return intersection / union if union else 0
    elif similarity_type == "object":
        scores = [object_similarity(val1, val2, threshold) for val1, val2 in zip(column_a, column_b)]
        return sum(scores) / len(scores)
    else: # Not used in the current version
        vectorizer = CountVectorizer().fit_transform(column_a + column_b)
        return cosine_similarity(vectorizer[:len(column_a)], vectorizer[len(column_a):])[0, 0]


