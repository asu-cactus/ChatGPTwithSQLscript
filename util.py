from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import psycopg2
import csv
import re
import logging
import json

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
        password="021111",
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
    insert_match = re.search(r'INSERT\s+INTO\s+(\w+)', sql_query, re.IGNORECASE)

    if insert_match:
        return insert_match.group(1)  # Return the captured group (table name) after INSERT INTO

    return None  # Return None if no match is found


def execute_sql(conn, query):
    cursor = conn.cursor()
    try:
        cursor.execute("BEGIN;")
        cursor.execute(query)
        # Assuming you want to commit after every SQL execution for simplicity
        conn.commit()

        # Check if the operation is not a SELECT statement
        if not query.strip().upper().startswith("SELECT"):
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
    if value1 in (0.0, None) and value2 in (0.0, None):
        return 1.0
    try:
        float_value1 = float(set(value1))
        float_value2 = float(set(value2))
        return 1.0 if abs(float_value1 - float_value2) <= threshold else 0.0
    except (ValueError, TypeError):
        return 0.0


def calculate_similarity(column_a, column_b, similarity_type="numerical", threshold=1e-10):
    """ Calculate similarity between two columns based on specified similarity type. """
    if similarity_type == "numerical":
        scores = [numerical_similarity(val1, val2, threshold) for val1, val2 in zip(column_a, column_b)]
        return sum(scores) / len(scores)
    elif similarity_type == "jaccard":
        intersection = len(set(column_a) & set(column_b))
        union = len(set(column_a) | set(column_b))
        return intersection / union if union else 0
    else: # Not used in the current version
        vectorizer = CountVectorizer().fit_transform(column_a + column_b)
        return cosine_similarity(vectorizer[:len(column_a)], vectorizer[len(column_a):])[0, 0]

def extract_source_table(source_data_name_to_find):
    sql_query = f"""select * from {source_data_name_to_find}"""

    return sql_query

def generate_information(json_file_path, source_data_name_to_find):
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

    return target_data_schema,source_data_schema,samples

def extract_table_schemas(sql_query,source_data_name_to_find,target_data_name):
    source_schema = {}
    target_schema = {}

    # Remove lines that are comments
    sql_query = '\n'.join([line for line in sql_query.split('\n') if not line.strip().startswith('--')]).strip()

    # Split SQL queries based on semicolon and filter out empty strings
    queries = [q.strip() for q in sql_query.split(";") if q.strip()]

    for query in queries:
        # Match CREATE TABLE statements
        match = re.match(r"CREATE TABLE (\w+) \((.+)\)", query.replace('\n', ' '), re.IGNORECASE)

        if match:
            table_name = match.group(1)
            columns_str = match.group(2)
            # This regular expression ensures that columns with spaces are captured correctly
            columns_list = re.findall(r'(?:(\w+)|"([^"]+)")\s+([\w()]+)', columns_str)

            # Extract column names and types
            column_schema = {}
            for col in columns_list:
                col_name = col[1] if col[1] else col[0]
                col_type = col[2].upper()
                column_schema[col_name] = col_type

            if table_name == source_data_name_to_find:
                source_schema = column_schema
            elif table_name == target_data_name:
                target_schema = column_schema

    return source_schema, target_schema

# def parse_schema_to_columns(data_schema):
#     if ',' in data_schema:  # For target_data_schema
#         return [re.split("\s+", x.strip()) for x in data_schema.split(",")]
#     else:  # For source_data_schema
#         return [x for x in data_schema.split() if x]

def parse_schema_to_columns(data_schema):
    # This regex will match both quoted and unquoted column names
    # It captures everything inside the quotes for quoted column names
    # and everything up to the first space (or end of string if no space) for unquoted column names
    matches = re.findall(r'"([^"]+)"\s*[^,]*,|\s*([^,\s]+)\s[^,]*,', data_schema + ',')

    # Process matches to extract column names
    parsed_columns = [' '.join(filter(None, match)).strip() for match in matches]

    return parsed_columns


def extract_agg_mapping_operator(gpt_output):
    # Regular expression for extracting mappings
    mapping_regex = r'Mapping\[(.+?),\s*(".*?"|\w+)\]'
    mapping_matches = re.findall(mapping_regex, gpt_output)

    # Regular expression for extracting aggregation functions
    agg_regex = r'Aggregation\[(.+?),\s*(".+?"|\w+),\s*(\w+)\]'
    agg_matches = re.findall(agg_regex, gpt_output)

    # Regular expression for extracting operators
    operator_regex = r'Existing operator: \((.+)\)'
    operator_matches = re.search(operator_regex, gpt_output)

    # Extracted mappings
    mappings = []
    for match in mapping_matches:
        source, target = match
        mappings.append((source.strip(), target.strip().replace('"', '')))  # Remove quotes from target

    # Extracted aggregation, if any
    aggs = []
    for match in agg_matches:
        column, agg_type = match[:2]
        aggs.append((column.strip(), agg_type.strip().replace('"', '')))

    # Extracted operators
    operators = set()
    if operator_matches:
        # Split the matched string by comma and strip spaces and quotes
        operators = set(op.strip().strip("'") for op in operator_matches.group(1).split(','))

    return aggs, mappings, operators


