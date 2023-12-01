from util import ( execute_sql, print_experiment_settings,
                   log_experiment_success, log_experiment_failed,
                   calculate_similarity,extract_source_table,generate_information,extract_table_schemas,parse_schema_to_columns)
from gpt import generate_prompt, chat_with_gpt
import sqlparse
from sqlparse.sql import Identifier, IdentifierList,Function
from sqlparse.tokens import DML,Keyword
from datetime import datetime
import decimal


def reverse_quality(json_file_path, gpt_output, sql_result, source_data_name_to_find, accuracy_list, conn,
                    all_similarity_scores):

    prompt, gt, td = generate_prompt(json_file_path, 8, gpt_output, source_data_name_to_find)
    print("prompt:", prompt)

    gpt_output = chat_with_gpt(prompt,True)
    gpt_output_source = extract_source_table(source_data_name_to_find)
    print("gpt_output:", gpt_output)
    print("gpt_output_source:", gpt_output_source)
    sql_result_reverse = execute_sql(conn, gpt_output)
    if "Error:" in sql_result_reverse:
        validation_error = "\n Error in the previous response:" + sql_result_reverse
        accuracy_list.append(0.0)
        reverse_score = 0
        print("error when reverse,the score is :", reverse_score)
        return reverse_score
    sql_ini = execute_sql(conn, gpt_output_source)
    print("sql_ini", sql_ini)
    print("sql_result_reverse", sql_result_reverse)
    # Validate the ChatGPT generated SQL script
    case_accuracy, is_correct, similarity_scores, validation_error, reverse_score = validation(sql_ini,
                                                                                               sql_result_reverse)
    accuracy_list.append(case_accuracy)
    all_similarity_scores.append(similarity_scores)
    print(is_correct)
    print("reverse_score:", reverse_score)
    return reverse_score

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

def schema_quality(sql_query, source_data_name_to_find,target_data_name,json_file_path):
    """
    Calculate the quality score of a SQL query based on static analysis.
    """
    # Extract the columns involved in the query
    #select_columns, target_columns = parse_complex_sql_script(sql_query)
    # print("select_columns", select_columns)
    # print("target_columns", target_columns)
    target_data_schema, source_data_schema, samples = generate_information(json_file_path, source_data_name_to_find)
    select_columns,target_columns = extract_table_schemas(sql_query,source_data_name_to_find,target_data_name)
    print("select_columns", select_columns)
    print("target_columns", target_columns)
    print("source_data_schema:", source_data_schema)
    print("target_data_schema:", target_data_schema)

    target_data_columns = parse_schema_to_columns(target_data_schema)
    source_data_columns = parse_schema_to_columns(source_data_schema)
    print("Parsed Target Schema Columns:", target_data_columns)
    print("Parsed Source Schema Columns:", source_data_columns)

    # Clean up column names from the SQL query (removing quotes)
    target_columns_clean = [x.replace('"', '') for x in target_columns]
    select_columns_clean = [x.split(" ")[-1].replace('"', '') if " AS " in x else x for x in
                            select_columns]  # Extract alias if exists
    print("Target Schema:", target_columns_clean)
    print("Source Schema:",select_columns_clean)

    # Count the number of columns that match in the target and source schema
    matching_target_count = len([col for col in target_columns_clean if col in target_data_columns])
    matching_source_count = len([col for col in select_columns_clean if col in source_data_columns])
    print("Matching Target Columns:", matching_target_count)
    print("Matching Source Columns:", matching_source_count)

    mismatch_1 = [col for col in target_columns_clean if col not in target_data_columns]
    mismatch_2 = [col for col in select_columns_clean if col not in source_data_columns]

    mismatch_feedback = "\n Mismatch in target column:" + str(mismatch_1) + "\n Mismatch in Source column:" + str(mismatch_2)

    # Calculate the individual scores
    score_3_2 = matching_target_count / len(target_data_columns) if len(target_data_columns) != 0 else 0

    # Calculate the individual scores
    score_3_1 = matching_source_count / len(source_data_columns) if len(source_data_columns) != 0 else 0

    print("score_3_1",score_3_1)
    print("score_3_2",score_3_2)
    # Weights for each score
    weight_1 = 0.4
    weight_2 = 0.6

    # Calculate the final quality score
    quality_score = (weight_1 * score_3_1) + (weight_2 * score_3_2)

    return quality_score,mismatch_feedback


def get_numeric_columns_from_dict(columns_dict):
    numeric_types = ['NUMERIC', 'INT', 'FLOAT']
    return [column for column, data_type in columns_dict.items() if data_type.upper() in numeric_types]


def column_exists(conn, table_name, column_name):
    cur = conn.cursor()
    try:
        cur.execute(f"SELECT {column_name} FROM {table_name} LIMIT 1;")
        cur.close()  # Close the cursor
        return True
    except:
        conn.rollback()  # Rollback the current transaction on error
        cur.close()  # Close the cursor
        return False


def extract_insert_select_query(sql_query):
    parsed_statements = sqlparse.parse(sql_query)

    # Extract statements related to the target table
    target_statements = []
    target_table_name = None

    for statement in parsed_statements:
        statement_str = str(statement)

        # Find the target table name after the DROP statement
        if "-- Drop the second table if it exists" in statement_str:
            target_table_name = statement_str.split("DROP TABLE IF EXISTS")[1].split(";")[0].strip()

        if target_table_name:
            if target_table_name in statement_str:
                target_statements.append(statement_str)

                if f"INSERT INTO {target_table_name}" in statement_str:
                    break

    return "\n".join(target_statements)


def source_diff(numeric_cols, conn, table_name, num):
    cur = conn.cursor()

    # Determine whether to use the row with the min or max value for numeric columns
    order_by_col = numeric_cols[0]
    order_by_direction = "ASC" if num <= 1 else "DESC"
    cur.execute(f"SELECT * FROM {table_name} ORDER BY \"{order_by_col}\" {order_by_direction} LIMIT 1;")
    row_to_copy = cur.fetchone()

    # Fetch column names from the table
    cur.execute(f"SELECT * FROM {table_name} LIMIT 0;")
    col_names = [desc[0] for desc in cur.description]

    # Prepare the new values, updating only the numeric columns
    new_values = list(row_to_copy)
    for i, col in enumerate(col_names):
        if col in numeric_cols:
            if num <= 1:
                cur.execute(f"SELECT MIN(\"{col}\") FROM {table_name};")
            else:
                cur.execute(f"SELECT MAX(\"{col}\") FROM {table_name};")
            val = cur.fetchone()[0]
            if val is not None:  # Only multiply if value is not NULL
                new_values[i] = num * float(val)

    # Format values for SQL
    formatted_values = []
    for value in new_values:
        if isinstance(value, datetime):
            formatted_values.append(f"'{value.strftime('%Y-%m-%d %H:%M:%S')}'")
        elif isinstance(value, (int, float, decimal.Decimal)):
            formatted_values.append(str(value))
        elif value is None:
            formatted_values.append("NULL")
        else:
            formatted_values.append(f"'{value}'")

    # Convert the row into a string format
    formatted_row = ", ".join(formatted_values)

    # Create a new table name
    new_table_name = f"{table_name}_{str(num).replace('.', '_')}"

    # Clone the existing table structure with no data
    cur.execute(f"CREATE TABLE IF NOT EXISTS {new_table_name} AS TABLE {table_name} WITH DATA;")

    # Insert the new row
    cur.execute(f"INSERT INTO {new_table_name} VALUES ({formatted_row});")

    conn.commit()
    cur.close()

    return new_table_name


def query_name_change(sql_query,table_name,new_table_name):
    # Adjust the table name to have the "_num" format based on the param
    return sql_query.replace(table_name, new_table_name)


def determine_aggregation(val_1, val_2, val_3, val_01, val_10):
    aggregation_type = None

    # Find the columns affected for val_2 and val_3
    affected_columns_1 = [col for col, (val_before, val_after) in zip(val_1.keys(), zip(val_1.values(), val_2.values())) if val_before != val_after]
    affected_columns_2 = [col for col, (val_before, val_after) in zip(val_1.keys(), zip(val_1.values(), val_3.values())) if val_before != val_after]

    # Check for min aggregation
    if all(val_2[col] == val_01[col] for col in affected_columns_1) and all(val_3[col] == val_1[col]  for col in affected_columns_2):
        aggregation_type = "min"

    # Check for max aggregation
    elif all(val_2[col] == val_1[col] for col in affected_columns_1) and all(val_3[col] == val_10[col] for col in affected_columns_2):
        aggregation_type = "max"

    # Check for sum aggregation
    elif all(val_2[col] > val_1[col] for col in affected_columns_1) and all(val_3[col] > val_10[col] for col in affected_columns_2):
        aggregation_type = "sum"

    # Check for avg aggregation
    elif all(val_2[col] < val_1[col] for col in affected_columns_1) and all(val_3[col] > val_1[col] for col in affected_columns_2):
        aggregation_type = "avg"

    return aggregation_type


def differential_quality(sql_query, conn, table_name, target_table_name):
    cur = conn.cursor()
    print("differential:")
    source_columns, target_columns = extract_table_schemas(sql_query, table_name, target_table_name)
    # Step 1: Find numerical attributes from the target_columns dictionary
    numerical_attributes = get_numeric_columns_from_dict(target_columns)
    print("numerical_attributes:", numerical_attributes)
    val_1 = {}
    val_2 = {}
    val_3 = {}

    # Step 2: Calculate val_1 for Target table
    try:
        with conn.cursor() as cur:  # Using a context manager to automatically close the cursor
            for column in numerical_attributes:
                if column_exists(conn, table_name, f"\"{column}\""):
                    col_name = f"\"{column}\""
                else:
                    col_name = column
                cur.execute(f"SELECT \"{col_name}\" FROM {target_table_name};")
                val_1[column] = cur.fetchone()[0]
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error in calculating val_1: {e}")

    print("val1.items():", val_1.items())
    val_01 = {key: (val / 10 if val is not None else None) for key, val in val_1.items()}
    val_10 = {key: (val * 10 if val is not None else None) for key, val in val_1.items()}
    print("val01.items():", val_01.items())
    print("val10.items():", val_10.items())

    # Step 4: Calculate the new output value val_2 and val_3 for the Target table
    # val_2
    try:
        with conn.cursor() as cur:
            # Detect the source_numeric_column automatically
            source_numeric_columns = get_numeric_columns_from_dict(source_columns)

            new_table_name = source_diff(source_numeric_columns, conn, table_name, 0.1)
            print("new_table_name",new_table_name)
            cur.execute(f"SELECT * FROM {new_table_name};")
            new_source = cur.fetchall()
            print("new_source_table:", new_source)
            sql_query_1 = query_name_change(sql_query, table_name,new_table_name)
            sql_insert = extract_insert_select_query(sql_query_1)
            #print("sql_insert:", sql_insert)
            execute_sql(conn, sql_insert)
            cur.execute(f"SELECT * FROM {target_table_name};")
            new_target = cur.fetchall()
            print("new_target_table:", new_target)

            for column in numerical_attributes:
                if column_exists(conn, table_name, f"\"{column}\""):
                    col_name = f"\"{column}\""
                else:
                    col_name = column
                cur.execute(f"SELECT \"{col_name}\" FROM {target_table_name};")
                val_2[column] = cur.fetchone()[0]
            print("val2.items():", val_2.items())
            execute_sql(conn,f"DROP TABLE IF EXISTS {new_table_name}")
    except Exception as e:
        conn.rollback()
        print(f"Error in calculating val_2: {e}")

    # val_3
    try:
        with conn.cursor() as cur:
            new_table_name = source_diff(source_numeric_columns, conn, table_name, 10)
            print("new_table_name", new_table_name)
            cur.execute(f"SELECT * FROM {new_table_name};")
            new_source = cur.fetchall()
            print("new_source_table:",new_source)
            sql_query_2 =query_name_change(sql_query,table_name,new_table_name)
            sql_insert = extract_insert_select_query(sql_query_2)
            #print("sql_insert:", sql_insert)
            execute_sql(conn, sql_insert)
            cur.execute(f"SELECT * FROM {target_table_name};")
            new_target = cur.fetchall()
            print("new_target_table:", new_target)
            for column in numerical_attributes:
                if column_exists(conn, table_name, f"\"{column}\""):
                    col_name = f"\"{column}\""
                else:
                    col_name = column
                cur.execute(f"SELECT \"{col_name}\" FROM {target_table_name};")
                val_3[column] = cur.fetchone()[0]
            print("val3.items():", val_3.items())
            execute_sql(conn, f"DROP TABLE IF EXISTS {new_table_name}")
    except Exception as e:
        conn.rollback()
        print(f"Error in calculating val_3: {e}")

    # Compare val_1,val_2,_val_3
    aggregation_detected = determine_aggregation(val_1, val_2, val_3, val_01, val_10)
    print(f"Detected aggregation: {aggregation_detected}")
    cur.close()
    return 1,aggregation_detected


def fd_quality(conn,gpt_output, source_data_name_to_find, target_data_name):
    pass


def extract_source_column_from_token(token, source_columns_clean):
    extracted_columns = []
    if isinstance(token, Identifier) and token.get_real_name() in source_columns_clean:
        extracted_columns.append(token.get_real_name())
    elif hasattr(token, 'tokens'):
        for subtoken in token.tokens:
            extracted_columns.extend(extract_source_column_from_token(subtoken, source_columns_clean))
    return extracted_columns



def extract_elements(sql_query):
    parsed = sqlparse.parse(sql_query)
    elements = {
        'group_by': [],
        'to_char': [],
        'max': [],
        'min': [],
        'sum': [],
        'avg': [],
        'case_statements': [],
        'extract': [],
        'greatest': [],
        'least': []
    }

    for statement in parsed:
        is_select_context = False
        for token in statement.flatten():
            if token.ttype is DML and token.value.upper() == 'SELECT':
                is_select_context = True
            elif token.ttype is Keyword and token.value.upper() == 'GROUP BY':
                elements['group_by'].append(token)

            if is_select_context:
                upper_value = token.value.upper()
                if upper_value == 'TO_CHAR':
                    elements['to_char'].append(token)
                if upper_value == 'MAX':
                    elements['max'].append(token)
                if upper_value == 'MIN':
                    elements['min'].append(token)
                if upper_value == 'SUM':
                    elements['sum'].append(token)
                if upper_value == 'AVG':
                    elements['avg'].append(token)
                if upper_value.startswith('CASE'):
                    elements['case_statements'].append(token)
                if upper_value == 'EXTRACT':
                    elements['extract'].append(token)
                if upper_value == 'GREATEST':
                    elements['greatest'].append(token)
                if upper_value == 'LEAST':
                    elements['least'].append(token)

    return elements

def calculate_mapping_score_and_mismatches(gpt_mapping, column_mappings):
    correct_mappings = 0
    mismatches = []

    # Check each gpt_mapping against the actual column_mappings
    for gpt_map in gpt_mapping:
        if gpt_map in column_mappings:
            correct_mappings += 1
        else:
            mismatches.append(gpt_map)

    # Calculate the score based on the number of correct mappings
    total_mappings = len(gpt_mapping)
    score = (correct_mappings / total_mappings) * 100 if total_mappings > 0 else 0
    return score, mismatches

def mapping_quality(gpt_output, source_data_name_to_find, target_data_name):
    source_columns, target_columns = extract_table_schemas(gpt_output, source_data_name_to_find, target_data_name)
    # Clean up column names from the SQL query (removing quotes)
    target_columns_clean = [x.replace('"', '') for x in target_columns]
    source_columns_clean = [x.split(" ")[-1].replace('"', '') if " AS " in x else x for x in source_columns]
    print("target_columns", target_columns_clean)
    print("source_columns", source_columns_clean)

    # Parse the SQL query
    parsed_statements = sqlparse.parse(gpt_output)
    column_mappings = []

    for statement in parsed_statements:
        if statement.get_type() == "INSERT":
            for token in statement.tokens:
                if token.ttype is DML and token.value.upper() == "SELECT":
                    select_part = token.parent
                    identifiers = [item for item in select_part.tokens if isinstance(item, IdentifierList)]
                    if identifiers:
                        identifier_list = identifiers[0].get_identifiers()
                        for idx, identifier in enumerate(identifier_list):
                            if idx < len(target_columns_clean):  # Check if idx is within the range
                                target_col = target_columns_clean[idx]
                                extracted_columns = extract_source_column_from_token(identifier, source_columns_clean)
                                for col in extracted_columns:
                                    column_mappings.append((col, target_col))
                            else:
                                print(f"Index {idx} is out of range for target columns.")

    gpt_mapping = [('Date','CST'),('1:00 AM', '1:00'), ('2:00 AM', '2:00'), ('3:00 AM', '3:00'), ('4:00 AM', '4:00'), ('5:00 AM', '5:00'),
             ('6:00 AM', '6:00'), ('7:00 AM', '7:00'), ('8:00 AM', '8:00'), ('9:00 AM', '9:00'), ('10:00 AM', '10:00'),
             ('11:00 AM', '11:00'), ('12:00 AM', '12:00'), ('1:00 PM', '13:00'), ('2:00 PM', '14:00'),
             ('3:00 PM', '15:00'), ('4:00 PM', '16:00'), ('5:00 PM', '17:00'), ('6:00 PM', '18:00'),
             ('7:00 PM', '19:00'), ('8:00 PM', '20:00'), ('9:00 PM', '21:00'), ('10:00 PM', '22:00'),
             ('11:00 PM', '23:00'), ('12:00 PM', '24:00')]

    mapping_score,mismatch = calculate_mapping_score_and_mismatches(gpt_mapping, column_mappings)
    print("mapping score:",mapping_score)
    operator_1 = extract_elements(gpt_output)
    print("column_mapping result:",column_mappings)
    print("Existing operator:",operator_1)
    return mapping_score,mismatch



def information_case(mapping_feedback,gpt_output,source_data_name_to_find, target_data_name):
    source_columns, target_columns = extract_table_schemas(gpt_output, source_data_name_to_find, target_data_name)
    # Clean up column names from the SQL query (removing quotes)
    target_columns_clean = [x.replace('"', '') for x in target_columns]
    source_columns_clean = [x.split(" ")[-1].replace('"', '') if " AS " in x else x for x in source_columns]
    print("target_columns", target_columns_clean)
    print("source_columns", source_columns_clean)



if __name__ == "__main__":

   gpt_output =  """-- Drop the first table if it exists
DROP TABLE IF EXISTS Source1_8;

-- Create the first table
CREATE TABLE Source1_8 (
    datetime TIMESTAMP,
    cerc_logger_1 NUMERIC
);

-- Insert rows into the first table
INSERT INTO Source1_8 (datetime, cerc_logger_1)
VALUES
    ('2/22/2018 0:30', 22.875),
    ('2/22/2018 0:40', 22.937),
    ('2/22/2018 0:50', 22.937),
    ('2/22/2018 1:00', 22.937),
    ('2/22/2018 1:10', 23);

-- Drop the second table if it exists
DROP TABLE IF EXISTS Target1;

-- Create the second table
CREATE TABLE Target1 (
    CST TEXT,
    "1:00" NUMERIC,
    "2:00" NUMERIC,
    "3:00" NUMERIC,
    "4:00" NUMERIC,
    "5:00" NUMERIC,
    "6:00" NUMERIC,
    "7:00" NUMERIC,
    "8:00" NUMERIC,
    "9:00" NUMERIC,
    "10:00" NUMERIC,
    "11:00" NUMERIC,
    "12:00" NUMERIC,
    "13:00" NUMERIC,
    "14:00" NUMERIC,
    "15:00" NUMERIC,
    "16:00" NUMERIC,
    "17:00" NUMERIC,
    "18:00" NUMERIC,
    "19:00" NUMERIC,
    "20:00" NUMERIC,
    "21:00" NUMERIC,
    "22:00" NUMERIC,
    "23:00" NUMERIC,
    "24:00" NUMERIC
);

-- Insert rows from the first table into the second table
INSERT INTO Target1 (CST, "1:00", "2:00", "3:00", "4:00", "5:00", "6:00", "7:00", "8:00", "9:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00")
SELECT
    TO_CHAR(datetime, 'Dy MM/DD/YYYY') AS CST,
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 1 THEN cerc_logger_1 END) AS "1:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 2 THEN cerc_logger_1 END) AS "2:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 3 THEN cerc_logger_1 END) AS "3:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 4 THEN cerc_logger_1 END) AS "4:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 5 THEN cerc_logger_1 END) AS "5:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 6 THEN cerc_logger_1 END) AS "6:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 7 THEN cerc_logger_1 END) AS "7:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 8 THEN cerc_logger_1 END) AS "8:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 9 THEN cerc_logger_1 END) AS "9:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 10 THEN cerc_logger_1 END) AS "10:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 11 THEN cerc_logger_1 END) AS "11:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 12 THEN cerc_logger_1 END) AS "12:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 13 THEN cerc_logger_1 END) AS "13:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 14 THEN cerc_logger_1 END) AS "14:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 15 THEN cerc_logger_1 END) AS "15:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 16 THEN cerc_logger_1 END) AS "16:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 17 THEN cerc_logger_1 END) AS "17:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 18 THEN cerc_logger_1 END) AS "18:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 19 THEN cerc_logger_1 END) AS "19:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 20 THEN cerc_logger_1 END) AS "20:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 21 THEN cerc_logger_1 END) AS "21:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 22 THEN cerc_logger_1 END) AS "22:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 23 THEN cerc_logger_1 END) AS "23:00",
    MAX(CASE WHEN EXTRACT(HOUR FROM datetime) = 0 THEN cerc_logger_1 END) AS "24:00"
FROM Source1_8
GROUP BY TO_CHAR(datetime, 'Dy MM/DD/YYYY');"""

#     gpt_output = """-- Drop the first table if it exists
# DROP TABLE IF EXISTS Source1_1;
#
# -- Create the first table
# CREATE TABLE Source1_1 (
#     DT_STRATA DATE,
#     DOW TEXT,
#     PCT_HOURLY_0100 NUMERIC,
#     PCT_HOURLY_0200 NUMERIC,
#     PCT_HOURLY_0300 NUMERIC,
#     PCT_HOURLY_0400 NUMERIC,
#     PCT_HOURLY_0500 NUMERIC,
#     PCT_HOURLY_0600 NUMERIC,
#     PCT_HOURLY_0700 NUMERIC,
#     PCT_HOURLY_0800 NUMERIC,
#     PCT_HOURLY_0900 NUMERIC,
#     PCT_HOURLY_1000 NUMERIC,
#     PCT_HOURLY_1100 NUMERIC,
#     PCT_HOURLY_1200 NUMERIC,
#     PCT_HOURLY_1300 NUMERIC,
#     PCT_HOURLY_1400 NUMERIC,
#     PCT_HOURLY_1500 NUMERIC,
#     PCT_HOURLY_1600 NUMERIC,
#     PCT_HOURLY_1700 NUMERIC,
#     PCT_HOURLY_1800 NUMERIC,
#     PCT_HOURLY_1900 NUMERIC,
#     PCT_HOURLY_2000 NUMERIC,
#     PCT_HOURLY_2100 NUMERIC,
#     PCT_HOURLY_2200 NUMERIC,
#     PCT_HOURLY_2300 NUMERIC,
#     PCT_HOURLY_2400 NUMERIC,
#     PCT_HOURLY_2500 NUMERIC
# );
#
# -- Insert rows into the first table
# INSERT INTO Source1_1 (DT_STRATA, DOW, PCT_HOURLY_0100, PCT_HOURLY_0200, PCT_HOURLY_0300, PCT_HOURLY_0400, PCT_HOURLY_0500, PCT_HOURLY_0600, PCT_HOURLY_0700, PCT_HOURLY_0800, PCT_HOURLY_0900, PCT_HOURLY_1000, PCT_HOURLY_1100, PCT_HOURLY_1200, PCT_HOURLY_1300, PCT_HOURLY_1400, PCT_HOURLY_1500, PCT_HOURLY_1600, PCT_HOURLY_1700, PCT_HOURLY_1800, PCT_HOURLY_1900, PCT_HOURLY_2000, PCT_HOURLY_2100, PCT_HOURLY_2200, PCT_HOURLY_2300, PCT_HOURLY_2400, PCT_HOURLY_2500)
# VALUES
#     ('1/1/16', 'H', .001222017108240, .001274017836250, .001131015834222, .001222017108240, .001222017108240, .001261017654247, .001144016016224, .001170016380229, .001170016380229, .001209016926237, .001183016562232, .001235017290242, .001157016198227, .001001014014196, .001014014196199, .001144016016224, .001209016926237, .001547021658303, .001560021840306, .001534021476301, .001573022022308, .001573022022308, .001326018564260, .001170016380229, .000000000000000),
#     ('1/2/16', '7', .001313018382257, .001248017472245, .001222017108240, .001196016744234, .001196016744234, .001287018018252, .001300018200255, .001339018746262, .001352018928265, .001404019656275, .001417019838278, .001326018564260, .001313018382257, .001352018928265, .001339018746262, .001391019474273, .001404019656275, .001495020930293, .001521021294298, .001508021112296, .001534021476301, .001534021476301, .001469020566288, .001339018746262, .000000000000000),
#     ('1/3/16', '1', .001157016198227, .001092015288214, .001053014742206, .001040014560204, .001040014560204, .001066014924209, .001053014742206, .001144016016224, .001183016562232, .001248017472245, .001326018564260, .001404019656275, .001313018382257, .001300018200255, .001313018382257, .001261017654247, .001261017654247, .001430020020280, .001469020566288, .001404019656275, .001456020384285, .001378019292270, .001287018018252, .001222017108240, .000000000000000),
#     ('1/4/16', '2', .001105015470217, .001053014742206, .001001014014196, .000975013650191, .001053014742206, .001079015106211, .001183016562232, .001170016380229, .001274017836250, .001170016380229, .001157016198227, .001196016744234, .001196016744234, .001144016016224, .001170016380229, .001261017654247, .001300018200255, .001495020930293, .001586022204311, .001651023114324, .001651023114324, .001521021294298, .001417019838278, .001274017836250, .000000000000000),
#     ('1/5/16', '3', .001183016562232, .001170016380229, .001079015106211, .001131015834222, .001170016380229, .001183016562232, .001300018200255, .001365019110268, .001404019656275, .001378019292270, .001352018928265, .001326018564260, .001365019110268, .001339018746262, .001300018200255, .001313018382257, .001404019656275, .001638022932321, .001716024024336, .001677023478329, .001625022750319, .001638022932321, .001573022022308, .001417019838278, .000000000000000);
#
# -- Drop the second table if it exists
# DROP TABLE IF EXISTS Target1;
#
# -- Create the second table
# CREATE TABLE Target1 (
#     CST TEXT,
#     "1:00" NUMERIC,
#     "2:00" NUMERIC,
#     "3:00" NUMERIC,
#     "4:00" NUMERIC,
#     "5:00" NUMERIC,
#     "6:00" NUMERIC,
#     "7:00" NUMERIC,
#     "8:00" NUMERIC,
#     "9:00" NUMERIC,
#     "10:00" NUMERIC,
#     "11:00" NUMERIC,
#     "12:00" NUMERIC,
#     "13:00" NUMERIC,
#     "14:00" NUMERIC,
#     "15:00" NUMERIC,
#     "16:00" NUMERIC,
#     "17:00" NUMERIC,
#     "18:00" NUMERIC,
#     "19:00" NUMERIC,
#     "20:00" NUMERIC,
#     "21:00" NUMERIC,
#     "22:00" NUMERIC,
#     "23:00" NUMERIC,
#     "24:00" NUMERIC
# );
#
# -- Insert rows from the first table into the second table
# INSERT INTO Target1 (CST, "1:00", "2:00", "3:00", "4:00", "5:00", "6:00", "7:00", "8:00", "9:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00")
# SELECT
#     TO_CHAR(DT_STRATA, 'Dy MM/DD/YYYY') AS CST,
#     PCT_HOURLY_0100,
#     PCT_HOURLY_0200,
#     PCT_HOURLY_0300,
#     PCT_HOURLY_0400,
#     PCT_HOURLY_0500,
#     PCT_HOURLY_0600,
#     PCT_HOURLY_0700,
#     PCT_HOURLY_0800,
#     PCT_HOURLY_0900,
#     PCT_HOURLY_1000,
#     PCT_HOURLY_1100,
#     PCT_HOURLY_1200,
#     PCT_HOURLY_1300,
#     PCT_HOURLY_1400,
#     PCT_HOURLY_1500,
#     PCT_HOURLY_1600,
#     PCT_HOURLY_1700,
#     PCT_HOURLY_1800,
#     PCT_HOURLY_1900,
#     PCT_HOURLY_2000,
#     PCT_HOURLY_2100,
#     PCT_HOURLY_2200,
#     PCT_HOURLY_2300,
#     PCT_HOURLY_2400
# FROM Source1_1;
# """
#     gpt_output = """-- Drop the first table if it exists
# DROP TABLE IF EXISTS Source1_5;
#
# -- Create the first table
# CREATE TABLE Source1_5 (
#     "date" DATE,
#     "Hour 1" NUMERIC,
#     "Hour 2" NUMERIC,
#     "Hour 3" NUMERIC,
#     "Hour 4" NUMERIC,
#     "Hour 5" NUMERIC,
#     "Hour 6" NUMERIC,
#     "Hour 7" NUMERIC,
#     "Hour 8" NUMERIC,
#     "Hour 9" NUMERIC,
#     "Hour 10" NUMERIC,
#     "Hour 11" NUMERIC,
#     "Hour 12" NUMERIC,
#     "Hour 13" NUMERIC,
#     "Hour 14" NUMERIC,
#     "Hour 15" NUMERIC,
#     "Hour 16" NUMERIC,
#     "Hour 17" NUMERIC,
#     "Hour 18" NUMERIC,
#     "Hour 19" NUMERIC,
#     "Hour 20" NUMERIC,
#     "Hour 21" NUMERIC,
#     "Hour 22" NUMERIC,
#     "Hour 23" NUMERIC,
#     "Hour 24" NUMERIC
# );
#
# -- Insert rows into the first table
# INSERT INTO Source1_5 VALUES
#     ('1/1/15', 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000119933, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000119933, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866),
#     ('1/2/15', 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000119933, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000119933, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866),
#     ('1/3/15', 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000119933, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000119933, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866),
#     ('1/4/15', 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000119933, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000119933, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866),
#     ('1/5/15', 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000119933, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000000001, 0.000119933, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866, 0.000239866);
#
# -- Drop the second table if it exists
# DROP TABLE IF EXISTS Target1;
#
# -- Create the second table
# CREATE TABLE Target1 (
#     CST TEXT,
#     "1:00" NUMERIC,
#     "2:00" NUMERIC,
#     "3:00" NUMERIC,
#     "4:00" NUMERIC,
#     "5:00" NUMERIC,
#     "6:00" NUMERIC,
#     "7:00" NUMERIC,
#     "8:00" NUMERIC,
#     "9:00" NUMERIC,
#     "10:00" NUMERIC,
#     "11:00" NUMERIC,
#     "12:00" NUMERIC,
#     "13:00" NUMERIC,
#     "14:00" NUMERIC,
#     "15:00" NUMERIC,
#     "16:00" NUMERIC,
#     "17:00" NUMERIC,
#     "18:00" NUMERIC,
#     "19:00" NUMERIC,
#     "20:00" NUMERIC,
#     "21:00" NUMERIC,
#     "22:00" NUMERIC,
#     "23:00" NUMERIC,
#     "24:00" NUMERIC
# );
#
# -- Insert rows from the first table into the second table
# INSERT INTO Target1 (CST, "1:00", "2:00", "3:00", "4:00", "5:00", "6:00", "7:00", "8:00", "9:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00", "23:00", "24:00")
# SELECT
#    TO_CHAR("date", 'Dy MM/DD/YYYY') AS CST,
#     "Hour 1",
#     "Hour 2",
#     "Hour 3",
#     "Hour 4",
#     "Hour 5",
#     "Hour 6",
#     "Hour 7",
#     "Hour 8",
#     "Hour 9",
#     "Hour 10",
#     "Hour 11",
#     "Hour 12",
#     "Hour 13",
#     "Hour 14",
#     "Hour 15",
#     "Hour 16",
#     "Hour 17",
#     "Hour 18",
#     "Hour 19",
#     "Hour 20",
#     "Hour 21",
#     "Hour 22",
#     "Hour 23",
#     "Hour 24"
# FROM Source1_5;"""
   mapping_result = mapping_quality(gpt_output,"Source1_8","Target1")
   print(mapping_result)
   elements = extract_elements(gpt_output)
   print("GROUP BY clauses:", elements['group_by'])
   print("TO_CHAR functions:", elements['to_char'])
   print("MAX functions:", elements['max'])
   print("MIN functions:", elements['min'])
   print("SUM functions:", elements['sum'])
   print("AVG functions:", elements['avg'])
   print("CASE statements:", elements['case_statements'])
   print("EXTRACT functions:", elements['extract'])
   print("GREATEST functions:", elements['greatest'])
   print("LEAST functions:", elements['least'])