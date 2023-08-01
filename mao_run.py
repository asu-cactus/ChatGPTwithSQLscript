import openai
import pymysql
from pymysql.err import OperationalError
import difflib
import numpy as np
from pymysql.err import ProgrammingError
openai.api_key = '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'

# define database connection
def create_connection():
    conn = None
    try:
        conn = pymysql.connect(host='localhost', user='root', password='sql', db='sql_gpt')
        print('Successfully connected to the database')
    except OperationalError as e:
        print(e)
    return conn

def execute_sql(conn, sql):
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()
        return cursor.fetchall()
    except (OperationalError, ProgrammingError) as e:
        print("Warning: The SQL query cannot be executed. Check the syntax and table name.")
        print("Error details:", e)
        return str(e)



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


def verify(sql_result, ground_truth):
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


# main script
def main(template_option):
    conn = create_connection()

    # Define the table you want to work with
    table_name_origin = "test_origin"
    table_name_target = "test_target"

    # Retrieve the original data from the specified table
    original_data_query = "SELECT * FROM test_origin;"
    original_data = execute_sql(conn, original_data_query)
    print(original_data)
    # Retrieve the ground truth from the specified table
    ground_truth_query = "SELECT * FROM test_target"
    ground_truth = execute_sql(conn, ground_truth_query)
    print(ground_truth)

    # Generate the prompt for the chatGPT model
    if template_option == 1:
       prompt = """You are a SQL developer. I want you to finish a task, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named test_1 {date, cerc_templogger_1 }
Insert 5 rows into the source table.
Second, you have to make a second table for the given attributes, named test_2: {CST,10, 20, 30, 40, 50}
Insert all rows from the first table into the second table."""

    elif template_option == 2:
        prompt = """You are a SQL developer. I want you to finish a task, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named SourceSchema {date, cerc_templogger_1 }
Second, you have to make a second table for the given attributes, named Target Schema: {CST,1:1:00, 1:2:00, 1:3:00, 1:4:00, 1:5:00}
The date column has a timestamp. 
The cerc_templogger_1 is the logger having each minute temperature value. 
In the target table, the CST is a date.
The  1:1:00,	1:2:00,	1:3:00,	1:4:00,	1:5:00 are minutes for temperature from. 
Insert all rows from the first table into the second table."""
    elif template_option == 3:
        prompt = """You are a SQL developer. I want you to finish a task, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named SourceSchema {date, cerc_templogger_1 }
Insert 5 rows into the source table.
Second, you have to make a second table for the given attributes, named Target Schema: {CST,1:1:00, 1:2:00, 1:3:00, 1:4:00, 1:5:00}
Insert all rows from the first table into the second table. 

Example-1 
Question
You are a SQL developer. I want you to finish a task, please generate a Postgres sql script to convert the first table to be consistent with the format of the second table. First, you have to create the first table with the given attribute names, named SourceSchema {Product, Month, Revenue}
Insert 5 rows into the source table.
Second, you have to make a second table for the given attributes, named Target Schema: {Product, Jan, Feb,Mar,Apr,May}. Insert all the rows from source to target table."""
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

        if "Error:" in gpt_output:
            prompt += " GPT Error: " + gpt_output
            continue

        # Execute the SQL query on the specified table
        print(gpt_output)
        sql_result = execute_sql(conn, gpt_output)
        print(sql_result)
        # Compare the SQL result with the ground truth
        is_correct = verify(ground_truth, ground_truth )
        print(is_correct)
        if isinstance(sql_result, str):
            prompt += " SQL Error: " + sql_result
            continue

        if is_correct:
            print("Successful SQL execution with correct result.")
            break
        else:
            prompt = "The SQL query can run,  but the result is like the following, which is wrong: " + str(sql_result)



if __name__ == "__main__":
    template_option = int(input("Choose template option (1/2/3): "))
    main(template_option)
