import openai
import json
from config import OPENAI_API_KEY


openai.api_key = OPENAI_API_KEY


def chat_with_gpt(prompt):
    """ Interact with chatGPT model and extract SQL script from the response. """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10000,
        )
        complete_response = response.choices[0]['message']['content']
        return ''.join(complete_response.split("```sql")[1].split("```")[0].strip())
    except Exception as e:
        return f"Error: {str(e)}"


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

        Second, insert the following row(s) into the first table (treat empty value as NULL):

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

        Second, insert the following row(s) into the first table and please don't remove any values (treat empty value as NULL):

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

        Second, insert the following row(s) into the first table and please don't remove any values (treat empty value as NULL):

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
        prompt = f"""You are a skilled Postgres SQL developer. 
        You're consulting for a tech firm working with Postgres databases. 
        Their primary focus is ensuring that time-related operations, especially those dealing with TIMESTAMP data types, are accurate and efficient.
        Let's perform some tasks:
        1. Creating the {source_data_name} Table:
        - Check if a table named {source_data_name} exists. If it does, delete it.
        - Create a new table named {source_data_name}. This table should have exact attributes from the following 
        schema: {source_data_schema}.
        - Note:{source_data_description}
        2. Populating the {source_data_name} Table:
        - Insert the provided rows (treat empty value as NULL): 
        {samples} 
        into the {source_data_name} table.
        3. Creating the {target_data_name} Table:
        - Check if a table named {target_data_name} exists. If it does, delete it.
        - Create a new table named {target_data_name}. This table should have exact attributes from the following 
        schema:{target_data_schema}.
        - Important: {target_data_description}
        4. Transforming Data from {source_data_name} to {target_data_name}:
        - Write a SQL transformation query to insert all rows from the {source_data_name} table to the {target_data_name} table.
        - Briefly explain your logic for the transformation.
        - Transformation hints: {schema_change_hints}
        Please don't remove the {source_data_name} table, because we need it for validation.
        Please quote the returned SQL script to perform these tasks between "```sql\n" and "\n```".
        Remember, accuracy and efficient handling of time data are paramount for the firm.
        """

    print(prompt)
    print("Ground Truth SQL Query:")
    print(ground_truth)

    return prompt, ground_truth, target_data_name