import openai
import csv
import json
openai.api_key = 'xxxx'

def read_csv_file(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)
    return data

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

        print("SQL Script Extracted from GPT Response:")
        print(sql_script)

        return sql_script
    except Exception as e:
        return str(e)

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

        Second, insert 5 rows into the first table:

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

        Second, insert 5 rows into the first table:

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

        Second, insert 5 rows into the first table:

        {samples}

        Third, you must create a second table named {target_data_name} with only the given attributes: {target_data_schema}. Please delete the table before creating it if the first table exists.

        Finally, insert all rows from the first table into the second table, note that the selection clause in the insert statement should ignore attributes that are not needed.

        Please don't remove the first table, because we need it for validation.

        Please quote the returned SQL script between "```sql\n" and "\n```".

        Some explanation for the first table: {source_data_description}

        Some explanation for the second table: {target_data_description}

        Some hints for the schema changes from the first table to the second table: {schema_change_hints}

        """

    return prompt, ground_truth, target_data_name

# main script
def main(template_option):

    json_file_path = 'chatgpt.json'
    target_id = 4
    source_id = 6
    max_target_id = 5
    max_source_id = 8


    while (target_id <= max_target_id):
        while (source_id <= max_source_id):
            # Source Data Name to find
            source_data_name_to_find = "Source" + str(target_id) + "_" + str(source_id)
            source_id = source_id + 1
            print(source_data_name_to_find)
            # Generate the prompt for the chatGPT model
            if template_option == 1 or template_option == 2 or template_option == 3:
                prompt, ground_truth_query, target_data_name = generate_prompt(json_file_path, template_option,
                                                                               source_data_name_to_find)
            else:
                print("Invalid template option.")
                return

            # Run the experiment

            gpt_output = chat_with_gpt(prompt)
            #print(gpt_output)

            if "Error:" in gpt_output:
                prompt += " GPT Error: " + gpt_output
                continue


        target_id = target_id + 1



if __name__ == "__main__":
    template_option = int(input("Choose template option (1/2/3): "))
    main(template_option)
