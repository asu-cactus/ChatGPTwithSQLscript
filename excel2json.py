import pandas as pd
import json

def convert_excel_to_json(excel_file_path, json_file_path):
    # Read the Excel file
    xls = pd.ExcelFile(excel_file_path)

    # Specify the columns to include in the JSON file
    columns_to_include = [
        "Target Data Name",
        "Target Data Schema",
        "Target Data Description",
        "Source Data Name",
        "Source Data Schema",
        "Source Data Description",
        "Contexts",
        "Schema Change Hints",
        "5 Samples of Source Data",
        "GroundTruth SQL",
        "Complexity",
        "Remark or Note"
    ]

    # Read the specified columns from the first sheet
    data_to_convert = pd.read_excel(xls, sheet_name=0, usecols=columns_to_include)

    # Fill missing values in the specified columns by forward filling
    columns_to_fill = ["Target Data Name", "Target Data Schema", "Target Data Description"]
    data_to_convert[columns_to_fill] = data_to_convert[columns_to_fill].ffill()

    # Open the JSON file for writing
    with open(json_file_path, 'w') as json_file:
        # Iterate through each row and write to the JSON file with indentation
        for _, row in data_to_convert.iterrows():
            json_file.write(json.dumps(row.to_dict(), indent=4) + '\n')

    print(f"JSON file has been saved to {json_file_path}")
# Path to the Excel file
excel_file_path = 'D:/SQL/ChatGPT Benchmark Datasets.xlsx'

# Path to save the JSON file
json_file_path = 'D:/SQL/ChatGPT Benchmark Datasets.json'

# Call the function to perform the conversion
convert_excel_to_json(excel_file_path, json_file_path)
