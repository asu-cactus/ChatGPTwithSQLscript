import pandas as pd
import json
from datetime import datetime

def convert_datetime(obj):
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError("Type not serializable")

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

    # Create a list to store the dictionaries
    json_data = []

    # Iterate through each row and append to the list
    for _, row in data_to_convert.iterrows():
        json_data.append(row.to_dict())

    # Open the JSON file for writing
    with open(json_file_path, 'w') as json_file:
        json_data = [row.to_dict() for _, row in data_to_convert.iterrows()]
        json_file.write(json.dumps(json_data, indent=4, default=convert_datetime))

    print(f"JSON file has been saved to {json_file_path}")

# Path to the Excel file
excel_file_path = 'D:/SQL/ChatGPT Benchmark Datasets.xlsx'

# Path to save the JSON file
json_file_path = 'D:/SQL/ChatGPT Benchmark Datasets.json'

# Call the function to perform the conversion
convert_excel_to_json(excel_file_path, json_file_path)
