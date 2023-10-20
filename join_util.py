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
