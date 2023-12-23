Please have a config.py file in the react folder with the following content:

```
OPENAI_API_KEY = "your_gpt_4_key"

```
Please download the content of the following link and put the 'github-pipelines' folder here: https://gitlab.com/jwjwyoung/autopipeline-benchmarks

Then, do the following steps:

1. run the react/pre_processing/remove_id_columns.py script to remove the id columns from the datasets
2. run the react/pre_processing/clean.py script to clean the dataset for length1_16, length1_40, and length1_59.
3. rename the react/github-pipelines/length1_16/test_0.csv to react/github-pipelines/length1_16/test_0_dirty.csv for backup
4. rename the react/github-pipelines/length1_40/test_0_removed.csv to react/github-pipelines/length1_40/test_0.csv
5. replace the test_0.csv in react/github-pipelines/length1_59 and react/github-pipelines/length1_40 with the test_0.csv in react/github-pipelines/length1_16
6. to get output.xlsx, run the react/pre_processing/generate_output.py

7. to get the json file, run the react/pre_processing/excel2json.py


