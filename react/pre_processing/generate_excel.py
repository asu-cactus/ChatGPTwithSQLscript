import os
import pandas as pd
import re
from pathlib import Path

def extract_numbers_from_text(s, pattern):
    match = pattern.search(s)
    return int(match.group(1)) if match else float('inf')

def extract_numbers_from_folder(path):
    folder_pattern = re.compile(r'length(\d+)_(\d+)')
    match = folder_pattern.search(str(path))
    return (int(match.group(1)), int(match.group(2))) if match else (float('inf'), float('inf'))

def remove_duplicates_in_join(df, columns):
    for col in columns:
        df[col] = df[col].where(df[col].ne(df[col].shift()), '')
    return df

def change_target_source_name(df):
    df['Target Data Name'] = 'Target' + df['Target Data Name'].str[6:]
    df['Source Data Name'] = 'Source' + df['Target Data Name'].str[6:] + '_' + df['Source Data Name'].str[5:]
    return df

# Define the directory where your subfolders are located
base_directory = Path('../github-pipelines')

# Initialize a dictionary to store data
data = {}

test_pattern = re.compile(r'test_(\d+)')

# Iterate through the subdirectories
for folder in sorted(base_directory.iterdir(), key=extract_numbers_from_folder):
    if folder.is_dir():
        target_csv_path = folder / 'target.csv'
        if target_csv_path.exists():
            target_df = pd.read_csv(target_csv_path, low_memory=False, index_col=0)
            target_sample = [[]] if target_df.empty else target_df.head(3).values.tolist()
            target_header = [] if target_df.empty else target_df.columns.tolist()

            data[folder.name] = {'Target Data Schema': target_header, 'Target Data Sample': target_sample, 'Test Files': []}

            test_csv_files = [file for file in folder.iterdir() if file.name.startswith('test_') and file.name.endswith('.csv')]
            for test_csv_file in test_csv_files:
                test_df = pd.read_csv(test_csv_file, low_memory=False, index_col=0)
                test_sample = [[]] if test_df.empty else test_df.head(3).values.tolist()
                test_columns = test_df.columns.tolist()

                data[folder.name]['Test Files'].append((test_csv_file.name, test_sample, test_columns))

# Prepare DataFrame
df_rows = []
for folder_name, contents in data.items():
    for file_name, sample, columns in sorted(contents['Test Files'], key=lambda x: extract_numbers_from_text(x[0], test_pattern)):
        df_rows.append({
            'Target Data Name': folder_name,
            'Target Data Schema': contents['Target Data Schema'],
            'Target Data Sample': contents['Target Data Sample'],
            'Source Data Name': file_name[:-4],
            'Source Data Schema': columns,
            '3 Samples of Source Data': sample,
        })

df = pd.DataFrame(df_rows)
df['Test File Count'] = df.groupby('Target Data Name')['Target Data Name'].transform('count')
df_join = df[df['Test File Count'] > 1].copy()
df_non_join = df[df['Test File Count'] == 1].copy()

df_join = change_target_source_name(df_join)
df_join = remove_duplicates_in_join(df_join, ['Target Data Name', 'Target Data Schema', 'Target Data Sample'])
df_non_join = change_target_source_name(df_non_join)
df = change_target_source_name(df)

df.drop(columns=['Test File Count'], inplace=True)

output_excel_path = base_directory / 'output.xlsx'

try:
    with pd.ExcelWriter(output_excel_path) as writer:
        df.to_excel(writer, sheet_name='All_Data', index=False)
        df_join.to_excel(writer, sheet_name='Join', index=False)
        df_non_join.to_excel(writer, sheet_name='Non_Join', index=False)
    print(f"Data saved to {output_excel_path}")
except Exception as e:
    print(f"Error occurred while saving data: {e}")
