import os
import pandas as pd

def remove_first_column_from_csv(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                # Drop the first column
                df.drop(df.columns[0], axis=1, inplace=True)
                # Save the modified DataFrame back to CSV
                df.to_csv(file_path, index=False)
                print(f"Processed {file_path}")

# Replace 'your_directory_path' with the path to your folder
remove_first_column_from_csv('../github-pipelines')
