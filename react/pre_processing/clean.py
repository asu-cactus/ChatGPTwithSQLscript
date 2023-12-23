import pandas as pd

# Load the CSV file
file_path = '../github-pipelines/length1_16/test_0.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure and the type of cleaning needed
data.head(10)

# Step 1: Remove unnecessary columns
# Dropping the 'Unnamed: 0' column
cleaned_data = data.drop(columns=['Unnamed: 0'])

# Step 2: Handling missing values
# For simplicity, we will fill missing values in 'ADDRESSLINE2' with an empty string,
# and in 'STATE', 'POSTALCODE', 'TERRITORY' with a placeholder 'Unknown'.
# However, depending on the specific use-case, other strategies might be more appropriate.
#cleaned_data['ADDRESSLINE2'].fillna('', inplace=True)
#cleaned_data.fillna({'STATE': 'Unknown', 'POSTALCODE': 'Unknown', 'TERRITORY': 'Unknown'}, inplace=True)

# Step 3: Standardizing Formats
# Converting 'ORDERDATE' to a datetime format, and keeping only the date part
#cleaned_data['ORDERDATE'] = pd.to_datetime(cleaned_data['ORDERDATE']).dt.date

# Step 4: Data Type Corrections
# Ensuring numerical columns are of correct data types
numerical_cols = ['QUANTITYORDERED', 'PRICEEACH', 'ORDERLINENUMBER', 'SALES', 'QTR_ID', 'MONTH_ID', 'YEAR_ID']
cleaned_data[numerical_cols] = cleaned_data[numerical_cols].apply(pd.to_numeric, errors='coerce')

# Step 5: Handling Duplicate Entries
# Removing duplicate rows if any
#cleaned_data = cleaned_data.drop_duplicates()

# Step 6: Data Validation
# For this step, a more in-depth understanding of the expected values in each column is needed.
# Assuming general knowledge, we will perform basic checks for some obvious columns.
# Ensuring 'COUNTRY' and 'STATUS' have consistent entries. This would normally require a list of valid countries and statuses.
# Here, we will just convert them to uppercase for consistency.
#cleaned_data['COUNTRY'] = cleaned_data['COUNTRY'].str.upper()
#cleaned_data['STATUS'] = cleaned_data['STATUS'].str.upper()

# Displaying the cleaned data
cleaned_data.head()


# Re-reading the CSV file with a closer look at possible quoted lines that might be causing issues
# We will try to detect and handle such anomalies in the CSV parsing

# Re-reading the file with a different approach to handle potential multi-line quoted cells
data_with_quoted_lines = pd.read_csv(file_path, quotechar='"', engine='python')

# Displaying the first few rows of this new dataframe to check for anomalies
data_with_quoted_lines.head()

# Searching for rows where quoted lines might be causing issues with multiple values in a single cell

# A function to detect if a row has a field that contains multiple comma-separated values
# which might indicate a quoted line causing issues
def detect_quoted_issues(row):
    for item in row:
        if isinstance(item, str) and ',' in item:
            return True
    return False

# Applying the function to the dataset
rows_with_issues = cleaned_data.apply(detect_quoted_issues, axis=1)

# Getting the indices of the rows with issues
indices_with_issues = cleaned_data[rows_with_issues].index

# Dropping these rows from the DataFrame
cleaned_data = cleaned_data.drop(indices_with_issues)

# Displaying the cleaned data
cleaned_data.head(10)
file = '../github-pipelines/length1_16/test_0_removed.csv'


cleaned_data.to_csv(file, index=False)
