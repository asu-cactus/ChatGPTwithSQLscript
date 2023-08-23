# sql_complexity_analyzer.py

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_from_json(json_file_path):
    """Load data from the provided JSON file."""
    with open(json_file_path, 'r') as file:
        return json.load(file)


def tokenize_sql_query(query):
    """
    Tokenize the given SQL query.
    - Remove comments
    - Extract words from the query
    """
    query = re.sub(r'--.*\n', '', query)  # Remove single line comments
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)  # Remove multi-line comments

    return re.findall(r'\b\w+\b', query.upper())


def analyze_sql_complexity(query):
    """
    Analyze the complexity of an SQL query based on the presence of certain keywords and its length.
    Returns the complexity score and the found keywords.
    """
    # Keywords indicating complexity
    complexity_keywords = [
        'JOIN', 'INNER JOIN', 'OUTER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'CROSS JOIN', 'SELF JOIN', 'FULL JOIN', 'UNION', 'UNION ALL',
        'SUBQUERY', 'WITH', 'EXISTS', 'NOT EXISTS', 'IN', 'NOT IN',
        'CASE', 'WHEN', 'ELSE', 'END', 'HAVING', 'DISTINCT', 'GROUP BY',
        'ORDER BY', 'LIMIT', 'OFFSET', 'WINDOW', 'OVER', 'PARTITION BY',
        'ROLLUP', 'CUBE', 'GROUPING SETS', 'RECURSIVE', 'PIVOT', 'UNPIVOT'
    ]

    tokens = tokenize_sql_query(query)
    found_keywords = [word for word in complexity_keywords if word in tokens]

    # Complexity score initially based on the number of keywords found
    complexity_score = len(found_keywords)

    if 'SELECT' in tokens:
        # Consider nested queries
        nested_score = len(re.findall(r'\(\s*SELECT', query.upper()))
        complexity_score += nested_score

    # Incorporate query length into the score
    normalized_length = len(query) / 200  # Normalize length by a chosen value

    complexity_score += normalized_length

    return complexity_score, found_keywords


def plot_complexity_scores_filled_area(queries, complexity_function):
    """Visualize complexity scores of SQL queries with a filled area plot."""
    # Calculate complexity scores
    complexity_scores = [complexity_function(query)[0] for query in queries]

    # Create a dataframe for visualization
    df = pd.DataFrame({
        'Query Number': range(1, len(queries) + 1),
        'Complexity Score': complexity_scores
    })

    # Set the style for the plot
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7.6, 3.5))

    # Create a color map for the scores
    norm = plt.Normalize(df['Complexity Score'].min(), df['Complexity Score'].max())
    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm)
    sm.set_array([])

    # Plot segments and fill the areas beneath them
    for i in range(1, len(df)):
        plt.plot(df['Query Number'].iloc[i - 1:i + 1], df['Complexity Score'].iloc[i - 1:i + 1],
                 color=plt.cm.RdYlBu_r(norm(df['Complexity Score'].iloc[i])))
        plt.fill_between(df['Query Number'].iloc[i - 1:i + 1], 0, df['Complexity Score'].iloc[i - 1:i + 1],
                         color=plt.cm.RdYlBu_r(norm(df['Complexity Score'].iloc[i])))

    # Add the colorbar, labels, and title
    plt.colorbar(sm)
    plt.title('Complexity Scores of SQL Queries')
    plt.ylabel('Complexity Score')
    plt.xlabel('Query Number')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Sample usage
    data_list = load_data_from_json("chatgpt.json")
    queries = [str(entry["Ground Truth SQL"]).replace("\n", " ") for entry in data_list]
    plot_complexity_scores_filled_area(queries, analyze_sql_complexity)
