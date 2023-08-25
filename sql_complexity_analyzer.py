import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_data_from_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)


def count_cases_in_groups(data: dict) -> list:
    group_case_count = {}
    pattern = re.compile(r"Source(\d+)_(\d+)")

    for entry in data:
        match = pattern.match(entry['Source Data Name'])
        if match:
            group = int(match.group(1))
            group_case_count[group] = group_case_count.get(group, 0) + 1

    return [group_case_count.get(i, 0) for i in range(1, max(group_case_count.keys()) + 1)]


def extract_correctness(data: dict, correctness_name: str) -> list:
    def extract_score(entry):
        try:
            return float(entry.get(correctness_name, 0))
        except ValueError:
            print(f"Problematic entry: {entry.get(correctness_name)}")
            return 0.0

    return [extract_score(entry) for entry in data]


def tokenize_sql_query(query: str) -> list:
    # Remove comments and extract words
    query = re.sub(r'--.*\n', '', query)
    query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
    return re.findall(r'\b\w+\b', query.upper())


def analyze_sql_complexity(query: str) -> tuple:
    complexity_keywords = [
        'JOIN', 'INNER JOIN', 'OUTER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'CROSS JOIN', 'SELF JOIN', 'FULL JOIN', 'UNION', 'UNION ALL',
        'SUBQUERY', 'WITH', 'EXISTS', 'NOT EXISTS', 'IN', 'NOT IN',
        'EXTRACT', 'SUM', 'COUNT', 'MIN', 'MAX', 'AVG', 'SUBSTRING',
        'COALESCE', 'TO_CHAR', 'TO_DATE', 'TO_NUMBER', 'CAST', 'CONVERT',
        'CASE', 'WHEN', 'ELSE', 'END', 'HAVING', 'DISTINCT', 'GROUP BY',
        'ORDER BY', 'LIMIT', 'OFFSET', 'WINDOW', 'OVER', 'PARTITION BY',
        'ROLLUP', 'CUBE', 'GROUPING SETS', 'RECURSIVE', 'PIVOT', 'UNPIVOT'
    ]

    tokens = tokenize_sql_query(query)
    found_keywords = set(tokens).intersection(complexity_keywords)
    keyword_score = len(found_keywords)
    nested_score = query.upper().count('( SELECT')
    normalized_length = len(query) / 200

    return keyword_score, keyword_score + nested_score + normalized_length, list(found_keywords)

def analyze_sql_complexity_separate(query: str) -> tuple:
    complexity_keywords = [
        'JOIN', 'INNER JOIN', 'OUTER JOIN', 'LEFT JOIN', 'RIGHT JOIN',
        'CROSS JOIN', 'SELF JOIN', 'FULL JOIN', 'UNION', 'UNION ALL',
        'SUBQUERY', 'WITH', 'EXISTS', 'NOT EXISTS', 'IN', 'NOT IN',
        'EXTRACT', 'SUM', 'COUNT', 'MIN', 'MAX', 'AVG', 'SUBSTRING',
        'COALESCE', 'TO_CHAR', 'TO_DATE', 'TO_NUMBER', 'CAST', 'CONVERT',
        'CASE', 'WHEN', 'ELSE', 'END', 'HAVING', 'DISTINCT', 'GROUP BY',
        'ORDER BY', 'LIMIT', 'OFFSET', 'WINDOW', 'OVER', 'PARTITION BY',
        'ROLLUP', 'CUBE', 'GROUPING SETS', 'RECURSIVE', 'PIVOT', 'UNPIVOT'
    ]

    tokens = tokenize_sql_query(query)
    found_keywords = set(tokens).intersection(complexity_keywords)
    keyword_score = len(found_keywords)
    nested_score = query.upper().count('( SELECT')
    length_score = len(query)/200

    return keyword_score, nested_score, length_score

def plot_complexity_scores_filled_area(queries_arg, complexity_function, group_counts_arg, correctness_scores):
    keyword_counts, nested_scores, lengths = zip(*[complexity_function(query) for query in queries_arg])
    correctness_percent = [score * 100 for score in correctness_scores]

    df = pd.DataFrame({
        'Query Number': range(1, len(queries_arg) + 1),
        'Keyword Count': keyword_counts,
        'Normalized Length': lengths,
        'Correctness Percent': correctness_percent
    })

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={'hspace': 0.1})

    # Set up color mappings for gradient fills
    norm_keyword_count = plt.Normalize(0, df['Keyword Count'].max())
    norm_length = plt.Normalize(0, df['Normalized Length'].max())

    # Plot Keyword Count on ax1 with a colorful colormap
    for i in range(1, len(df)):
        ax1.fill_between(df['Query Number'].iloc[i - 1:i + 1], 0, df['Keyword Count'].iloc[i - 1:i + 1],
                         color=plt.cm.RdYlBu_r(norm_keyword_count(df['Keyword Count'].iloc[i])))

    # Plot Normalized Lengths on ax2 with another colorful colormap
    for i in range(1, len(df)):
        ax2.fill_between(df['Query Number'].iloc[i - 1:i + 1], 0, df['Normalized Length'].iloc[i - 1:i + 1],
                         color=plt.cm.RdYlBu_r(norm_length(df['Normalized Length'].iloc[i])))

    # Plot Correctness on ax3 using a red-green color map
    for i in range(1, len(df)):
        ax3.fill_between(df['Query Number'].iloc[i - 1:i + 1], 0, df['Correctness Percent'].iloc[i - 1:i + 1],
                         color=sns.diverging_palette(10, 133, as_cmap=True)(correctness_scores[i - 1]))

    # Common code for plotting (e.g., adding vertical dashed lines)
    boundaries = np.cumsum(group_counts_arg)
    for boundary in boundaries[:-1]:
        for ax in (ax1, ax2, ax3):
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8)

    # Add color bars for legends
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm_keyword_count), ax=ax1, orientation="vertical").set_label('Keyword Count')
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm_length), ax=ax2, orientation="vertical").set_label('Normalized Length')
    fig.colorbar(plt.cm.ScalarMappable(cmap=sns.diverging_palette(10, 133, as_cmap=True), norm=plt.Normalize(0, 100)), ax=ax3, orientation="vertical").set_label('Correctness (%)')

    # Set labels and title
    ax1.set_ylabel('Keyword Count')
    ax2.set_ylabel('Normalized Length')
    ax3.set_ylabel('Correctness (%)')
    ax3.set_xlabel('Query Number')

    # Set the main title for the figure
    fig.suptitle('Complexity and Correctness Scores of SQL Queries')

    # Adjust layout and save the plot
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    plt.savefig("complexity_correctness.pdf", format="pdf")
    plt.show()



if __name__ == '__main__':
    data = load_data_from_json("chatgpt.json")
    queries = [str(entry["Ground Truth SQL"]).replace("\n", " ") for entry in data]
    group_counts = count_cases_in_groups(data)
    correctness = extract_correctness(data, "Prompt-2 Results")
    plot_complexity_scores_filled_area(queries, analyze_sql_complexity_separate, group_counts, correctness)