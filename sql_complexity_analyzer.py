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


def plot_complexity_scores_filled_area(queries_arg, complexity_function, group_counts_arg, correctness_scores):
    complexity_scores = [complexity_function(query)[1] for query in queries_arg]
    correctness_percent = [score * 100 for score in correctness_scores]
    df = pd.DataFrame({
        'Query Number': range(1, len(queries_arg) + 1),
        'Complexity Score': complexity_scores,
    })

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'hspace': 0.05})

    # Set up color mappings
    norm_complexity = plt.Normalize(0, df['Complexity Score'].max())

    # Plot complexity on ax1
    for i in range(1, len(df)):
        ax1.fill_between(df['Query Number'].iloc[i - 1:i + 1], 0, df['Complexity Score'].iloc[i - 1:i + 1],
                         color=plt.cm.RdYlBu_r(norm_complexity(df['Complexity Score'].iloc[i])))

    # Plot correctness on ax2 using a red-green color map
    for i in range(1, len(df)):
        ax2.fill_between(df['Query Number'].iloc[i - 1:i + 1], 0, correctness_percent[i - 1],
                         color=sns.diverging_palette(10, 133, as_cmap=True)(correctness_scores[i - 1]))

    # Draw vertical dashed lines to divide groups
    boundaries = np.cumsum(group_counts_arg)
    for boundary in boundaries[:-1]:
        ax1.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8)
        ax2.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.8)

    # Add color bars for legends at the right of the plot
    ax1_pos = ax1.get_position()
    cbar_ax1 = fig.add_axes([ax1_pos.x1 + 0.02, ax1_pos.y0, 0.02, ax1_pos.height])
    plt.colorbar(plt.cm.ScalarMappable(cmap="RdYlBu_r", norm=norm_complexity), cax=cbar_ax1,
                 orientation="vertical").set_label('Complexity Score')

    ax2_pos = ax2.get_position()
    cbar_ax2 = fig.add_axes([ax2_pos.x1 + 0.02, ax2_pos.y0, 0.02, ax2_pos.height])
    plt.colorbar(plt.cm.ScalarMappable(cmap=sns.diverging_palette(10, 133, as_cmap=True), norm=plt.Normalize(0, 100)),
                 cax=cbar_ax2, orientation="vertical").set_label('Correctness Score (%)')

    # Set labels and title
    fig.suptitle('Complexity and Correctness Scores of SQL Queries')
    ax2.set_xlabel('Query Number')
    ax1.set_ylabel('Complexity Score')
    ax2.set_ylabel('Correctness Score (%)')

    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False)  # Disable x-axis ticks and labels for ax1
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for colorbars
    plt.show()

if __name__ == '__main__':
    data = load_data_from_json("chatgpt.json")
    queries = [str(entry["Ground Truth SQL"]).replace("\n", " ") for entry in data]
    group_counts = count_cases_in_groups(data)
    correctness = extract_correctness(data, "Prompt-1 Results")
    plot_complexity_scores_filled_area(queries, analyze_sql_complexity, group_counts, correctness)