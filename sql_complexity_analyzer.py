import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlparse

# Load JSON data from a given filepath
def load_data_from_json(file_path: str) -> dict:
    with open(file_path, 'r') as file:
        return json.load(file)

# Count the number of cases per group based on the Source Data Name pattern
def count_cases_in_groups(data: dict) -> list:
    pattern = re.compile(r"Source(\d+)_(\d+)")
    group_case_count = {
        int(match.group(1)): group_case_count.get(int(match.group(1)), 0) + 1
        for entry in data if (match := pattern.match(entry['Source Data Name']))
    }
    return [group_case_count.get(i, 0) for i in range(1, max(group_case_count) + 1)]

# Extract correctness scores from the data
def extract_correctness(data: dict, correctness_name: str) -> list:
    def extract_score(entry):
        try:
            return float(entry.get(correctness_name, 0))
        except ValueError:
            print(f"Problematic entry: {entry.get(correctness_name)}")
            return 0.0
    return [extract_score(entry) for entry in data]

# Extract SQL keywords from a query while filtering out unwanted patterns
def extract_keywords(query: str) -> list:
    parsed = sqlparse.parse(query)
    keywords = []

    def extract_tokens(token_list):
        for token in token_list:
            # Check for functions (like TO_CHAR)
            if isinstance(token, sqlparse.sql.Function):
                keywords.append(token.get_name().upper())

            # If it's a simple keyword
            elif token.ttype in (sqlparse.tokens.Keyword.DML, sqlparse.tokens.Keyword):
                keywords.append(token.value.upper())

            # Handle other potential keywords in nested tokens
            elif token.ttype is None:
                extract_tokens(token.tokens)

    for statement in parsed:
        extract_tokens(statement.tokens)

    # Remove duplicates
    keywords = list(set(keywords))

    # Filtering out unwanted keywords based on a pattern (e.g., table or schema names)
    unwanted_patterns = ["SOURCE", "TARGET"]
    return [kw for kw in set(keywords) if not any(pattern in kw for pattern in unwanted_patterns)]

# Analyze the complexity of a SQL query based on keyword count, nesting, and length
def analyze_sql_complexity_separate(query: str) -> tuple:
    found_keywords = extract_keywords(query)
    return (
        len(found_keywords),
        query.upper().count('( SELECT'),
        len(query)
    )

# Plot complexity scores using a filled area chart
def plot_complexity_scores_filled_area(queries_arg, complexity_function, group_counts_arg, correctness_scores):
    keyword_counts, nested_scores, lengths = zip(*[complexity_function(query) for query in queries_arg])
    df = pd.DataFrame({
        'Query Number': range(1, len(queries_arg) + 1),
        'Keyword Count': keyword_counts,
        'Normalized Length': lengths,
        'Correctness Percent': [score * 100 for score in correctness_scores]
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



# Entry point of the script
if __name__ == '__main__':
    data = load_data_from_json("chatgpt.json")
    queries = [str(entry["Ground Truth SQL"]).replace("\n", " ") for entry in data]
    group_counts = count_cases_in_groups(data)
    correctness = extract_correctness(data, "Prompt-2 Results")
    plot_complexity_scores_filled_area(queries, analyze_sql_complexity_separate, group_counts, correctness)
