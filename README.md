# ChatGPTwithSQLscript

## Overview

This repository contains Python scripts designed to automate the generation of SQL queries using the GPT-3.5 model from OpenAI. It also includes utility functions for database operations, file reading, logging, and similarity calculations.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Contributing](#contributing)

---

## Dependencies

- Python 3.8+
- `pandas`
- `openai`
- `matplotlib`
- `seaborn`
- `sqlparse`
- `psycopg2-binary`
- `numpy`
- `sklearn`

---

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/lmong11/ChatGPTwithSQLscript.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ChatGPTwithSQLscript
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1. Update the `config.py` with your OpenAI API key.
2. Run the `run.py` script to start the experiment:
    ```bash
    python run.py
    ```

You can set the template_option,source,target in the run.py.

```template_option = 1
    target_id, max_target_id = 26, 26
    source_id, max_source_id = 1, 1
```
Detailed template_option are in the gpt.py.

target_id,max_target_id mean the first group and the last group the script will iterate.
source_id, max_source_id mean the first source and the last source the script will iterate.

---

## Scripts

- `excel2json.py`: Converts Excel data to JSON format.
- `gpt.py`: Interacts with the GPT-3 model to generate SQL queries.
- `join.py`: Handles JOIN operations in SQL.
- `run.py`: Main script to run the experiments.
- `sql_complexity_analyzer.py`: Analyzes the complexity of SQL queries.
- `util.py`: Contains utility functions for database operations and more.

---

## Contributing

Feel free to fork the project and submit a pull request with your changes!

---


