# ChatGPTwithSQLscript

## Overview

This repository contains Python scripts designed to automate the generation of SQL queries using the GPT-3 model from OpenAI. It also includes utility functions for database operations, file reading, logging, and similarity calculations.

---

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Contributing](#contributing)
- [License](#license)

---

## Dependencies

- Python 3.x
- `pymysql`
- `psycopg2`
- `csv`
- `re`
- `difflib`
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

---

## Scripts

- `Generate_GTSQL.py`: Generates ground truth SQL queries.
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

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

