import os
import fire
import json
from pathlib import Path

import pandas as pd

def build_table(
    results_dir
):
    records = []
    for file_name in os.listdir(results_dir):
        file_path = Path(os.path.join(results_dir, file_name))
        model_name = file_path.stem
        with open(file_path, encoding="utf-8") as r:
            result = json.load(r)
        result.pop("outputs")
        result["model_name"] = model_name
        records.append(result)
    columns = ["model_name", "final_score", "refusal_ratio", "stay_in_character_score", "language_fluency_score", "entertainment_score"]
    pd.set_option("display.precision", 2)

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(pd.DataFrame(records).sort_values(by="final_score", ascending=False)[columns])

    from tabulate import tabulate

    df = pd.DataFrame(records).sort_values(by="final_score", ascending=False)[columns]

    # Convert DataFrame to list of lists for tabulate
    table_data = df.values.tolist()

    # Add column names as the first row
    table_data.insert(0, columns)

    # Create the table using tabulate
    table = tabulate(table_data, headers="firstrow", tablefmt="github", floatfmt=".2f")

    print("\n----- Github MD format -----\n")

    # Output table in Github MD format
    print(table)


if __name__ == "__main__":
    fire.Fire(build_table)
