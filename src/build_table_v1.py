import os
import fire  # type: ignore
import json
from typing import Optional, List
from pathlib import Path
from statistics import mean, median

import pandas as pd  # type: ignore
from tabulate import tabulate
import numpy as np

from src.build_player_html import generate_html


def build_table(
    results_dir: str, output_path: Optional[str] = None, dialogues_path: Optional[str] = None
) -> None:
    records = []
    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".json"):
            continue
        file_path = Path(os.path.join(results_dir, file_name))
        model_name = file_path.stem
        with open(file_path, encoding="utf-8") as r:
            result = json.load(r)
        outputs = result.pop("outputs")
        result["num_situations"] = len(outputs)
        result["avg_length"] = int(
            mean(
                [
                    len(m["content"])
                    for o in outputs
                    for m in o["messages"]
                    if m["role"] == "assistant"
                ]
            )
        )
        results_dir = results_dir.rstrip("/").lstrip("/")
        result["model_name"] = (
            f"[{model_name}]({{{{ '/{results_dir}/{model_name}' | relative_url}}}})"
        )
        records.append(result)

    median_length = median([r["avg_length"] for r in records])
    min_score = min([r["final_score"] for r in records])
    max_score = max([r["final_score"] for r in records])
    score_range = max_score - min_score
    for record in records:
        x = median_length - record["avg_length"]
        coef = np.tanh(x / (median_length * 3))
        diff = score_range * min(0, coef)
        record["length_norm_score"] = record["final_score"] + diff

    columns = [
        "model_name",
        "final_score",
        "length_norm_score",
        "refusal_ratio",
        "stay_in_character_score",
        "language_fluency_score",
        "entertainment_score",
        "num_situations",
        "avg_length",
    ]
    pd.set_option("display.precision", 2)

    # Set display options to show all columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    df = pd.DataFrame(records).sort_values(by="final_score", ascending=False)[columns]
    print(df)

    # Convert DataFrame to list of lists for tabulate
    table_data = df.values.tolist()

    # Add column names as the first row
    columns = [row.replace("_", " ").capitalize() for row in columns]
    table_data.insert(0, columns)

    # Create the table using tabulate
    table = tabulate(table_data, headers="firstrow", tablefmt="github", floatfmt=".2f")
    if output_path:
        with open(output_path, "w") as w:
            w.write(table)
    if dialogues_path:
        os.makedirs(dialogues_path, exist_ok=True)
        for file_name in os.listdir(results_dir):
            if not file_name.endswith(".json"):
                continue
            input_path = os.path.join(results_dir, file_name)
            output_path = os.path.join(dialogues_path, file_name.replace(".json", ".html"))
            with open(input_path) as r:
                data = json.load(r)
            html = generate_html(data)
            html = "---\nlayout: default\n---\n" + html
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)


if __name__ == "__main__":
    fire.Fire(build_table)
