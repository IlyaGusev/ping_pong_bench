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
        with open(file_path) as r:
            result = json.load(r)
        result.pop("outputs")
        result["model_name"] = model_name
        records.append(result)
    columns = ["model_name", "final_score", "refusal_ratio", "stay_in_character_score", "language_fluency_score", "entertainment_score"]
    pd.set_option("display.precision", 2)
    print(pd.DataFrame(records).sort_values(by="final_score", ascending=False)[columns])


if __name__ == "__main__":
    fire.Fire(build_table)
