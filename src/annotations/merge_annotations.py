import fire  # type: ignore
import json
from typing import Dict, List
from collections import defaultdict
from statistics import mean


def main(files: str, output_path: str) -> None:
    all_files = files.split(",")

    records = dict()
    scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for f in all_files:
        with open(f) as r:
            for line in r:
                record = json.loads(line)
                messages = str(record["messages"])
                records[messages] = record
                for k, v in record["human_scores"].items():
                    scores[messages][k].append(v)

    with open(output_path, "w") as w:
        for key, record in records.items():
            record_scores = scores[key]
            final_human_scores = {k: mean(s) for k, s in record_scores.items()}
            record["human_scores"] = final_human_scores
            w.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
