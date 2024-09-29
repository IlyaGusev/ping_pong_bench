import os
import fire  # type: ignore
import json
from typing import Optional, List, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime
from statistics import mean, median
from collections import defaultdict

import pandas as pd  # type: ignore
from tabulate import tabulate
import numpy as np
from git import Repo

from src.build_player_html import generate_html


def bootstrap_mean(data: List[float], n_bootstrap: int = 1000) -> Tuple[float, float, float]:
    means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
        point_estimate = np.mean(means)
        ci_lower, ci_upper = np.percentile(means, [2.5, 97.5])
    return point_estimate, ci_lower, ci_upper


def get_last_commit_info() -> Dict[str, Any]:
    repo = Repo(".")
    latest_commit = repo.head.commit
    return {
        "hash": latest_commit.hexsha,
        "date": datetime.fromtimestamp(latest_commit.committed_date),
    }


def build_table(
    results_dir: str, output_path: Optional[str] = None, dialogues_path: Optional[str] = None
) -> None:
    results_dir = results_dir.rstrip("/").lstrip("/")

    all_scores: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    player_scores: Dict[str, List[Any]] = defaultdict(list)
    player_dialogs: Dict[str, Dict[str, Any]] = defaultdict(dict)
    player_refusals: Dict[str, Set[str]] = defaultdict(set)
    player2shortname = dict()
    for file_name in os.listdir(results_dir):
        if not file_name.endswith(".json"):
            continue
        file_path = Path(os.path.join(results_dir, file_name))
        with open(file_path, encoding="utf-8") as r:
            data = json.load(r)
        for output in data["outputs"]:
            player = data["player"]
            player_name = player["model_name"]
            player2shortname[player_name] = (
                file_name.split("player")[-1].replace(".json", "").strip("_")
            )
            player_dialogs[player_name][str(output["messages"])] = output["messages"]
            judge = data["judge"]
            judge_name = judge["model_name"]
            scores = output["scores"]
            output["player"] = player
            output["judge"] = judge
            output["interrogator"] = data["interrogator"]
            is_refusal = scores["is_refusal"]
            player_scores[player_name].append(output)
            if max(is_refusal) == 1:
                player_refusals[player_name].add(str(output["messages"]))
                continue
            key = str(output["messages"])
            all_scores[key][judge_name] = output

    weights = {"claude-3-5-sonnet-20240620": 1.0, "gpt-4o": 1.0}
    final_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _, example_scores in all_scores.items():
        example_judge_scores: Dict[str, Dict[str, Any]] = defaultdict(dict)
        player_name = None
        for judge_model, output in example_scores.items():
            player_name = output["player"]["model_name"]
            judge_scores = output["scores"]
            output_scores = []
            for key, value in judge_scores.items():
                if "refusal" in key:
                    continue
                score = mean(value)
                example_judge_scores[key][judge_model] = score
                output_scores.append(score)
            final_score = mean(output_scores)
            example_judge_scores["final"][judge_model] = final_score
        for key, scores in example_judge_scores.items():
            final_score = mean([weights[k] * v for k, v in scores.items()])
            final_scores[player_name][key].append(final_score)

    players = dict()
    for player_name, key_scores in final_scores.items():
        record: Dict[str, Any] = {}
        model_name = player2shortname[player_name]
        record["model_name"] = (
            f"[{model_name}]({{{{ '/{results_dir}/{model_name}' | relative_url}}}})"
        )
        record["num_situations"] = len(player_dialogs[player_name])
        outputs = list(player_dialogs[player_name].values())
        record["avg_length"] = int(
            mean([len(m["content"]) for o in outputs for m in o if m["role"] == "assistant"])
        )
        record["refusal_ratio"] = len(player_refusals[player_name]) / len(
            player_dialogs[player_name]
        )
        for k, s in key_scores.items():
            m, ci_lower, ci_upper = bootstrap_mean(s)
            record[k] = m
            record[k + "_ci_width"] = (ci_upper - ci_lower) / 2
        players[player_name] = record

    # Length normalization
    median_length = median([r["avg_length"] for r in players.values()])
    min_score = min([r["final"] for r in players.values()])
    max_score = max([r["final"] for r in players.values()])
    score_range = max_score - min_score
    adjustment_factor = 0.07
    for player_name, key_scores in final_scores.items():
        record = players[player_name]
        x = median_length / record["avg_length"]
        x = 1 + (x - 1) * adjustment_factor
        x = max(x, 1 - adjustment_factor)
        x = min(x, 1)

        s = [s * x for s in key_scores["final"]]
        m, ci_lower, ci_upper = bootstrap_mean(s)
        record["length_norm_score"] = m
        record["length_norm_score_ci_width"] = (ci_upper - ci_lower) / 2

    records = list(players.values())
    for record in records:
        record["final"] = "{:.2f}<sub><sup>±{:.2f}</sup></sub>".format(record["final"], record.pop("final_ci_width"))
        record["length_norm_score"] = "{:.2f}<sub><sup>±{:.2f}</sup></sub>".format(record["length_norm_score"], record.pop("length_norm_score_ci_width"))

    mapping = (
        ("model_name", "model_name"),
        ("length_norm_score", "length_norm_score"),
        ("final", "avg_score"),
        ("refusal_ratio", "refusal_ratio"),
        ("in_character", "stay_in_character_score"),
        ("fluency", "language_fluency_score"),
        ("entertaining", "entertain_score"),
        ("num_situations", "num_cases"),
        ("avg_length", "avg_length"),
    )
    for record in records:
        for key, value in mapping:
            record[value] = record.pop(key)
    records.sort(key=lambda x: x["length_norm_score"], reverse=True)
    rank = 0
    prev_final_score = None
    for i, record in enumerate(records):
        final_score = float(record["length_norm_score"].split("<sub>")[0])
        if not prev_final_score:
            rank = i + 1
            prev_final_score = final_score
        elif abs(final_score - prev_final_score) > 0.06:
            rank = i + 1
            prev_final_score = final_score
        record["#"] = rank

    columns = ["#"] + [m[1] for m in mapping]
    pd.set_option("display.precision", 2)

    # Set display options to show all columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    df = pd.DataFrame(records).sort_values(by="length_norm_score", ascending=False)[columns]
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
            commit_info = get_last_commit_info()
            languages = {"ru": "Russian", "en": "English"}
            parts = list(Path(results_dir).parts)
            language = "English"
            for part in parts:
                if part in languages:
                    language = languages[part]
            meta = f"### {language} learderboard, v2"
            meta += f"\n\n<sup>Last updated: {commit_info['date']}</sup>\n\n"
            w.write(meta + table)
    if dialogues_path:
        os.makedirs(dialogues_path, exist_ok=True)
        for player, scores in player_scores.items():
            name = player2shortname[player]
            judge2records = defaultdict(list)
            for record in scores:
                judge2records[record["judge"]["model_name"]].append(record)
            output_path = os.path.join(dialogues_path, f"{name}.html")
            html = "---\nlayout: default\n---\n"
            for judge, records in sorted(judge2records.items()):
                player = records[0]["player"]
                player["short_name"] = player2shortname[player["model_name"]]
                judge = records[0]["judge"]
                judge["short_name"] = player2shortname[judge["model_name"]]
                html += generate_html(
                    {
                        "outputs": records,
                        "player": records[0]["player"],
                        "judge": records[0]["judge"],
                        "interrogator": records[0]["interrogator"],
                    }
                )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)


if __name__ == "__main__":
    fire.Fire(build_table)
