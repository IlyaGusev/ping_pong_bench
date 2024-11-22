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


SELECTOR_CODE = """
<div class="selector-container" style="display: flex; align-items: center;">
  <label for="weightSelector"><sup>Score weights:</sup> </label>
  <select id="weightSelector" onchange="updateVisibility()" class="weight-select">
    <option value="0.333_0.333_0.333" selected>character: 1, entertain: 1, fluency: 1</option>
    <option value="0.25_0.5_0.25">character: 1, entertain: 2, fluency: 1</option>
    <option value="0.5_0.25_0.25">character: 2, entertain: 1, fluency: 1</option>
    <option value="0.25_0.25_0.5">character: 1, entertain: 1, fluency: 2</option>
  </select>
  <style>
  .weight-select {
    padding: 3px;
    margin: 0 0 0 3px;
    font-family: monospace;
    background: #2D2D2D;
    color: #B5B5B5;
    border: 1px solid #404040;
    border-radius: 3px;
    line-height: 1
    height: 20px;
  }
  .weight-select option {
    background: #2D2D2D;
    color: #B5B5B5;
  }
  sub[data-weight]:not([data-weight="0.333_0.333_0.333"]) {
    display: none;
  }
  </style>

  <script>
    function updateVisibility() {
      const selected = document.getElementById('weightSelector').value;
      document.querySelectorAll('sub[data-weight]').forEach(sub => {
        sub.hidden = sub.dataset.weight !== selected;
      });

      document.querySelectorAll('th:has(sub[data-weight])').forEach(th => {
        const hasSelectedWeight = th.querySelector(`sub[data-weight="${selected}"]`);
        const columnIndex = Array.from(th.parentElement.children).indexOf(th);
        if (columnIndex > -1) {
          document.querySelectorAll(`tr td:nth-child(${columnIndex + 1})`).forEach(td => {
            td.style.display = hasSelectedWeight ? 'table-cell' : 'none';
          });
          th.style.display = hasSelectedWeight ? 'table-cell' : 'none';

          if (hasSelectedWeight && th.textContent.includes('Length norm score')) {
            const tbody = th.closest('table').querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            rows.sort((a, b) => {
              const aValue = parseFloat(a.children[columnIndex].textContent);
              const bValue = parseFloat(b.children[columnIndex].textContent);
              return bValue - aValue;
            });
            tbody.innerHTML = '';
            rows.forEach(row => tbody.appendChild(row));
          }
        }
      });
    }
    document.addEventListener('DOMContentLoaded', updateVisibility);
  </script>
</div>
"""


def display_str(text: str) -> str:
    return text.replace("_", " ").capitalize()


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

    judge_model_mapping = {"gpt-4o-2024-08-06": "gpt-4o"}
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
            judge_name = judge_model_mapping.get(judge_name, judge_name)
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

    model_weights = {"claude-3-5-sonnet-20240620": 1.0, "gpt-4o": 1.0}
    model_weights = {k: v / sum(model_weights.values()) for k, v in model_weights.items()}
    metric_header = ("in_character", "entertaining", "fluency")
    metric_weights = [
        (0.333, 0.333, 0.333),
        (0.25, 0.5, 0.25),
        (0.5, 0.25, 0.25),
        (0.25, 0.25, 0.5),
    ]
    default_weight_signature = "_".join(map(str, metric_weights[0]))

    final_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for _, example_scores in all_scores.items():
        example_judge_scores: Dict[str, Dict[str, Any]] = defaultdict(dict)
        player_name = None
        for judge_model, output in example_scores.items():
            player_name = output["player"]["model_name"]
            judge_scores = output["scores"]
            output_scores = dict()
            for key, value in judge_scores.items():
                if "refusal" in key:
                    continue
                score = mean(value)
                example_judge_scores[key][judge_model] = score
                output_scores[key] = score
            for metric_weight in metric_weights:
                merged_metric_weight = dict(zip(metric_header, metric_weight))
                final_score = sum([merged_metric_weight[k] * v for k, v in output_scores.items()])
                metric_weight_signature = "_".join(map(str, metric_weight))
                example_judge_scores[f"final_{metric_weight_signature}"][judge_model] = final_score
        for key, scores in example_judge_scores.items():
            final_score = sum([model_weights[k] * v for k, v in scores.items()])
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
    for metric_weight in metric_weights:
        metric_weight_signature = "_".join(map(str, metric_weight))
        final_key = f"final_{metric_weight_signature}"
        min_score = min([r[final_key] for r in players.values()])
        max_score = max([r[final_key] for r in players.values()])
        score_range = max_score - min_score
        adjustment_factor = 0.07
        for player_name, key_scores in final_scores.items():
            record = players[player_name]
            x = median_length / record["avg_length"]
            x = 1 + (x - 1) * adjustment_factor
            x = max(x, 1 - adjustment_factor)
            x = min(x, 1)
            v = key_scores[final_key]
            s = [s * x for s in v]
            m, ci_lower, ci_upper = bootstrap_mean(s)
            record[f"length_norm_score_{metric_weight_signature}"] = m
            record[f"length_norm_score_{metric_weight_signature}_ci_width"] = (ci_upper - ci_lower) / 2

    records = list(players.values())
    for record in records:
        for key in list(record.keys()):
            if ("final" in key or "length_norm_score" in key) and "ci_width" not in key:
                ci_width_key = key + "_ci_width"
                record[key] = "{:.2f}<sub><sup>±{:.2f}</sup></sub>".format(record[key], record.pop(ci_width_key))

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

    for key, value in mapping:
        for record in records:
            for rk, rv in list(record.items()):
                if key in rk:
                    new_key = rk.replace(key, value)
                    record[new_key] = record.pop(rk)

    length_norm_key = f"length_norm_score_{default_weight_signature}"
    records.sort(key=lambda x: x[length_norm_key], reverse=True)
    rank = 0
    prev_final_score = None
    for i, record in enumerate(records):
        final_score = float(record[length_norm_key].split("<sub>")[0])
        if not prev_final_score:
            rank = i + 1
            prev_final_score = final_score
        elif abs(final_score - prev_final_score) > 0.06:
            rank = i + 1
            prev_final_score = final_score
        record["#"] = rank

    columns = [k for k in records[0].keys() if "ci_width" not in k][:-1]
    columns.insert(0, "#")
    df = pd.DataFrame(records).sort_values(by=length_norm_key, ascending=False)[columns]
    table_data = df.values.tolist()
    columns = [display_str(row) for row in columns]
    table_data.insert(0, columns)

    # Create the table using tabulate
    table = tabulate(table_data, headers="firstrow", tablefmt="github", floatfmt=".2f")
    for metric_weight in metric_weights:
        metric_weight_signature = "_".join(map(str, metric_weight))
        key = display_str(f"avg_score_{metric_weight_signature}")
        table = table.replace(key, f'Avg score<sub data-weight="{metric_weight_signature}"></sub>')
        key = display_str(f"length_norm_score_{metric_weight_signature}")
        table = table.replace(key, f'Length norm score<sub data-weight="{metric_weight_signature}"></sub>')
    table = SELECTOR_CODE + "\n\n" + table

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
                judge_full_name = judge_model_mapping.get(judge["model_name"], judge["model_name"])
                judge["short_name"] = player2shortname[judge_full_name]
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
