import json
import os
from statistics import mean, median
from collections import defaultdict
from typing import List, Dict, Any

import fire  # type: ignore
import numpy as np
from scipy.stats import spearmanr  # type: ignore
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pyvis.network import Network  # type: ignore


def main(input_dir: str) -> None:
    agg_data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_scores: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
    judge_scores = defaultdict(list)
    models = list()
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        with open(path) as r:
            data = json.load(r)
            for output in data["outputs"]:
                player = output["player"]
                judge = data["judge"]
                scores = output["new_scores"]
                is_refusal = scores.pop("is_refusal")
                if max(is_refusal) == 1:
                    continue
                output_scores = []
                for value in scores.values():
                    output_scores.append(mean(value))
                models.append(judge["model_name"])
                agg_data[judge["model_name"]][player["model_name"]].append(mean(output_scores))
                judge_scores[judge["model_name"]].append(mean(output_scores))
                all_scores[str(output["messages"])][judge["model_name"]] = output
    models = list(set(models))
    G = nx.DiGraph()
    for i, model in enumerate(models):
        G.add_node(i, label=model)
    for judge_model, scores in agg_data.items():
        all_judge_scores = []
        for player_model, s in scores.items():
            if player_model in models:
                all_judge_scores.extend(s)
        min_score = np.percentile(all_judge_scores, 10)
        max_score = np.percentile(all_judge_scores, 90)
        print(min_score, max_score)
        for player_model, player_scores in scores.items():
            if player_model not in models:
                continue
            if player_model == judge_model:
                continue
            real_score = mean(player_scores)
            final_score = min(max((real_score - min_score) / (max_score - min_score), 0.0), 1.0)
            G.add_edge(
                models.index(judge_model),
                models.index(player_model),
                weight=final_score,
                label=str(int(final_score * 100)),
            )
            print(judge_model, player_model, final_score, real_score)

    pr = nx.pagerank(G, alpha=1.0)
    model_weights = dict()
    for idx, value in pr.items():
        model_weights[models[idx]] = value
    print(model_weights)

    mean_scores = {k: mean(v) for k, v in judge_scores.items()}

    total_human_scores = []
    total_model_pr_scores = []
    total_model_uniform_scores = []
    total_model_sonnet_scores = []
    total_model_top_2_scores = []
    for _, example_scores in all_scores.items():
        human_scores = None
        total_model_score = dict()
        count = 0
        for judge_model, output in example_scores.items():
            human_scores = output["human_scores"]
            judge_scores = output["new_scores"]
            judge_scores.pop("is_refusal", None)
            output_scores = []
            for value in judge_scores.values():
                output_scores.append(mean(value))
            total_judge_score = mean(output_scores) - mean_scores[judge_model]
            total_model_score[judge_model] = total_judge_score
            count += 1
        if not human_scores:
            continue
        if "claude-3-5-sonnet-20240620" not in total_model_score:
            continue

        total_human_score = mean(human_scores.values())
        total_human_scores.append(total_human_score)

        weights = {k: model_weights[k] for k in total_model_score.keys()}
        s = sum(weights.values())
        pr_weights = {k: v / s for k, v in weights.items()}
        uniform_weights = {k: 1 / len(weights) for k, v in weights.items()}
        top_2_weights = {
            k: 1 / 2 for k, v in weights.items() if k in ("claude-3-5-sonnet-20240620", "gpt-4o")
        }
        pr_total_model_score = sum([v * pr_weights[k] for k, v in total_model_score.items()])
        uniform_total_model_score = sum(
            [v * uniform_weights[k] for k, v in total_model_score.items()]
        )
        top_2_total_model_score = sum(
            [v * top_2_weights.get(k, 0.0) for k, v in total_model_score.items()]
        )
        print(top_2_total_model_score)
        total_model_top_2_scores.append(top_2_total_model_score)
        total_model_pr_scores.append(pr_total_model_score)
        total_model_uniform_scores.append(uniform_total_model_score)
        total_model_sonnet_scores.append(total_model_score["claude-3-5-sonnet-20240620"])

    print("Support:", len(total_human_scores))
    print("PageRank:", spearmanr(total_human_scores, total_model_pr_scores)[0])
    print("Average:", spearmanr(total_human_scores, total_model_uniform_scores)[0])
    print("Best model only:", spearmanr(total_human_scores, total_model_sonnet_scores)[0])
    print("Top-2:", spearmanr(total_human_scores, total_model_top_2_scores)[0])

    net = Network(directed=True, height="1000px")
    net.from_nx(G)
    net.toggle_physics(True)
    net.barnes_hut(
        spring_strength=0.006,
    )
    net.show("graph.html", notebook=False)


if __name__ == "__main__":
    fire.Fire(main)
