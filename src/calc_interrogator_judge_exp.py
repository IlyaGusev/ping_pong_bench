import os
import json
from collections import defaultdict
from typing import Dict, Any

import fire  # type: ignore

from scipy.stats import kendalltau  # type: ignore


def collect_interrogator_exp(
    input_dir: str
) -> None:
    players = set()
    judges = set()
    interrogators = set()
    scores: Dict[str, Dict[str, Any]] = defaultdict(dict)
    judge_scores: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for file_name in os.listdir(input_dir):
        with open(os.path.join(input_dir, file_name)) as r:
            record = json.load(r)
            interrogator = record["interrogator"]["model_name"]
            player = record["player"]["model_name"]
            judge = record["judge"]["model_name"]
            final_score = record["final_score"]
            players.add(player)
            judges.add(judge)
            interrogators.add(interrogator)
            scores[interrogator][player] = (final_score, len(record["outputs"]))
            judge_scores[judge][player] = (final_score, len(record["outputs"]))

    interrogators_list = list(interrogators)
    players_list = list(players)
    judges_list = list(judges)

    print("Players:", players_list)
    print("By interrogator:")
    rankings = list()
    for interrogator in interrogators_list:
        player_scores = sorted([(v, k) for k, v in scores[interrogator].items()], reverse=True)
        print(f"interrogator={interrogator}")
        ranking = [-1 for _ in range(len(players_list))]
        for rank, (score, player) in enumerate(player_scores):
            index = players_list.index(player)
            ranking[index] = rank
            print(f"player={player}, score={score}")
        assert -1 not in ranking, players_list[ranking.index(-1)]
        rankings.append(ranking)
        print()
    print(rankings)
    coefs = []
    for i, r1 in enumerate(rankings):
        for r2_index in range(i + 1, len(rankings)):
            r2 = rankings[r2_index]
            coefs.append(kendalltau(r1, r2).statistic)
    print(coefs)
    if coefs:
        print("Average Kendall Tau:", sum(coefs) / len(coefs))

    print()
    print("By judge:")
    rankings = list()
    for judge in judges_list:
        player_scores = sorted([(v, k) for k, v in judge_scores[judge].items()], reverse=True)
        print(f"judge={judge}")
        ranking = [-1 for _ in range(len(players_list))]
        for rank, (score, player) in enumerate(player_scores):
            index = players_list.index(player)
            ranking[index] = rank
            print(f"player={player}, score={score}")
        assert -1 not in ranking, players_list[ranking.index(-1)]
        rankings.append(ranking)
        print()
    print(rankings)
    coefs = []
    for i, r1 in enumerate(rankings):
        for r2_index in range(i + 1, len(rankings)):
            r2 = rankings[r2_index]
            coefs.append(kendalltau(r1, r2).statistic)
    print(coefs)
    if coefs:
        print("Average Kendall Tau:", sum(coefs) / len(coefs))


if __name__ == "__main__":
    fire.Fire(collect_interrogator_exp)
