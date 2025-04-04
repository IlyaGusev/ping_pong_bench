import os
import json
from collections import defaultdict

import fire

from scipy.stats import kendalltau

def collect_interrogator_exp(
    input_dir
):
    players = set()
    judges = set()
    interrogators = set()
    scores = defaultdict(dict)
    judge_scores = defaultdict(dict)
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

    interrogators = list(interrogators)
    players = list(players)
    judges = list(judges)

    print("Players:", players)
    print("By interrogator:")
    rankings = list()
    for interrogator in interrogators:
        player_scores = sorted([(v, k) for k, v in scores[interrogator].items()], reverse=True)
        print(f"interrogator={interrogator}")
        ranking = [-1 for _ in range(len(players))]
        for rank, (score, player) in enumerate(player_scores):
            index = players.index(player)
            ranking[index] = rank
            print(f"player={player}, score={score}")
        assert -1 not in ranking, players[ranking.index(-1)]
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
    for judge in judges:
        player_scores = sorted([(v, k) for k, v in judge_scores[judge].items()], reverse=True)
        print(f"judge={judge}")
        ranking = [-1 for _ in range(len(players))]
        for rank, (score, player) in enumerate(player_scores):
            index = players.index(player)
            ranking[index] = rank
            print(f"player={player}, score={score}")
        assert -1 not in ranking, players[ranking.index(-1)]
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
