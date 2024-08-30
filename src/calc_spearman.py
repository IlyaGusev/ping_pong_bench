import sys
import json
from collections import defaultdict
from statistics import mean

import fire  # type: ignore
from scipy.stats import spearmanr  # type: ignore


def main(
    pred_path: str,
    ref_path: str,
    use_old_keys: bool = False,
    scores_key: str = "new_scores",
    ref_key: str = "human_scores",
) -> None:
    with open(pred_path) as r:
        predictions = json.load(r)["outputs"]
    with open(ref_path) as r:
        ref_list = [json.loads(line) for line in r]
        references = {str(rec["messages"]): rec for rec in ref_list}

    human_scores = defaultdict(list)
    model_scores = defaultdict(list)
    old_key_mapping = {
        "stay_in_character": "in_character",
        "entertainment": "entertaining",
        "language_fluency": "fluency",
    }
    model_names = list()
    for prediction in predictions:
        reference = references[str(prediction["messages"])]
        for key, score in reference[ref_key].items():
            human_scores[key].append(score)
        for key, score in prediction[scores_key].items():
            if use_old_keys:
                key = old_key_mapping[key]
            model_scores[key].append(mean(score))
        model_names.append(reference["player"]["model_name"])

    final_human_scores = [mean(s) for s in zip(*[scores for _, scores in human_scores.items()])]
    final_model_scores = [mean(s) for s in zip(*[scores for _, scores in model_scores.items()])]
    assert len(final_human_scores) == len(final_model_scores)
    print("Support:", len(final_human_scores))
    for key, ref_scores in human_scores.items():
        pred_scores = model_scores[key]
        corr = spearmanr(ref_scores, pred_scores)[0]
        print(f"{key}, {corr:.3f}")
    final_corr = spearmanr(final_human_scores, final_model_scores)[0]
    print(f"final, {final_corr:.3f}")

    print()
    print("Human learderboard:")
    human_leaderboard_scores = defaultdict(list)
    for model_name, score in zip(model_names, final_human_scores):
        human_leaderboard_scores[model_name].append(score)
    human_leaderboard_results = list()
    for model_name, s in human_leaderboard_scores.items():
        human_leaderboard_results.append((mean(s), len(s), model_name))
    human_leaderboard_results.sort(reverse=True)
    for score, num, model_name in human_leaderboard_results:
        print(f"{score:.2f}, {num}, {model_name}")

    print()
    print("Model learderboard:")
    model_leaderboard_scores = defaultdict(list)
    for model_name, score in zip(model_names, final_model_scores):
        model_leaderboard_scores[model_name].append(score)

    model_leaderboard_results = list()
    for model_name, s in model_leaderboard_scores.items():
        model_leaderboard_results.append((mean(s), len(s), model_name))
    model_leaderboard_results.sort(reverse=True)
    for score, num, model_name in model_leaderboard_results:
        print(f"{score:.2f}, {num}, {model_name}")


if __name__ == "__main__":
    fire.Fire(main)
