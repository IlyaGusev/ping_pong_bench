import fire
import json
import time
from statistics import mean
from collections import defaultdict
from scipy.stats import spearmanr  # type: ignore

from src.run_eval_v2 import run_judge, Character, Situation, Settings, LLMProvider


def main(
    providers_path: str,
    settings_path: str,
    input_path: str,
    output_path: str,
    judge_name: str,
    language: str = "ru"
):
    with open(providers_path, encoding="utf-8") as r:
        providers = {name: LLMProvider(**provider) for name, provider in json.load(r).items()}
    with open(settings_path, encoding="utf-8") as r:
        settings = Settings.from_dict(json.load(r)[language])

    judge_provider = providers[judge_name]
    with open(input_path) as r:
        records = [json.loads(line) for line in r]

    all_prev_scores = defaultdict(list)
    all_model_scores = defaultdict(list)
    all_human_scores = defaultdict(list)
    for i, record in enumerate(records[:100]):
        character = Character.from_dict(record["character"])
        situation = Situation.from_dict(record["situation"])
        messages = record["messages"]
        try:
            scores = run_judge(
                character=character,
                situation=situation,
                messages=messages,
                user_prompt_path=settings.judge_user_prompt_path,
                system_prompt_path=settings.judge_system_prompt_path,
                character_prompt_path=settings.character_prompt_path,
                provider=judge_provider,
                temperature=0.1,
                top_p=0.95,
                max_tokens=4096
            )
        except Exception:
            continue
        scores = scores.to_dict()["scores"]
        fixed_scores = defaultdict(list)
        for s in scores:
            for k, v in s.items():
                if k.endswith("score"):
                    fixed_scores[k.replace("_score", "")].append(v)
        scores = dict()
        for k, v in fixed_scores.items():
            scores[k] = mean(v)

        human_scores = record["human_scores"]
        prev_scores = record["scores"]
        mapping = {
            "entertainment": "entertaining",
            "language_fluency": "fluency",
            "stay_in_character": "in_character"
        }
        for k, v in mapping.items():
            prev_scores[v] = mean(prev_scores.pop(k))
        for key, value in scores.items():
            assert key in human_scores
            assert key in prev_scores
            all_model_scores[key].append(value)
            all_prev_scores[key].append(prev_scores[key])
            all_human_scores[key].append(human_scores[key])
        all_model_scores["total"].append(mean(scores.values()))
        all_prev_scores["total"].append(mean(prev_scores.values()))
        all_human_scores["total"].append(mean(human_scores.values()))
        print()
        print("======")
        print(len(all_model_scores["total"]))
        print("PREV")
        for key in all_prev_scores:
            print(key, spearmanr(all_prev_scores[key], all_human_scores[key])[0])

        print("NEW")
        for key in all_model_scores:
            print(key, spearmanr(all_model_scores[key], all_human_scores[key])[0])
        print("======")
        print()
        time.sleep(2)
if __name__ == "__main__":
    fire.Fire(main)
