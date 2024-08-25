import json
import traceback
import time
from typing import List, Any, Optional, Dict
from statistics import mean
from collections import defaultdict
from dataclasses import dataclass

import fire  # type: ignore
from scipy.stats import spearmanr  # type: ignore
from dataclasses_json import DataClassJsonMixin

from src.data import Character, Situation, ChatMessages, Settings
from src.util import encode_prompt, generate, parse_output
from src.provider import LLMProvider


@dataclass
class JudgeSingleOutput(DataClassJsonMixin):
    is_refusal_explanation: str
    is_refusal: bool
    in_character_explanation: str
    in_character_score: int
    fluency_explanation: str
    fluency_score: int
    entertaining_explanation: str
    entertaining_score: int


@dataclass
class JudgeOutput(DataClassJsonMixin):
    scores: List[JudgeSingleOutput]

    def get_aggregated(self) -> Dict[str, List[int]]:
        fixed_scores = defaultdict(list)
        for s in self.scores:
            fixed_scores["in_character"].append(s.in_character_score)
            fixed_scores["entertaining"].append(s.entertaining_score)
            fixed_scores["fluency"].append(s.fluency_score)
            fixed_scores["is_refusal"].append(int(s.is_refusal))
        return fixed_scores


def run_judge(
    character: Character,
    situation: Situation,
    messages: ChatMessages,
    system_prompt_path: str,
    user_prompt_path: str,
    character_prompt_path: str,
    provider: LLMProvider,
    **kwargs: Any
) -> JudgeOutput:
    char_description = encode_prompt(character_prompt_path, character=character)
    system_prompt = encode_prompt(system_prompt_path)
    user_prompt = encode_prompt(
        user_prompt_path,
        char_description=char_description,
        situation=situation.text,
        messages=messages,
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    output: Optional[JudgeOutput] = None
    for _ in range(3):
        try:
            print(prompt[0]["content"])
            print(prompt[1]["content"])
            result = generate(prompt, provider=provider, **kwargs)
            print(result)
            print()
            print("=============")
            print()
            output = JudgeOutput.from_dict(parse_output(result))
            break
        except Exception:
            traceback.print_exc()
            time.sleep(10)
            continue
    assert output is not None
    return output


def main(
    providers_path: str,
    settings_path: str,
    input_path: str,
    output_path: str,
    judge_name: str,
    language: str = "ru",
) -> None:
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
    for i, record in enumerate(records):
        character = Character.from_dict(record["character"])
        situation = Situation.from_dict(record["situation"])
        messages = record["messages"]
        try:
            output = run_judge(
                character=character,
                situation=situation,
                messages=messages,
                user_prompt_path=settings.judge_user_prompt_path,
                system_prompt_path=settings.judge_system_prompt_path,
                character_prompt_path=settings.character_prompt_path,
                provider=judge_provider,
                temperature=0.1,
                top_p=0.95,
                max_tokens=4096,
            )
        except Exception:
            continue

        fixed_scores = output.get_aggregated()
        scores = {k: mean(v) for k, v in fixed_scores.items() if k != "is_refusal"}

        human_scores = record["human_scores"]

        mapping = {
            "entertainment": "entertaining",
            "language_fluency": "fluency",
            "stay_in_character": "in_character",
        }
        prev_scores = {v: mean(record["scores"].pop(k)) for k, v in mapping.items()}

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


if __name__ == "__main__":
    fire.Fire(main)
