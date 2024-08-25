import os
import json
import copy
import time
import shutil
import traceback
from typing import cast, Any, List, Dict, Tuple, Optional
from statistics import mean
from collections import defaultdict
from dataclasses import dataclass, field

import requests
import fire  # type: ignore
from tqdm import tqdm
from jinja2 import Template
from dataclasses_json import DataClassJsonMixin

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from src.util import encode_prompt, generate, parse_output
from src.data import Character, ChatMessages, Situation, Settings
from src.run_judge import run_judge
from src.provider import LLMProvider


@dataclass
class InterrogatorOutput(DataClassJsonMixin):
    next_utterance: str


def run_player(
    character: Character,
    messages: ChatMessages,
    provider: LLMProvider,
    character_prompt_path: str,
) -> str:
    system_message = encode_prompt(character_prompt_path, character=character)
    messages = [{"role": "system", "content": system_message}] + messages
    output = None
    for _ in range(5):
        try:
            for m in messages:
                print(f'{m["role"]}: {m["content"]}')
                print()
            print()
            output = generate(
                provider=provider,
                messages=messages,
                **provider.params,
            )
            print(output)
            print()
            print("=============")
            print()
            print()
        except Exception:
            traceback.print_exc()
            time.sleep(10)
            continue
        break
    assert output is not None
    return output


def run_interrogator(
    character: Character,
    situation: Situation,
    messages: ChatMessages,
    system_prompt_path: str,
    user_prompt_path: str,
    character_prompt_path: str,
    provider: LLMProvider,
    **kwargs: Any,
) -> InterrogatorOutput:
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
    output: Optional[InterrogatorOutput] = None
    for _ in range(5):
        try:
            result = generate(prompt, provider=provider, **kwargs)
            print(result)
            print()
            print("=============")
            print()
            output = InterrogatorOutput.from_dict(parse_output(result))
            break
        except Exception:
            traceback.print_exc()
            time.sleep(10)
            continue
    assert output is not None
    return output


def save(
    output_path: str,
    outputs: List[Dict[str, Any]],
    interrogator_provider: LLMProvider,
    judge_provider: LLMProvider,
    player_provider: LLMProvider,
    version: int,
) -> None:
    scores: Dict[str, List[int]] = defaultdict(list)
    refusal_count = sum([int(max(o["scores"]["is_refusal"])) for o in outputs])
    for o in outputs:
        example_scores = o["scores"]
        is_refusal = max(example_scores["is_refusal"])
        if not is_refusal:
            for k, v in example_scores.items():
                if k.endswith("score"):
                    scores[k].extend(v)

    agg_scores = dict()
    if scores:
        agg_scores = {k: mean(v) for k, v in scores.items()}
        agg_scores["final_score"] = mean(agg_scores.values())
    refusal_ratio = refusal_count / len(outputs)

    tmp_path = output_path + "_tmp"
    with open(tmp_path, "w", encoding="utf-8") as w:
        json.dump(
            {
                "outputs": outputs,
                "version": version,
                "refusal_ratio": refusal_ratio,
                "judge": judge_provider.to_dict(),
                "interrogator": interrogator_provider.to_dict(),
                "player": player_provider.to_dict(),
                **agg_scores,
            },
            w,
            ensure_ascii=False,
            indent=4,
        )
    shutil.move(tmp_path, output_path)


def compose_key(character: Character, situation: Situation) -> Tuple[str, str]:
    return (character.char_name, situation.text)


def process_situation(
    character: Character,
    situation: Situation,
    settings: Settings,
    player_provider: LLMProvider,
    interrogator_provider: LLMProvider,
    judge_provider: LLMProvider,
) -> Dict[str, Any]:
    messages: ChatMessages = []
    scores: Dict[str, List[int]] = defaultdict(list)
    for turn in range(situation.num_turns + 1):
        output = run_interrogator(
            character=character,
            situation=situation,
            messages=messages,
            user_prompt_path=settings.interrogator_user_prompt_path,
            system_prompt_path=settings.interrogator_system_prompt_path,
            character_prompt_path=settings.character_prompt_path,
            provider=interrogator_provider,
        )
        messages.append({"role": "user", "content": output.next_utterance})
        bot_message = run_player(
            provider=player_provider,
            messages=messages,
            character=character,
            character_prompt_path=settings.character_prompt_path,
        )
        messages.append({"role": "assistant", "content": bot_message})
    judge_output = run_judge(
        character=character,
        situation=situation,
        messages=messages,
        user_prompt_path=settings.judge_user_prompt_path,
        system_prompt_path=settings.judge_system_prompt_path,
        character_prompt_path=settings.character_prompt_path,
        provider=judge_provider,
    )
    final_output = {
        "messages": messages,
        "character": character.to_dict(),
        "situation": situation.to_dict(),
        "scores": judge_output.get_aggregated()
    }
    return final_output


def run_eval(
    providers_path: str,
    settings_path: str,
    output_path: str,
    player_name: str,
    interrogator_name: str,
    judge_name: str,
    language: str = "ru",
    every_x: int = 1,
) -> None:
    with open(providers_path, encoding="utf-8") as r:
        providers = {name: LLMProvider(**provider) for name, provider in json.load(r).items()}
    with open(settings_path, encoding="utf-8") as r:
        settings = Settings.from_dict(json.load(r)[language])

    outputs = []
    existing_keys = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as r:
            outputs = json.load(r)["outputs"]
            for output in outputs:
                character = Character.from_dict(output["character"])
                situation = Situation.from_dict(output["situation"])
                record_key = compose_key(character=character, situation=situation)
                existing_keys.add(record_key)

    interrogator_provider = providers[interrogator_name]
    player_provider = providers[player_name]
    judge_provider = providers[judge_name]

    total_iterations = len(settings.characters) * len(settings.situations)
    with tqdm(total=total_iterations, desc="Processing pairs") as pbar:
        index = -2
        for character in settings.characters:
            index += 1

            for situation in settings.situations:
                index += 1
                pbar.update(1)

                if index % every_x != 0:
                    continue
                record_key = compose_key(character=character, situation=situation)
                if record_key in existing_keys:
                    print(f"Existing key: {record_key}")
                    continue
                try:
                    final_output = process_situation(
                        character=character,
                        situation=situation,
                        settings=settings,
                        player_provider=player_provider,
                        interrogator_provider=interrogator_provider,
                        judge_provider=judge_provider,
                    )
                    outputs.append(final_output)
                except Exception:
                    traceback.print_exc()
                    time.sleep(30)
                    continue

                save(
                    output_path=output_path,
                    outputs=outputs,
                    interrogator_provider=interrogator_provider,
                    judge_provider=judge_provider,
                    player_provider=player_provider,
                    version=settings.version,
                )


if __name__ == "__main__":
    fire.Fire(run_eval)
