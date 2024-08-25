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

from src.provider import LLMProvider


ChatMessage = Dict[str, Any]
ChatMessages = List[ChatMessage]


@dataclass
class Character(DataClassJsonMixin):
    char_name: str
    system_prompt: str
    tags: Optional[List[str]] = None
    example_prompt: Optional[str] = None
    initial_message: Optional[str] = None


@dataclass
class Situation(DataClassJsonMixin):
    text: str
    tags: Optional[List[str]] = None
    num_turns: int = 5


@dataclass
class Settings(DataClassJsonMixin):
    characters: List[Character]
    situations: List[Situation]
    version: int
    user_prompt_path: str
    system_prompt_path: str
    character_prompt_path: str


@dataclass
class TesterOutput(DataClassJsonMixin):
    next_user_utterance: str
    is_refusal_explanation: str
    is_refusal: bool
    stay_in_character_explanation: str
    stay_in_character_score: int
    language_fluency_explanation: str
    language_fluency_score: int
    entertainment_explanation: str
    entertainment_score: int

    def get_scores(self) -> Dict[str, int]:
        return {
            "stay_in_character": self.stay_in_character_score,
            "language_fluency": self.language_fluency_score,
            "entertainment": self.entertainment_score,
        }


def encode_prompt(template_path: str, **kwargs: Any) -> str:
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())

    new_kwargs = copy.deepcopy(kwargs)
    if "messages" in kwargs:
        messages = copy.deepcopy(kwargs["messages"])
        mapping = {"assistant": "bot"}
        for m in messages:
            m["role"] = mapping.get(m["role"], m["role"])
        new_kwargs["messages"] = messages

    return template.render(**new_kwargs).strip()


def generate(messages: ChatMessages, provider: LLMProvider, **kwargs: Any) -> str:
    params = copy.deepcopy(provider.params)
    for k, v in kwargs.items():
        params[k] = v

    messages_copy = copy.deepcopy(messages)

    # If we have additional system prompt in provider, add it to messages
    if provider.system_prompt != "" and messages_copy[0]["role"] == "system":
        messages_copy[0]["content"] = provider.system_prompt + "\n\n" + messages_copy[0]["content"]

    if provider.merge_system and messages_copy[0]["role"] == "system":
        system_content = messages_copy[0]["content"]
        user_content = messages_copy[1]["content"]
        messages_copy = messages_copy[1:]
        messages_copy[0]["content"] = f"{system_content}\n\nВопрос пользователя: {user_content}"

    casted_messages = [cast(ChatCompletionMessageParam, message) for message in messages_copy]
    chat_completion = provider.api.chat.completions.create(
        model=provider.model_name, messages=casted_messages, **params
    )
    return str(chat_completion.choices[0].message.content).strip()


def run_testee(
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


def parse_output(output: str) -> Dict[str, Any]:
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index : end_index + 1]
    text = text.strip()
    record: Dict[str, Any] = json.loads(text)
    for k in record:
        assert isinstance(k, str)
    return record


def run_tester(
    character: Character,
    situation: Situation,
    messages: ChatMessages,
    system_prompt_path: str,
    user_prompt_path: str,
    character_prompt_path: str,
    provider: LLMProvider,
) -> TesterOutput:
    char_description = encode_prompt(character_prompt_path, character=character)
    user_prompt = encode_prompt(
        user_prompt_path,
        char_description=char_description,
        situation=situation.text,
        messages=messages,
    )
    prompt = [
        {"role": "system", "content": encode_prompt(system_prompt_path)},
        {"role": "user", "content": user_prompt},
    ]
    output: Optional[TesterOutput] = None
    for _ in range(5):
        try:
            result = generate(prompt, provider=provider, temperature=0.1, top_p=0.9)
            print(result)
            print()
            print("=============")
            print()
            output = TesterOutput.from_dict(parse_output(result))
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
    tester_provider: LLMProvider,
    testee_provider: LLMProvider,
    version: int,
) -> None:
    scores: Dict[str, List[int]] = defaultdict(list)
    refusal_count = sum([int(o["has_refusal"]) for o in outputs])
    for o in outputs:
        if not o["has_refusal"]:
            for k, v in o["scores"].items():
                scores[k].extend(v)

    agg_scores = dict()
    if scores:
        agg_scores = {k + "_score": mean(v) for k, v in scores.items()}
        agg_scores["final_score"] = mean(agg_scores.values())
    refusal_ratio = refusal_count / len(outputs)

    tmp_path = output_path + "_tmp"
    with open(tmp_path, "w", encoding="utf-8") as w:
        json.dump(
            {
                "outputs": outputs,
                "version": version,
                "refusal_ratio": refusal_ratio,
                "tester": tester_provider.to_dict(),
                "testee": testee_provider.to_dict(),
                **agg_scores,
            },
            w,
            ensure_ascii=False,
            indent=4,
        )
    shutil.move(tmp_path, output_path)


def compose_key(character: Character, situation: Situation) -> Tuple[str, str]:
    return (character.char_name, situation.text)


def run_eval(
    providers_path: str,
    settings_path: str,
    output_path: str,
    testee_name: str,
    tester_name: str,
    language: str = "ru",
    every_x: int = 1,
) -> None:
    with open(providers_path, encoding="utf-8") as r:
        providers = {name: LLMProvider(**provider) for name, provider in json.load(r).items()}
    with open(settings_path, encoding="utf-8") as r:
        all_settings = json.load(r)
        settings = Settings.from_dict(all_settings[language])

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

    tester_provider = providers[tester_name]
    testee_provider = providers[testee_name]

    total_iterations = len(settings.characters) * len(settings.situations)

    with tqdm(total=total_iterations, desc="Processing pairs") as pbar:

        index = -2
        for character in settings.characters:
            index += 1

            for situation in settings.situations:
                index += 1
                pbar.update(1)

                if index % every_x == 0:  # process every X situation
                    messages: ChatMessages = []
                    scores: Dict[str, List[int]] = defaultdict(list)
                    has_refusal = False
                    record_key = compose_key(character=character, situation=situation)
                    if record_key in existing_keys:
                        print(f"Existing key: {record_key}")
                        continue
                    try:
                        for turn in range(situation.num_turns + 1):
                            pbar.set_description(f"Turn {turn}/{situation.num_turns} for situation")
                            output = run_tester(
                                character=character,
                                situation=situation,
                                messages=messages,
                                user_prompt_path=settings.user_prompt_path,
                                system_prompt_path=settings.system_prompt_path,
                                character_prompt_path=settings.character_prompt_path,
                                provider=tester_provider,
                            )

                            if messages:
                                if output.is_refusal:
                                    has_refusal = True
                                    break
                                output_scores = output.get_scores()
                                for key, score in output_scores.items():
                                    scores[key].append(score)
                            if turn == situation.num_turns:
                                break
                            messages.append({"role": "user", "content": output.next_user_utterance})
                            bot_message = run_testee(
                                provider=testee_provider,
                                messages=messages,
                                character=character,
                                character_prompt_path=settings.character_prompt_path,
                            )
                            messages.append({"role": "assistant", "content": bot_message})
                        final_output = {
                            **output.to_dict(),
                            "messages": messages,
                            "character": character.to_dict(),
                            "situation": situation.to_dict(),
                            "has_refusal": has_refusal,
                            "scores": scores,
                        }
                        outputs.append(final_output)
                    except Exception:
                        traceback.print_exc()
                        time.sleep(30)
                        continue

                    save(
                        output_path=output_path,
                        outputs=outputs,
                        tester_provider=tester_provider,
                        testee_provider=testee_provider,
                        version=settings.version,
                    )


if __name__ == "__main__":
    fire.Fire(run_eval)
