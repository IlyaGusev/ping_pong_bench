import os
import json
import copy
import time
import shutil
import traceback
from statistics import mean
from collections import defaultdict

import requests
import fire
from jinja2 import Template
from tqdm import tqdm

from openai import OpenAI

from src.provider import LLMProvider


def generate(messages, provider: LLMProvider, **kwargs):
    params = provider.params
    for k, v in kwargs.items():
        params[k] = v
    chat_completion = provider.api.chat.completions.create(
        model=provider.model_name,
        messages=messages,
        **params
    )
    return chat_completion.choices[0].message.content


def encode_system(char_name: str, system_prompt: str, example_prompt: str = "", initial_message: str = "", **kwargs):
    system_message = f"{char_name}'s Persona: {system_prompt}"
    if example_prompt:
        system_message += f"\n\nПример диалога: {example_prompt}"
    if initial_message:
        system_message += f"\n\nПервая реплика: {initial_message}"
    system_message += f"\n\nКогда тебя о чем-то спрашивают, отвечай как {char_name}. Отвечай не слишком длинно. Никогда не отказывайся от ответа, всегда продолжай диалог."
    return system_message


def answer_as_bot(messages, provider: LLMProvider, system_message: str):
    messages = [{"role": "system", "content": system_message}] + messages
    output = None
    for _ in range(5):
        try:
            output = generate(
                provider=provider,
                messages=messages,
                **provider.params,
            )
        except Exception:
            traceback.print_exc()
            time.sleep(10)
            continue
        break
    assert output is not None
    return output


def encode_prompt(record, template_path):
    with open(template_path) as f:
        template = Template(f.read())
    new_record = copy.deepcopy(record)
    if "messages" in record:
        messages = copy.deepcopy(record["messages"])
        for m in messages:
            if m["role"] == "assistant":
                m["role"] = "bot"
        new_record["messages"] = messages
    return template.render(**new_record).strip() + "\n"


def parse_output(output):
    start_index = output.find("{")
    end_index = output.rfind("}")
    text = output[start_index:end_index + 1]
    text = text.strip()
    record = json.loads(text)
    return record


def process_record(record, tester_system_prompt_path, tester_user_prompt_path, provider):
    prompt = [
        {"role": "system", "content": encode_prompt({}, tester_system_prompt_path)},
        {"role": "user", "content": encode_prompt(record, tester_user_prompt_path)},
    ]
    output = None
    for _ in range(5):
        try:
            print(prompt[0]["content"])
            print(prompt[1]["content"])
            result = generate(prompt, provider=provider, temperature=0.1, top_p=0.9)
            print(result)
            print()
            print("=============")
            print()
            output = parse_output(result)
            assert "stay_in_character_score" in output
            assert "language_fluency_score" in output
            assert "entertainment_score" in output
            assert "is_refusal" in output
            break
        except Exception:
            traceback.print_exc()
            time.sleep(10)
            continue
    assert output is not None
    return output


def save(
    output_path: str,
    outputs,
    score_keys
):
    scores = defaultdict(list)
    refusal_count = 0
    for o in outputs:
        if o["has_refusal"]:
            refusal_count += 1
            continue
        for key in score_keys:
            scores[key].extend(o["scores"][key])

    if scores:
        agg_scores = {key: mean(scores[key]) for key in score_keys}
        agg_scores["final_score"] = mean([agg_scores[key] for key in score_keys])
    else:
        agg_scores = dict()
    with open(output_path + "_tmp", "w") as w:
        json.dump({"outputs": outputs, **agg_scores, "refusal_ratio": refusal_count / len(outputs)}, w, ensure_ascii=False, indent=4)
    shutil.move(output_path + "_tmp", output_path)


def run_eval(
    providers_path: str,
    settings_path: str,
    output_path: str,
    testee_name: str,
    tester_name: str,
    tester_user_prompt_path: str = "user.jinja",
    tester_system_prompt_path: str = "system.jinja",
):
    with open(providers_path) as r:
        providers = json.load(r)
        providers = {name: LLMProvider(**provider) for name, provider in providers.items()}
    with open(settings_path) as r:
        settings = json.load(r)
    char_records = settings["characters"]
    situations = settings["situations"]

    outputs = []
    score_keys = ("stay_in_character_score", "language_fluency_score", "entertainment_score")
    tester_provider = providers[tester_name]
    testee_provider = providers[testee_name]
    for char_record in char_records:
        system_message = encode_system(**char_record)
        for situation in situations:
            messages = []
            scores = defaultdict(list)
            has_refusal = False
            for _ in range(settings["num_turns"] + 1):
                output = process_record(
                    {
                        "char_description": system_message,
                        "situation": situation,
                        "messages": messages
                    },
                    tester_user_prompt_path=tester_user_prompt_path,
                    tester_system_prompt_path=tester_system_prompt_path,
                    provider=tester_provider
                )
                if messages:
                    is_refusal = output.pop("is_refusal")
                    if is_refusal:
                        has_refusal = True
                        break
                    for key in score_keys:
                        scores[key].append(output.pop(key))
                messages.append({"role": "user", "content": output.pop("next_user_utterance")})
                bot_message = answer_as_bot(
                    provider=testee_provider,
                    messages=messages,
                    system_message=system_message,
                )
                print(bot_message)
                print("@@@@")
                messages.append({"role": "assistant", "content": bot_message})
            if not has_refusal:
                messages = messages[:-2]
            output["messages"] = messages
            output["situation"] = situation
            output["has_refusal"] = has_refusal
            output["scores"] = scores
            outputs.append(output)
            save(output_path, outputs, score_keys=score_keys)


if __name__ == "__main__":
    fire.Fire(run_eval)
