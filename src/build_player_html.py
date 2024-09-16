import json
import html
import base64
import math
from statistics import mean
from typing import List, Dict, Any, Union, Optional

import fire  # type: ignore

from src.util import encode_prompt


def generate_html(data: Dict[str, Any], template_path: str = "templates/player_page.jinja") -> str:
    characters: List[str] = sorted(set(o["character"]["char_name"] for o in data["outputs"]))
    situations: List[str] = sorted(set(o["situation"]["text"] for o in data["outputs"]))
    keys: Dict[str, Dict[str, str]] = {situation: {} for situation in situations}
    grouped_outputs: Dict[str, Dict[str, Dict[str, Any]]] = {
        situation: {} for situation in situations
    }
    for output in data["outputs"]:
        situation = output["situation"]["text"]
        char_name = output["character"]["char_name"]
        grouped_outputs[situation][char_name] = output

    scores: Dict[str, Dict[str, Optional[float]]] = {
        situation: {char: None for char in characters} for situation in situations
    }
    dialogs: Dict[str, Any] = {}

    for situation, char_outputs in grouped_outputs.items():
        for char_name, output in char_outputs.items():
            assert "scores" in output
            example_scores = output["scores"]
            is_refusal = False
            if "is_refusal" in example_scores and max(example_scores["is_refusal"]) == 1:
                is_refusal = True
            if "has_refusal" in output and output["has_refusal"]:
                is_refusal = True
            if is_refusal:
                scores[situation][char_name] = None
            else:
                scores[situation][char_name] = mean(
                    [
                        mean(
                            example_scores.get(
                                "stay_in_character", example_scores.get("in_character", [])
                            )
                        ),
                        mean(
                            example_scores.get(
                                "language_fluency", example_scores.get("fluency", [])
                            )
                        ),
                        mean(
                            example_scores.get(
                                "entertainment", example_scores.get("entertaining", [])
                            )
                        ),
                    ]
                )
            key = base64.b64encode(f"{char_name}::{situation}".encode("utf-8")).decode("utf-8")
            dialogs[key] = {
                "messages": output["messages"],
                "character": char_name,
                "situation": situation,
                "scores": example_scores,
            }
            keys[situation][char_name] = key

    character_scores: Dict[str, List[float]] = {char: [] for char in characters}
    situation_scores: Dict[str, List[float]] = {situation: [] for situation in situations}
    situation_averages: Dict[str, Optional[float]] = dict()
    character_averages: Dict[str, Optional[float]] = dict()
    for situation in situations:
        for char in characters:
            score = scores[situation][char]
            if score is not None and score > 0.0:
                character_scores[char].append(score)
                situation_scores[situation].append(score)
    for char, s in character_scores.items():
        character_averages[char] = mean(s) if s else None
    for situation, s in situation_scores.items():
        situation_averages[situation] = mean(s) if s else None

    overall_scores: List[float] = []
    for char in characters:
        char_average = character_averages[char]
        if char_average is not None and not math.isnan(char_average):
            overall_scores.append(char_average)
    overall_average = mean(overall_scores) if overall_scores else None

    html_content = encode_prompt(
        template_path,
        characters=characters,
        situations=situations,
        player=data.get("testee", data.get("player")),
        judge=data.get("tester", data.get("judge")),
        interrogator=data.get("interrogator"),
        keys=keys,
        scores=scores,
        situation_averages=situation_averages,
        character_averages=character_averages,
        overall_average=overall_average,
        dialogs=json.dumps(dialogs),
    )
    return html_content


def run_build_html(
    json_path: str,
    output_path: str,
) -> None:
    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Generate HTML
    html_output = generate_html(data)

    # Write HTML to file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(html_output)

    print(f"HTML file '{output_path}' has been generated.")


if __name__ == "__main__":
    fire.Fire(run_build_html)
