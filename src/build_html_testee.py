import json
import html
import base64
import math
from statistics import mean
from typing import List, Dict, Any, Union, Optional

import fire  # type: ignore


def safe_mean(lst: List[float]) -> float:
    return mean(lst) if lst else float("nan")


def format_score_safe(score: Optional[float]) -> str:
    if score is None:
        return "-"
    if math.isnan(score):
        return "NaN"
    return f"{score:.2f}"


def generate_html(data: Dict[str, Any]) -> str:
    characters: List[str] = sorted(
        set(output["character"]["char_name"] for output in data["outputs"])
    )
    situations: List[str] = sorted(set(output["situation"]["text"] for output in data["outputs"]))

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
            if "has_refusal" in output and output["has_refusal"]:
                scores[situation][char_name] = float("nan")
            elif "scores" in output:
                if output["scores"]:
                    escores = output["scores"]
                    avg_score = safe_mean(
                        [
                            safe_mean(escores.get("stay_in_character", escores.get("in_character", []))),
                            safe_mean(escores.get("language_fluency", escores.get("fluency", []))),
                            safe_mean(escores.get("entertainment", escores.get("entertaining", []))),
                        ]
                    )
                    scores[situation][char_name] = avg_score
                else:
                    scores[situation][char_name] = float("nan")

            key = base64.b64encode(f"{char_name}::{situation}".encode("utf-8")).decode("utf-8")
            dialogs[key] = {
                "messages": output["messages"],
                "character": char_name,
                "situation": situation,
            }

    html_content = """
    <style>
        h3 { margin: 10px 0px 10px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid black; padding: 8px; text-align: left; }
        .average { font-weight: bold; }
        .dialog { margin-top: 20px; border: 1px solid #ddd; padding: 10px; }
        .user { color: #6a9fb5; }
        .assistant { color: #90a959; }
    </style>
    """

    html_content += (
        "<h3>Judge</h3><code>"
        + json.dumps(data.get("tester", data.get("judge")), ensure_ascii=False)
        + "</code>"
    )
    html_content += (
        "<h3>Player</h3><code>"
        + json.dumps(data.get("testee", data.get("player")), ensure_ascii=False)
        + "</code>"
    )

    html_content += """
    <h3>Scores</h3>
    <table id="scoreTable">
        <tr>
            <th>Situation</th>
    """

    for char in characters:
        html_content += f"<th>{html.escape(char)}</th>"
    html_content += "<th>Average by situation</th></tr>"

    character_averages: Dict[str, List[float]] = {char: [] for char in characters}
    for situation in situations:
        truncated_situation = html.escape(
            situation[:50] + "..." if len(situation) > 50 else situation
        )
        html_content += f"<tr><td>{truncated_situation}</td>"

        situation_scores: List[float] = []
        for char in characters:
            score = scores[situation][char]
            if score is not None:
                if not math.isnan(score):
                    character_averages[char].append(score)
                    situation_scores.append(score)
                    dialog_key = base64.b64encode(f"{char}::{situation}".encode("utf-8")).decode(
                        "utf-8"
                    )
                    html_content += f'<td><a href="#" onclick="showDialog(event, \'{dialog_key}\')">{format_score_safe(score)}</a></td>'
                else:
                    html_content += "<td>REF</td>"
            else:
                html_content += "<td>-</td>"

        situation_average = safe_mean(situation_scores) if situation_scores else None
        html_content += f"<td class='average'>{format_score_safe(situation_average)}</td></tr>"

    html_content += "<tr><td class='average'>Average by character</td>"
    overall_averages: List[float] = []
    for char in characters:
        char_average = safe_mean(character_averages[char]) if character_averages[char] else None
        if char_average is not None and not math.isnan(char_average):
            overall_averages.append(char_average)
        html_content += f"<td class='average'>{format_score_safe(char_average)}</td>"

    overall_average = safe_mean(overall_averages) if overall_averages else None
    html_content += f"<td class='average'>{format_score_safe(overall_average)}</td></tr>"

    html_content += """
        </table>
        <div id="dialogContainer" class="dialog" hidden></div>
        <script>
        const dialogs = """

    html_content += json.dumps(dialogs)

    html_content += """
        function showDialog(e, key) {
            e = e || window.event;
            e.preventDefault();
            const dialog = dialogs[key]["messages"];
            if (!dialog) {
                document.getElementById('dialogContainer').innerHTML = 'Dialog not found';
                return;
            }
            let dialogHtml = '';
            dialogHtml += '<p>Character: ' + dialogs[key]["character"] + '</p>';
            dialogHtml += '<p>Situation: ' + dialogs[key]["situation"] + '</p>';
            dialogHtml += '<h3>Dialog</h3>';
            for (const message of dialog) {
                dialogHtml += '<p class="' + message.role + '"><strong>' + message.role + ':</strong> ' + message.content + '</p>';
            }
            document.getElementById('dialogContainer').innerHTML = dialogHtml;
            document.getElementById('dialogContainer').removeAttribute("hidden");
            document.getElementById('dialogContainer').scrollIntoView();
        }
        </script>
    """

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
