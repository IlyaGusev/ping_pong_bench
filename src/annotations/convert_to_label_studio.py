import json
import csv
from typing import Dict, Any, List

import fire  # type: ignore
import markdown  # type: ignore


def to_markdown(record: Dict[str, Any]) -> str:
    result = ""
    messages = record["messages"]
    for m in messages:
        content = m["content"]
        result += "\n**{role}**:\n\n{content}\n\n".format(role=m["role"].capitalize(), content=content)
    return result


def markdown_to_html(text: str) -> str:
    html: str = markdown.markdown(text)
    html = "<link href='http://fonts.googleapis.com/css?family=Roboto' rel='stylesheet' type='text/css'>\n" + html
    user_color = "#6a9fb5"
    assistant_color = "#4f6b12"
    template = "<p{style}>"
    user_style = f''' style="color: {user_color}; font-family: 'Roboto', sans-serif;"'''
    assistant_style = f''' style="color: {assistant_color}; font-family: 'Roboto', sans-serif;"'''
    current_role = None
    new_lines = []
    for line in html.split("\n"):
        if "Assistant" in line:
            current_role = "assistant"
        elif "User" in line:
            current_role = "user"
        style = assistant_style if current_role == "assistant" else user_style
        line = line.replace(template.format(style=""), template.format(style=style))
        new_lines.append(line)
    html = "\n".join(new_lines)
    return html


def main(input_path: str, output_path: str) -> None:
    new_records: List[Dict[str, Any]] = []
    with open(input_path) as r:
        for idx, line in enumerate(r):
            record = json.loads(line)
            markdown = to_markdown(record)
            html = markdown_to_html(markdown)
            new_records.append({
                "markdown": markdown,
                "html": html,
                "char_info": record["character"]["system_prompt"],
                "char_name": record["character"]["char_name"],
                "idx": idx,
            })

    with open(output_path, "w") as w:
        writer = csv.writer(w)
        header = list(new_records[0].keys())
        writer.writerow(header)
        for rec in new_records:
            row = [rec[k] for k in header]
            writer.writerow(row)


if __name__ == "__main__":
    fire.Fire(main)
