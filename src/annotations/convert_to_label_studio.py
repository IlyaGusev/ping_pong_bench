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
    user_color = "#6a9fb5"
    assistant_color = "#4f6b12"
    template = "<p{style}><strong>{role}</strong>:</p>\n<p{style}>"
    user_style = f' style="color: {user_color};"'
    assistant_style = f' style="color: {assistant_color};"'
    html = html.replace(template.format(style="", role="Assistant"), template.format(style=assistant_style, role="Assistant"))
    html = html.replace(template.format(style="", role="User"), template.format(style=user_style, role="User"))
    print(html)
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
