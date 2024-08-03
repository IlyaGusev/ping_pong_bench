import json
from statistics import mean
import html
import base64
import uuid


def generate_html(data):
    characters = list(set(output['character']['char_name'] for output in data['outputs']))
    situations = list(set(output['situation']['text'] for output in data['outputs']))

    grouped_outputs = {situation: [] for situation in situations}
    for output in data['outputs']:
        grouped_outputs[output['situation']['text']].append(output)

    scores = {situation: {char: [] for char in characters} for situation in situations}
    dialogs = {}

    for situation, outputs in grouped_outputs.items():
        for output in outputs:
            char_name = output['character']['char_name']
            avg_score = mean([
                mean(output['scores']['stay_in_character']),
                mean(output['scores']['language_fluency']),
                mean(output['scores']['entertainment'])
            ])
            scores[situation][char_name].append(avg_score)

            # Save dialog data with base64 encoded key
            key = base64.b64encode(f"{char_name}::{situation}".encode('utf-8')).decode('utf-8')
            dialogs[key] = output['messages']

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Character Scores and Dialogs</title>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .average { font-weight: bold; background-color: #e6e6e6; }
            .dialog { margin-top: 20px; border: 1px solid #ddd; padding: 10px; }
            .user { color: blue; }
            .assistant { color: green; }
        </style>
    </head>
    <body>
        <table id="scoreTable">
            <tr>
                <th>Situation</th>
    """

    for char in characters:
        html_content += f"<th>{html.escape(char)}</th>"
    html_content += "<th>Среднее</th></tr>"

    character_averages = {char: [] for char in characters}
    for situation in situations:
        truncated_situation = html.escape(situation[:50] + '...' if len(situation) > 50 else situation)
        html_content += f"<tr><td>{truncated_situation}</td>"

        situation_scores = []
        for char in characters:
            char_scores = scores[situation][char]
            if char_scores:
                avg_score = mean(char_scores)
                character_averages[char].append(avg_score)
                situation_scores.append(avg_score)
                dialog_key = base64.b64encode(f"{char}::{situation}".encode('utf-8')).decode('utf-8')
                html_content += f'<td><a href="#" onclick="showDialog(\'{dialog_key}\')">{avg_score:.2f}</a></td>'
            else:
                html_content += "<td>N/A</td>"

        situation_average = mean(situation_scores) if situation_scores else 0
        html_content += f"<td class='average'>{situation_average:.2f}</td></tr>"

    html_content += "<tr><td class='average'>Среднее по персонажу</td>"
    overall_averages = []
    for char in characters:
        char_average = mean(character_averages[char]) if character_averages[char] else 0
        overall_averages.append(char_average)
        html_content += f"<td class='average'>{char_average:.2f}</td>"

    overall_average = mean(overall_averages) if overall_averages else 0
    html_content += f"<td class='average'>{overall_average:.2f}</td></tr>"

    html_content += """
        </table>
        <div id="dialogContainer" class="dialog"></div>
        <script>
        const dialogs = """

    html_content += json.dumps(dialogs)

    html_content += """
        function showDialog(key) {
            const dialog = dialogs[key];
            if (!dialog) {
                document.getElementById('dialogContainer').innerHTML = 'Dialog not found';
                return;
            }
            let dialogHtml = '<h3>Dialog</h3>';
            for (const message of dialog) {
                dialogHtml += '<p class="' + message.role + '"><strong>' + message.role + ':</strong> ' + message.content + '</p>';
            }
            document.getElementById('dialogContainer').innerHTML = dialogHtml;
        }
        </script>
    </body>
    </html>
    """

    return html_content


# Load JSON data
with open('../results/gemma-2-27b-it.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Generate HTML
html_output = generate_html(data)

# Write HTML to file
with open('../output.html', 'w', encoding='utf-8') as file:
    file.write(html_output)

print("HTML file 'output.html' has been generated.")
