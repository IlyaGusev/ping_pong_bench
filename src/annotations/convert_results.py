import fire
import json

mapping = {
    "Полностью не согласен": 1,
    "Не согласен": 2,
    "Не знаю": 3,
    "Согласен": 4,
    "Полностью согласен": 5,
}

def main(input_path, orig_path, output_path):
    orig_records = dict()
    with open(orig_path) as r:
        for idx, line in enumerate(r):
            orig_records[idx] = json.loads(line)

    with open(input_path) as r, open(output_path, "w") as w:
        records = json.load(r)
        for record in records:
            idx = record["data"]["idx"]
            annotations = record["annotations"][0]["result"]
            fluency_score = 3
            in_character_score = 3
            entertaining_score = 3
            for annot in annotations:
                if annot["from_name"] == "fluency":
                    fluency_score = mapping.get(annot["value"]["choices"][0], 3)
                elif annot["from_name"] == "in_character":
                    in_character_score = mapping.get(annot["value"]["choices"][0], 3)
                elif annot["from_name"] == "entertaining":
                    entertaining_score = mapping.get(annot["value"]["choices"][0], 3)
            human_scores = {
                "fluency": fluency_score,
                "in_character": in_character_score,
                "entertaining": entertaining_score
            }
            full_record = orig_records[idx]
            full_record["human_scores"] = human_scores
            w.write(json.dumps(full_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(main)
