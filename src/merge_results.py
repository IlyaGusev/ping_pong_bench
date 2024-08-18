import json
import os
import sys
import random

input_path = sys.argv[1]
output_path = sys.argv[2]
records = []
for name in os.listdir(input_path):
    with open(os.path.join(input_path, name)) as r:
        records += json.load(r)["outputs"]
random.shuffle(records)
with open(output_path, "w") as w:
    for r in records:
        w.write(json.dumps(r, ensure_ascii=False) + "\n")
