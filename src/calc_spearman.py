import sys
import json
from statistics import mean

from scipy.stats import spearmanr  # type: ignore

input_path = sys.argv[1]
output_path = sys.argv[2]
with open(input_path) as r:
    records = [json.loads(line) for line in r]
    records = [rec for rec in records if "human_scores" in rec]

avg_human_in_char = [r["human_scores"]["in_character"] for r in records]
avg_model_in_char = [mean(r["scores"]["stay_in_character"]) for r in records]
avg_human_entertaining = [r["human_scores"]["entertaining"] for r in records]
avg_model_entertaining = [mean(r["scores"]["entertainment"]) for r in records]
avg_human_entertaining = [r["human_scores"]["entertaining"] for r in records]
avg_model_entertaining = [mean(r["scores"]["entertainment"]) for r in records]
avg_human_fluency = [r["human_scores"]["fluency"] for r in records]
avg_model_fluency = [mean(r["scores"]["language_fluency"]) for r in records]
assert len(avg_human_in_char) == len(avg_model_in_char)
assert len(avg_human_entertaining) == len(avg_model_entertaining)
assert len(avg_human_in_char) == len(avg_human_entertaining)
assert len(avg_human_fluency) == len(avg_model_fluency)
assert len(avg_human_in_char) == len(avg_human_fluency)

avg_human = [
    sum([c, e, f]) for c, e, f in zip(avg_human_in_char, avg_human_entertaining, avg_human_fluency)
]
avg_model = [
    sum([c, e, f]) for c, e, f in zip(avg_model_in_char, avg_model_entertaining, avg_model_fluency)
]
assert len(avg_human) == len(avg_model)
print(len(avg_human_in_char))
print(spearmanr(avg_human_in_char, avg_model_in_char)[0])
print(spearmanr(avg_human_entertaining, avg_model_entertaining)[0])
print(spearmanr(avg_human_fluency, avg_model_fluency)[0])
print(spearmanr(avg_human, avg_model)[0])
