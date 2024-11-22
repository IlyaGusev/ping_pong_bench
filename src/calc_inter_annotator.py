import json
import os
from statistics import mean, median
from collections import defaultdict
from typing import List, Dict, Any, Optional

import fire  # type: ignore
import numpy as np
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from pyvis.network import Network  # type: ignore
from scipy.stats import spearmanr # type: ignore
from krippendorff import alpha  # type: ignore


def main(input_dir: str, exclude_name: Optional[str] = None) -> None:
    ratings = defaultdict(list)
    header = ["in_character", "entertaining", "fluency", "final"]
    for name in os.listdir(input_dir):
        if not name.endswith(".jsonl"):
            continue
        if exclude_name and name == exclude_name:
            continue
        file_name = os.path.join(input_dir, name)
        with open(file_name) as r:
            for line in r:
                record = json.loads(line)
                key = str(record["messages"])
                human_scores = record["human_scores"]
                scores = [human_scores[k] for k in header if k != "final"]
                scores.append(mean(scores))
                ratings[key].append(np.array(scores))
    plain_ratings = []
    for _, rating in ratings.items():
        plain_ratings.append(np.array(rating))
    ratings_array = np.array(plain_ratings)
    for i, criterion in enumerate(header):
        criterion_ratings = ratings_array[:, :, i]
        criterion_ratings = np.transpose(criterion_ratings)
        k_alpha = alpha(reliability_data=criterion_ratings, level_of_measurement="ordinal")
        print("Alpha", criterion, k_alpha)


if __name__ == "__main__":
    fire.Fire(main)
