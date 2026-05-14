import json
from typing import List


def load_ranking_selection_dataset(path: str) -> List[dict]:
    cases = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))

    return cases