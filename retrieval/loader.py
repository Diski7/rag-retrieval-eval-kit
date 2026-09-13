"""Load a corpus JSON file into chunks, and answer the coverage question."""
import json
import os

from .retriever import Chunk

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "corpus", "sample_programs.json")
PROBES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "corpus", "probes.json")
GOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "eval", "gold_queries.json")


def load_chunks(path=CORPUS):
    """One chunk per section. Section level chunking is deliberate: a whole
    document retrieves as one blob and drags unrelated detail into the answer."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    chunks = []
    for program in data["programs"]:
        for section, text in program["sections"].items():
            chunks.append(Chunk(
                id=f"{program['slug']}::{section}",
                source=program["slug"],
                label=program["label"],
                division=program["division"],
                section=section,
                text=text,
            ))
    return chunks


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
