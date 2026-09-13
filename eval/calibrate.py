"""Pick the gate from the data instead of copying one.

A similarity threshold is a property of the embedder and the corpus, not a
universal constant. Amazon Titan at 1,024 dimensions and a lexical hashing
embedder at 256 do not put the same pair of texts at the same distance, so a
threshold carried between them either refuses good questions or lets bad ones
through. This script prints the band where a gate is safe, or says plainly that
there is no clean cut and a threshold alone will not be enough.

    python3 eval/calibrate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import known_words, load_chunks, load_json, uncovered_entity
from retrieval.loader import GOLD, PROBES
from retrieval.retriever import Retriever


def main():
    chunks = load_chunks()
    retriever = Retriever(chunks, min_cosine=0.0)   # gate open, measure only
    known = known_words(retriever.sources(), {c.label for c in chunks})
    gold = load_json(GOLD)["queries"]
    probes = load_json(PROBES)["probes"]

    print(f"{len(chunks)} chunks, {retriever.embedder.dim} dimensions, "
          f"embedder {type(retriever.embedder).__name__}")
    print()

    print("questions that should be answered, best raw cosine for each")
    gold_best = []
    for row in gold:
        hits = retriever.retrieve(row["q"], k=5, min_cosine=0.0)
        raw = max((h.cosine for h in hits), default=0.0)
        gold_best.append(raw)
    print(f"  min {min(gold_best):.3f}   median "
          f"{sorted(gold_best)[len(gold_best) // 2]:.3f}   max {max(gold_best):.3f}")
    print()

    print("questions that should be refused, best raw cosine for each")
    probe_best = []
    for probe in probes:
        named = uncovered_entity(probe["q"], known)
        hits = retriever.retrieve(probe["q"], k=5, min_cosine=0.0)
        raw = max((h.cosine for h in hits), default=0.0)
        if named:
            print(f"  {raw:.3f}  {probe['q']}")
            print(f"         caught by the coverage check: {named}")
        else:
            probe_best.append(raw)
            print(f"  {raw:.3f}  {probe['q']}")
    print()

    if not probe_best:
        print("every probe is caught by the coverage check, so the gate only has to")
        print("separate the gold questions from unrelated text.")
        print()
        print(f"safe window: any gate at or below {min(gold_best):.3f}")
        return 0

    low = max(probe_best)
    high = min(gold_best)
    print(f"highest cosine the gate must exclude: {low:.3f}")
    print(f"lowest cosine the gate must admit:    {high:.3f}")
    print()
    if low < high:
        suggested = round((low + high) / 2.0, 2)
        print(f"clean band {low:.3f} to {high:.3f}, {high - low:.3f} wide")
        print(f"suggested gate: {suggested}")
    else:
        print("NO CLEAN CUT. The two distributions overlap, so no single threshold")
        print("separates them and the threshold is doing work a threshold cannot do.")
        print("That is the case that needs a coverage check or a named entity check,")
        print("not a smaller number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
