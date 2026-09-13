"""Evaluation harness.

Answers two questions with numbers rather than assertions:

1. Of the questions this library should answer, how many does it get right, and in
   which position?
2. Of the questions it should refuse, how many does it actually refuse?

Both are needed. A retrieval system that scores 100 percent on the good questions
and answers the bad ones is worse than useless, because the wrong answers are the
confident ones.

    python3 eval/run_eval.py

Exits non zero if any probe is answered or the good questions drop below their
recorded score, so it can run as a regression gate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import known_words, load_chunks, load_json, uncovered_entity
from retrieval.config import MIN_COSINE
from retrieval.loader import GOLD, PROBES
from retrieval.retriever import Retriever


def answer(retriever, known, query, k=5):
    """The full path a question takes: coverage check, then retrieval."""
    named = uncovered_entity(query, known)
    if named:
        return [], f"refused: {named} is not in the library"
    hits = retriever.retrieve(query, k=k)
    if not hits:
        return [], "refused: nothing in the library is close enough"
    return hits, "answered"


def main():
    chunks = load_chunks()
    retriever = Retriever(chunks, min_cosine=MIN_COSINE)
    known = known_words(retriever.sources(), {c.label for c in chunks})
    gold = load_json(GOLD)["queries"]
    probes = load_json(PROBES)["probes"]

    print(f"index: {len(chunks)} chunks from {len(retriever.sources())} programs, "
          f"{retriever.embedder.dim} dimensions, gate {MIN_COSINE}")
    print()

    hits_at = {1: 0, 3: 0, 5: 0}
    section_ok = 0
    reciprocal = 0.0
    top_cosines = []
    failures = []

    for row in gold:
        hits, status = answer(retriever, known, row["q"])
        rank = None
        for position, hit in enumerate(hits, start=1):
            if hit.chunk.source == row["expect"]:
                rank = position
                break
        if rank:
            for k in hits_at:
                if rank <= k:
                    hits_at[k] += 1
            reciprocal += 1.0 / rank
            if hits[rank - 1].chunk.section in row["sections"]:
                section_ok += 1
            else:
                failures.append((row["q"], f"right program, wrong section: "
                                           f"{hits[rank - 1].chunk.section}"))
        else:
            failures.append((row["q"], f"no hit for {row['expect']} ({status})"))
        if hits:
            top_cosines.append(hits[0].cosine)

    total = len(gold)
    print(f"gold questions: {total}")
    print(f"  hit@1 {hits_at[1]}/{total}   hit@3 {hits_at[3]}/{total}   "
          f"hit@5 {hits_at[5]}/{total}")
    print(f"  right section {section_ok}/{total}   "
          f"MRR {reciprocal / total:.3f}")
    print()

    refused = 0
    probe_cosines = []
    for probe in probes:
        hits, status = answer(retriever, known, probe["q"])
        if not hits:
            refused += 1
            print(f"  refused  {probe['q']}")
        else:
            print(f"  ANSWERED {probe['q']}  ->  {hits[0].chunk.label} "
                  f"({hits[0].chunk.section}) raw {hits[0].cosine:.3f} "
                  f"boosted {hits[0].score:.3f}")
            probe_cosines.append(hits[0].cosine)

    print()
    print(f"probes: {len(probes)}, refused {refused}/{len(probes)}")
    if probe_cosines:
        print(f"  worst answered probe had raw cosine {max(probe_cosines):.3f}")
    print()
    print("score separation")
    print(f"  lowest top hit cosine across gold questions: {min(top_cosines):.3f}")
    print(f"  gate: {MIN_COSINE}")

    if failures:
        print()
        print("failures")
        for query, why in failures:
            print(f"  {query}  ->  {why}")

    ok = True
    if hits_at[5] != total:
        print(f"\nFAIL hit@5 {hits_at[5]}/{total}, expected {total}")
        ok = False
    if refused != len(probes):
        print(f"\nFAIL answered {len(probes) - refused} of {len(probes)} probes")
        ok = False
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
