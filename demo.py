"""One command demo:

    python3 demo.py

Walks a question through the whole path and prints the numbers behind each
decision, so the gate is visible rather than asserted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import known_words, load_chunks, uncovered_entity
from retrieval.config import MIN_COSINE
from retrieval.retriever import Retriever

QUESTIONS = [
    "What formation does Northgate University play?",
    "How does Ashford College press?",
    "Which positions does Dunmoor University need?",
]

REFUSALS = [
    "What formation does Real Madrid women's play?",
    "Who is the head coach of the Kingsbury Rovers reserves?",
    "How do I build a sourdough starter?",
]


def main():
    chunks = load_chunks()
    retriever = Retriever(chunks, min_cosine=MIN_COSINE)
    known = known_words(retriever.sources(), {c.label for c in chunks})

    print(f"library: {len(retriever.sources())} programs, {len(chunks)} chunks, "
          f"{retriever.embedder.dim} dims, gate {MIN_COSINE}")
    print()

    for query in QUESTIONS:
        print(f"Q  {query}")
        hits = retriever.retrieve(query, k=3)
        for position, hit in enumerate(hits, start=1):
            boost = ", ".join(f"{k} +{v:.2f}" for k, v in hit.boosts.items()) or "none"
            print(f"   {position}. {hit.chunk.label} / {hit.chunk.section}   "
                  f"cosine {hit.cosine:.3f}   score {hit.score:.3f}   boosts {boost}")
        print()

    for query in REFUSALS:
        print(f"Q  {query}")
        named = uncovered_entity(query, known)
        if named:
            print(f"   refused: {named} is not in the library")
        else:
            probe = retriever.retrieve(query, k=1, min_cosine=0.0)
            best = f"{probe[0].cosine:.3f}" if probe else "n/a"
            print(f"   refused: nothing clears the gate (closest cosine {best})")
        print()

    print("A refusal is the product working. The alternative is a confident")
    print("profile of a program the question was not about.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
