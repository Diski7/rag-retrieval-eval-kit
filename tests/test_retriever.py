"""Tests. No third party dependencies, so they run anywhere python3 runs:

    python3 -m unittest discover -s tests -v

The important test is `BoostCannotCreateAHit`, which pins a real production bug so
it cannot come back.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import HashingEmbedder, known_words, load_chunks, load_json, uncovered_entity
from retrieval.config import MIN_COSINE
from retrieval.loader import GOLD, PROBES
from retrieval.retriever import Chunk, Retriever


def two_program_index(min_cosine=MIN_COSINE):
    chunks = [
        Chunk(id="alpha::overview", source="alpha", label="Alpha College",
              division="women", section="overview",
              text="Alpha College women's team plays a patient possession game and "
                   "builds from the goalkeeper."),
        Chunk(id="beta::overview", source="beta", label="Beta College",
              division="men", section="overview",
              text="Beta College defends deep in a low block and counters at speed."),
    ]
    return Retriever(chunks, embedder=HashingEmbedder(), min_cosine=min_cosine)


class BoostCannotCreateAHit(unittest.TestCase):
    """Regression test for the bug that motivated gating on the raw cosine.

    The gate used to read the boosted score. A question about a team the library
    does not cover, phrased with the word "women's", picked up a 0.25 division
    boost on every women's chunk in the index, and that was enough to walk it past
    the gate and get answered with the nearest program. The fix is one line: gate
    on the raw cosine, let boosts reorder only.

    This test places the gate strictly between the raw value and the boosted one,
    which is exactly the region where the old code misbehaved.
    """

    def test_boosted_score_does_not_open_the_gate(self):
        retriever = two_program_index(min_cosine=0.0)
        query = "What formation does Real Madrid women's play?"

        open_gate = retriever.retrieve(query, k=2, min_cosine=0.0)
        self.assertTrue(open_gate, "the test needs a candidate to reason about")
        top = open_gate[0]

        raw, boosted = top.cosine, top.score
        self.assertGreater(boosted, raw, "the query should attract a boost")

        gate = (raw + boosted) / 2.0
        self.assertLess(raw, gate, "raw similarity sits below the chosen gate")
        self.assertGreaterEqual(boosted, gate, "the boosted score clears it")

        # Old behaviour: the boosted score clears the gate, so this returned a
        # profile of Alpha College for a question about Real Madrid.
        # Fixed behaviour: the gate reads the raw cosine, so nothing is returned.
        gated = two_program_index(min_cosine=gate).retrieve(query)
        self.assertEqual(gated, [], "a boost must never create a hit")


class GateInvariants(unittest.TestCase):

    def test_every_hit_clears_the_gate_on_the_raw_cosine(self):
        retriever = two_program_index()
        for query in ["What does Alpha College do in possession?",
                      "How does Beta College defend?",
                      "What formation does Alpha College play?"]:
            for hit in retriever.retrieve(query, k=5):
                self.assertGreaterEqual(hit.cosine, MIN_COSINE)

    def test_an_off_topic_question_returns_nothing(self):
        retriever = two_program_index()
        self.assertEqual(retriever.retrieve("How do I build a sourdough starter?"), [])
        self.assertEqual(retriever.retrieve("What is the capital of France?"), [])

    def test_a_relevant_question_still_returns_something(self):
        retriever = two_program_index()
        self.assertTrue(retriever.retrieve("How does Alpha College build from the back?"))


class PerSourceCap(unittest.TestCase):

    def test_one_program_cannot_fill_every_slot(self):
        chunks = [
            Chunk(id=f"alpha::{s}", source="alpha", label="Alpha College",
                  division="women", section=s, text=f"Alpha College {s} detail.")
            for s in ("formations", "formations", "formations", "formations")
        ]
        chunks.append(Chunk(id="beta::formation", source="beta", label="Beta College",
                            division="women", section="formation",
                            text="Beta College formation detail."))
        retriever = Retriever(chunks, embedder=HashingEmbedder(), min_cosine=0.05)
        hits = retriever.retrieve("Alpha College formation", k=5, max_per_source=2)
        self.assertEqual(len([h for h in hits if h.chunk.source == "alpha"]), 2)


class CoverageCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()
        retriever = Retriever(cls.chunks, min_cosine=MIN_COSINE)
        cls.known = known_words(retriever.sources(), {c.label for c in cls.chunks})

    def test_names_the_uncovered_team_rather_than_answering(self):
        self.assertEqual(
            uncovered_entity("Who is the head coach of the Kingsbury Rovers reserves?",
                             self.known),
            "Kingsbury Rovers")

    def test_refuses_a_team_the_library_does_not_hold(self):
        self.assertEqual(
            uncovered_entity("What formation does Real Madrid women's play?", self.known),
            "Real Madrid")

    def test_a_covered_program_is_not_flagged(self):
        for query in ["What formation does Northgate University play?",
                      "How does Ashford College press?",
                      "Which positions does Elmstead Institute need?"]:
            self.assertIsNone(uncovered_entity(query, self.known), query)

    def test_possessive_is_trimmed_from_the_refusal(self):
        self.assertEqual(
            uncovered_entity("Tell me about Manchester United's academy", self.known),
            "Manchester United")


class GoldSet(unittest.TestCase):

    def test_hit_at_5_is_complete(self):
        chunks = load_chunks()
        retriever = Retriever(chunks, min_cosine=MIN_COSINE)
        known = known_words(retriever.sources(), {c.label for c in chunks})
        misses = []
        for row in load_json(GOLD)["queries"]:
            if uncovered_entity(row["q"], known):
                misses.append((row["q"], "refused a covered program"))
                continue
            hits = retriever.retrieve(row["q"], k=5)
            if not any(h.chunk.source == row["expect"] for h in hits):
                misses.append((row["q"], row["expect"]))
        self.assertEqual(misses, [], f"gold questions missed: {misses}")

    def test_every_probe_is_refused(self):
        chunks = load_chunks()
        retriever = Retriever(chunks, min_cosine=MIN_COSINE)
        known = known_words(retriever.sources(), {c.label for c in chunks})
        answered = []
        for probe in load_json(PROBES)["probes"]:
            if uncovered_entity(probe["q"], known):
                continue
            if retriever.retrieve(probe["q"], k=1):
                answered.append(probe["q"])
        self.assertEqual(answered, [], f"probes answered that should be refused: {answered}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
