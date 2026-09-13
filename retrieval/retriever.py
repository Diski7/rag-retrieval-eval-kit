"""Grounded retrieval over a small document library.

Three mechanisms, in the order they run:

1. Cosine similarity between the query vector and every chunk vector.
2. Boosts, which adjust the RANK of candidates that are plausibly about the query
   (matching the division the question asks about, sharing a word with the source
   name). Boosts reorder. They never create a candidate.
3. A per source cap, so one long program cannot fill every slot.

The rule that matters is in `retrieve`. The relevance gate is applied to the RAW
cosine, not to the boosted score. That is a correction, and it was a real bug:
boosting before gating let a query about a team this library does not cover ride a
division boost straight past the gate and get answered with the nearest program
that happened to share a division. `tests/test_retriever.py` pins that behaviour.
"""
from dataclasses import dataclass, field
import re

from .embedder import HashingEmbedder, cosine, tokens

# Facet words. A question that names a facet should prefer that facet, which is what
# breaks ties between two sections of the same program that share the same words.
SECTION_HINTS = {
    "formation": {"formation", "shape", "system", "lineup"},
    "press": {"press", "pressing", "block", "defend", "defends", "defending", "counter"},
    "recruiting": {"recruit", "recruiting", "recruitment", "camp", "camps", "scholarship"},
    "positions": {"position", "need", "depth", "gap", "roster"},
    "overview": {"overview", "style", "identity", "philosophy"},
}


@dataclass
class Chunk:
    """One retrievable unit. Section level, never a whole document."""

    id: str
    source: str      # stable identifier for the program, used for the cap
    label: str       # display name
    division: str    # "men" or "women"
    section: str
    text: str

    @property
    def embed_text(self):
        """What actually gets embedded.

        The program name and the section heading are part of the indexed text on
        purpose. The body of a formation section rarely contains the word
        "formation", so without the heading a question about formation has nothing
        to match and the nearest paragraph wins on whatever words it happens to
        share. Indexing the heading is what makes section level retrieval work.
        """
        return f"{self.label} {self.section}. {self.text}"


@dataclass
class Hit:
    chunk: Chunk
    cosine: float    # raw similarity, what the gate reads
    score: float     # cosine plus boosts, what the ranking reads
    boosts: dict = field(default_factory=dict)


class Retriever:
    """In memory index. Load once, score in one pass, no vector database."""

    def __init__(self, chunks, embedder=None, boost_division=0.25, boost_label=0.18,
                 boost_section=0.12, min_cosine=0.28):
        self.chunks = list(chunks)
        self.embedder = embedder or HashingEmbedder()
        self.boost_division = boost_division
        self.boost_label = boost_label
        self.boost_section = boost_section
        self.min_cosine = min_cosine
        self.vectors = self.embedder.embed_batch([c.embed_text for c in self.chunks])

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _division_hint(query):
        """Which division the question is about, or None.

        Word boundaries matter here. Asking for a substring ("men" in the query)
        matches the "men" inside "women's", so both flags fire and the hint is
        dropped, which silently switches the boost off for exactly the questions it
        was written for.
        """
        q = query.lower()
        women = re.search(r"\b(women|womens|women's)\b", q) is not None
        men = re.search(r"\b(men|mens|men's)\b", q) is not None
        if women and not men:
            return "women"
        if men and not women:
            return "men"
        return None

    def _source_tokens(self, chunk):
        return set(tokens(chunk.label)) | set(tokens(chunk.source.replace("-", " ")))

    # -- the core ------------------------------------------------------------

    def retrieve(self, query, k=5, max_per_source=2, min_cosine=None):
        """Top k chunks for a query. Returns [] when nothing clears the gate.

        An empty list is a result, not a failure: it is what makes a grounded
        refusal possible instead of an answer invented from the nearest neighbour.
        """
        gate = self.min_cosine if min_cosine is None else min_cosine
        query_vector = self.embedder.embed(query)
        hint = self._division_hint(query)
        query_tokens = set(tokens(query))
        hinted_sections = {section for section, words in SECTION_HINTS.items()
                           if query_tokens & words}

        scored = []
        for index, vector in enumerate(self.vectors):
            chunk = self.chunks[index]
            cos = cosine(query_vector, vector)
            score = cos
            boosts = {}

            if hint and chunk.division == hint:
                score += self.boost_division
                boosts["division"] = self.boost_division

            if chunk.section in hinted_sections:
                score += self.boost_section
                boosts["section"] = self.boost_section

            overlap = query_tokens & self._source_tokens(chunk)
            if overlap:
                amount = self.boost_label * (len(overlap) / max(1, len(self._source_tokens(chunk))))
                score += amount
                boosts["label"] = amount

            scored.append((index, cos, score, boosts))

        # Rank on the boosted score, gate on the raw one. Sorting stays on the
        # boosted score so a division or name boost still orders neighbours well.
        scored.sort(key=lambda row: row[2], reverse=True)

        picked = []
        per_source = {}
        for index, cos, score, boosts in scored:
            if cos < gate:
                continue
            chunk = self.chunks[index]
            if per_source.get(chunk.source, 0) >= max_per_source:
                continue
            per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
            picked.append(Hit(chunk=chunk, cosine=cos, score=score, boosts=boosts))
            if len(picked) == k:
                break
        return picked

    def sources(self):
        """The programs this index can answer about."""
        return sorted({c.source for c in self.chunks})

    def can_answer(self, query, min_cosine=None):
        return bool(self.retrieve(query, k=1, min_cosine=min_cosine))
