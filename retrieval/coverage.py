"""Coverage check: refuse a named source the library does not hold.

The gate in `retriever.py` refuses a query with no relevance to the corpus. On its
own that is not enough. A query naming a real team this library does not cover is
often relevant to the DOMAIN: "what formation does X play" shares vocabulary with
every formation section in the index, so it retrieves something that clears the
gate. Left alone, the system answers about whichever program scored nearest. That
is a confident answer to a question nobody asked, which is worse than a refusal.

So a named entity that matches nothing in the index is refused explicitly and by
name, before retrieval runs.
"""
import re

# Words that carry no identifying information in a query about a program.
FILLER = re.compile(
    r"\b(who|what|which|where|when|how|is|are|was|does|do|did|the|of|for|a|an|"
    r"about|tell|me|us|coach|coaches|team|teams|squad|roster|play|plays|playing|"
    r"academy|record|season|program|programs|college|university|school)\b",
    re.I,
)
DIVISION_WORDS = re.compile(r"\b(men|men's|mens|women|women's|womens)\b", re.I)


def _runs(text):
    """Consecutive capitalised words, two or more.

    One capitalised word is too weak a signal (it catches sentence openers and
    single words like a city). Two in a row is almost always a name.
    """
    stripped = DIVISION_WORDS.sub(" ", FILLER.sub(" ", text))
    words = re.findall(r"[A-Za-z][A-Za-z'.-]*", stripped)
    runs = []
    current = []
    for word in words:
        if word[:1].isupper():
            current.append(word)
        else:
            if len(current) >= 2:
                runs.append(" ".join(current))
            current = []
    if len(current) >= 2:
        runs.append(" ".join(current))
    return runs


def known_words(sources, labels):
    """Every word that appears in a covered slug or label, lowercased."""
    known = set()
    for text in list(sources) + list(labels):
        known.update(w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2)
    known -= {"college", "university", "institute", "state", "school", "the", "and"}
    return known


def uncovered_entity(query, known):
    """Return the first name in the query that matches no covered source, else None."""
    for phrase in _runs(query):
        words = [w for w in re.findall(r"[a-z]+", phrase.lower()) if len(w) > 2]
        if not words:
            continue
        if any(w in known for w in words):
            continue
        return re.sub(r"'s$", "", phrase)
    return None
