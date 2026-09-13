# rag-retrieval-eval-kit

A small, dependency free reference for the part of a retrieval system that decides
whether to answer at all.

Everything here runs on stock python3 with no installs:

```
python3 demo.py                      # a question through the whole path
python3 eval/run_eval.py             # score the good questions and the bad ones
python3 eval/calibrate.py            # pick the similarity gate from the data
python3 -m unittest discover -s tests
```

Current state, from `eval/run_eval.py`:

```
index: 30 chunks from 6 programs, 256 dimensions, gate 0.31
gold questions: 24
  hit@1 24/24   hit@3 24/24   hit@5 24/24
  right section 23/24   MRR 1.000
probes: 5, refused 5/5
```

## Why this exists

I built a retrieval system that answers questions about college soccer programs
from a private library of team reports. It scored well on the questions it was
supposed to answer. Then I tested what it does with a question it should refuse:
someone asks about a team the library does not cover. It answered, confidently,
with the profile of a different school.

The cause was ordering. Relevance was gated on a score that already had boosts
folded into it, and one of those boosts was worth 0.25 for a chunk matching the
division the question asked about. Asking about a team that does not exist in the
library, phrased with the word "women's", lifted every women's chunk in the index
past the gate. The boost was doing a job it was never given: it was not reordering
candidates, it was creating them.

The fix is one line, and it is the idea this repository is built around:

```python
cos = cosine(query_vector, vector)   # raw similarity, what the gate reads
score = cos + boosts                 # ranking only, never the gate
```

Boosts reorder the candidates that were already relevant. They do not get a vote on
relevance. `tests/test_retriever.py::BoostCannotCreateAHit` pins that, with the gate
placed strictly between the raw and boosted values, which is exactly the region
where the old code misbehaved.

## What is in here

```
retrieval/embedder.py    hashing embedder for tests, Bedrock Titan adapter for production
retrieval/retriever.py   chunking, cosine, boosts, the gate, the per source cap
retrieval/coverage.py    refuse a named team the library does not hold
retrieval/config.py      the tunables, with where each number came from
eval/gold_queries.json  questions that must be answered, each with a known answer
corpus/probes.json       questions that must be refused, and what makes each dangerous
demo.py                  one question through the whole path, with the numbers
```

The corpus is synthetic. The programs in it are invented so that nothing here
resembles real recruiting intelligence, and every gold question has a right answer
that can be checked by reading the file. A retrieval score you cannot verify by
hand is decoration.

## Design decisions

**Chunks are sections, not documents.** A whole program profile retrieves as one
blob and drags unrelated detail into the answer. Section level chunks let the
retriever return the paragraph that was asked about.

**The section heading is part of the indexed text.** The body of a formation
section rarely contains the word "formation". Without the heading, a question about
formation has nothing to match and the nearest paragraph wins on whatever words it
happens to share.

**The gate reads the raw cosine.** See above.

**A named team that is not in the library is refused by name.** The gate cannot
catch this on its own, and this is the part I would have missed. A question about
an uncovered team is still relevant to the *domain*, so it retrieves something that
clears the gate, and the system answers about the wrong school. The check that
works is explicit: find the name, match it against the programs the index holds,
and refuse by name when nothing matches.

```
Q  What formation does Real Madrid women's play?
   refused: Real Madrid is not in the library
Q  What formation does Northgate University play?
   1. Northgate University / formation   cosine 0.364   score 0.664
```

**A per source cap.** Two chunks per program, so one long program cannot fill every
slot.

**An in memory index.** A few thousand chunks score in one pass. A vector database
at this size adds a dependency, a service and a failure mode in exchange for
nothing.

## The gate is measured, not chosen

`eval/calibrate.py` reports the band where a threshold is safe:

```
highest cosine the gate must exclude: 0.306
lowest cosine the gate must admit:    0.314
clean band 0.306 to 0.314, 0.008 wide
suggested gate: 0.31
```

That band is narrow, and it is narrow because the two distributions nearly touch.
It is also specific to this embedder and this corpus. The production system
gates at 0.28, where it embeds with Amazon Titan at 1,024 dimensions and the same
texts sit further apart.

I proved this the hard way while building this repository. Changing the indexed
text dropped hit@5 from 24/24 to 9/24 without touching the gate, because the
cosines moved and the threshold did not. Recalibrate whenever the embedder or the
corpus changes, or the gate quietly starts refusing good questions.

When the two distributions overlap and the band closes to nothing, a threshold is
being asked to do work a threshold cannot do. That is the case that needs a
coverage check, not a smaller number.

## Known limits

The bundled embedder matches on shared vocabulary. It is deterministic and free,
which is what makes the tests meaningful, but it is not a semantic model and the
numbers above are not evidence about semantic retrieval.

One gold question lands on the right program and the wrong section: "What formation
does Ashford College use?" returns the overview. The retriever ranks the formation
chunk into second place. The harness reports it rather than hiding it. Section
precision is where a real embedding model earns its cost.

There is no reranker and no answer synthesis. The kit stops at retrieved, gated
context, which is the layer this is meant to demonstrate.

## Production

Extracted from a private system that answers over a 56 program library, embedded
with Amazon Titan Text Embeddings v2 at 1,024 dimensions, evaluated on 75 questions
with a known right answer and 5 refusal probes. `retrieval/embedder.py` keeps the
Bedrock adapter next to the local one so the retrieval code is identical either way.

## License

MIT.
