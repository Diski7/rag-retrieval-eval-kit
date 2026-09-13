"""Embedders.

The production system this is extracted from embeds with Amazon Titan Text
Embeddings v2 at 1,024 dimensions on Bedrock. That path needs AWS credentials and
costs money per call, so it is kept here as an adapter next to a dependency free
embedder that runs anywhere.

The retriever does not care which one it is handed, which is the point: retrieval
logic should be testable without a network call.
"""
import hashlib
import json
import math
import re

_WORD = re.compile(r"[a-z0-9']+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "they", "this", "to", "was", "what", "when", "where",
    "which", "who", "with", "you", "your",
}


def tokens(text):
    """Lowercase word tokens with stopwords removed and a crude plural strip."""
    out = []
    for word in _WORD.findall(text.lower()):
        if word.endswith("'s"):
            word = word[:-2]
        if word in STOPWORDS or len(word) < 2:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


def cosine(a, b):
    """Cosine similarity of two equal length vectors. Returns 0.0 on a zero vector."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class Embedder:
    """Interface. Subclasses return a list of floats."""

    dim = 0

    def embed(self, text):
        raise NotImplementedError

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


class HashingEmbedder(Embedder):
    """Deterministic bag of words hashing into a fixed number of dimensions.

    No dependencies, no network, and identical output on every run, which is what
    makes the test suite meaningful. It is not a semantic model: it matches on
    shared vocabulary, so it is a fair stand in for testing retrieval mechanics
    (gating, boosting, capping) but not for measuring semantic recall.
    """

    def __init__(self, dim=256):
        self.dim = dim

    def embed(self, text):
        vec = [0.0] * self.dim
        words = tokens(text)
        features = list(words) + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feature in features:
            index = int(hashlib.md5(feature.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            vec[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class BedrockEmbedder(Embedder):
    """Amazon Titan Text Embeddings v2 on Bedrock, 1,024 dimensions.

    Not exercised by the demo. Included because the retriever is written against
    this shape in production, and because the dimension count is a real design
    decision rather than a default: 1,024 keeps the index small enough to load
    into memory and score in one pass.
    """

    def __init__(self, model_id="amazon.titan-embed-text-v2:0", region="us-east-1", dim=1024):
        self.model_id = model_id
        self.region = region
        self.dim = dim

    def embed(self, text):
        import boto3  # imported here so the kit runs without AWS installed

        client = boto3.client("bedrock-runtime", region_name=self.region)
        body = json.dumps({"inputText": text, "dimensions": self.dim, "normalize": True})
        response = client.invoke_model(modelId=self.model_id, body=body)
        return json.loads(response["body"].read())["embedding"]
