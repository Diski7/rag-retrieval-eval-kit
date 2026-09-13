"""Grounded retrieval kit: chunking, cosine retrieval, boosts, a raw similarity
gate, a named entity coverage check, and the evaluation harness that measures all
of it."""

from .coverage import known_words, uncovered_entity
from .embedder import BedrockEmbedder, Embedder, HashingEmbedder, cosine, tokens
from .loader import load_chunks, load_json
from .retriever import Chunk, Hit, Retriever

__all__ = [
    "BedrockEmbedder",
    "Chunk",
    "Embedder",
    "HashingEmbedder",
    "Hit",
    "Retriever",
    "cosine",
    "known_words",
    "load_chunks",
    "load_json",
    "tokens",
    "uncovered_entity",
]
