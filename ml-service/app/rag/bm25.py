from __future__ import annotations

import math
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been being but by can could do does did for from has have how i if in into is it its "
    "may me more most my no not of on or our out over so some such than that the their then there these they this "
    "to too up was we were what when where which who whom why will with would you your about after all also any "
    "because one other should while".split()
)


def stem(token: str) -> str:
    """Very light suffix stripping so 'lesions' matches 'lesion' and 'biopsies' matches 'biopsy'."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    if len(token) > 6 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, dict[int, int]] = {}
        for i, tokens in enumerate(docs):
            counts: dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                self.postings.setdefault(t, {})[i] = c
        self.idf = {
            t: math.log(1.0 + (self.n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def matched(self, doc: int, terms: set[str]) -> int:
        return sum(1 for t in terms if doc in self.postings.get(t, ()))

    def search(self, q_tokens: list[str], n: int = 30, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        if not self.n or not q_tokens:
            return []
        avg = self.avgdl or 1.0
        scores = np.zeros(self.n, dtype=np.float64)
        for term in set(q_tokens):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, tf in posting.items():
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / avg)
                scores[i] += idf * tf * (self.k1 + 1.0) / denom
        if mask is not None:
            scores = np.where(mask, scores, 0.0)
        order = np.argsort(-scores, kind="stable")[:n]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
