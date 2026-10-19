"""Optional, session-local semantic questions; no network work in sync retrieval."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
import math
import os
from typing import Sequence

from .llm_client import LLMClient, LLMConfig, load_llm_config


def normalized_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or not all(math.isfinite(value) for value in vector) or not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must be a finite, nonzero vector")
    return tuple(value / norm for value in vector)


class OpenAIQuestionEmbeddings:
    """Explicitly configured embeddings, separate from chat and archetype models."""

    def __init__(self, config: LLMConfig, model: str):
        if config.provider != "openai" or not model.strip():
            raise ValueError("Question embeddings require OpenAI configuration and an explicit model")
        self.config = replace(config, model=model, timeout_seconds=max(1.0, min(20.0, float(config.timeout_seconds))))
        self.client = LLMClient(self.config)
        base = (config.api_base or self.client._default_base()).rstrip("/")
        self.url = base + "/embeddings"
        self.calls = 0
        self.input_tokens: int | None = 0
        self.served_model: str | None = None

    @property
    def identity(self) -> str:
        return hashlib.sha256((self.url + "\n" + self.config.model).encode()).hexdigest()

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        headers = {"Authorization": "Bearer " + self.config.api_key, "Content-Type": "application/json"}
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        headers.update(self.config.extra_headers)
        self.calls += 1
        try:
            response = await self.client._post_json(
                self.url, {"model": self.config.model, "input": list(texts)}, headers,
                timeout_seconds=self.config.timeout_seconds,
            )
        except Exception:
            self.input_tokens = None
            raise
        tokens = (response.get("usage") or {}).get("total_tokens")
        self.input_tokens = (self.input_tokens + tokens
                             if self.input_tokens is not None and type(tokens) is int and tokens >= 0 else None)
        served_model = response.get("model")
        if (not isinstance(served_model, str) or not served_model
                or (self.served_model is not None and served_model != self.served_model)):
            raise ValueError("Embedding response model missing or changed")
        rows = response.get("data", [])
        if len(rows) != len(texts) or sorted(item.get("index", -1) for item in rows) != list(range(len(texts))):
            raise ValueError("Embedding response coverage differs from the request")
        vectors = [normalized_vector(item["embedding"]) for item in sorted(rows, key=lambda item: item["index"])]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError("Embedding dimensions differ")
        self.served_model = served_model
        return vectors


class QuestionSemanticIndex:
    """Bounded question cache prepared asynchronously before controller perception.

    Only recent candidate questions are sent; answers/drafts are never embedded.
    Missing embeddings return None, allowing explicit lexical fallback. Caches
    belong to one session, and do not contain clients with conversational state.
    """

    def __init__(self, backend, *, min_similarity: float = 0.70, max_items: int = 512,
                 relative_cutoff: float = 0.8, scope_min_similarity: float = 0.50):
        if (not 0 <= min_similarity <= 1 or max_items < 2 or not 0 <= relative_cutoff <= 1
                or not 0 <= scope_min_similarity <= 1):
            raise ValueError("Invalid semantic index limits")
        self.backend = backend
        self.min_similarity = min_similarity
        self.max_items = max_items
        self.relative_cutoff = relative_cutoff
        self.scope_min_similarity = scope_min_similarity
        self.scope_resolver = None
        self.identity = str(backend.identity)
        self.vectors: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self.dimension: int | None = None

    async def prepare(self, questions: Sequence[str], *, batch_size: int = 64) -> dict:
        if batch_size < 1:
            raise ValueError("Positive embedding batch size required")
        if str(self.backend.identity) != self.identity:
            raise ValueError("Embedding identity changed; create a new index")
        # Do not truncate a question and then attribute its vector to the full text.
        eligible = list(dict.fromkeys(q for q in questions if isinstance(q, str) and q.strip() and len(q.encode()) <= 6000))
        missing = []
        batch_bytes = 0
        for question in eligible:
            if question in self.vectors:
                continue
            if len(missing) == min(batch_size, self.max_items) or batch_bytes + len(question.encode()) > 48000:
                break
            missing.append(question)
            batch_bytes += len(question.encode())
        if missing:
            vectors = [normalized_vector(v) for v in await self.backend.embed(missing)]
            if str(self.backend.identity) != self.identity:
                raise ValueError("Embedding identity changed during preparation")
            if len(vectors) != len(missing) or len({len(v) for v in vectors}) != 1:
                raise ValueError("Invalid embedding batch")
            dimension = len(vectors[0])
            if self.dimension is not None and self.dimension != dimension:
                raise ValueError("Embedding dimension changed; create a new index")
            # Commit only after validating the entire batch.
            self.dimension = dimension
            for question, vector in zip(missing, vectors):
                self.vectors[question] = vector
        for question in reversed(eligible):
            if question in self.vectors:
                self.vectors.move_to_end(question)
        while len(self.vectors) > self.max_items:
            self.vectors.popitem(last=False)
        return {"embedded_questions": len(missing), "cached_questions": len(self.vectors),
                "requested_questions": len(eligible), "embedding_id": self.identity,
                "unprepared_questions": sum(q not in self.vectors for q in eligible)}

    def embedding_questions(self, questions: Sequence[str]) -> list[str]:
        """Bound the original input before deriving content-only question views."""
        return list(dict.fromkeys(
            self.scope_resolver.query(q).content if self.scope_resolver is not None else q
            for q in questions if isinstance(q, str) and q.strip() and len(q.encode()) <= 6000
        ))

    def threshold_for(self, left: str, right: str, *, source_scope: str | None = None) -> float:
        if self.scope_resolver is not None:
            query, source = self.scope_resolver.query(left), self.scope_resolver.source(right, source_scope)
            if query.scopes and source.scopes:
                return self.scope_min_similarity
        return self.min_similarity

    def similarity(self, left: str, right: str, *, source_scope: str | None = None) -> float | None:
        if (str(self.backend.identity) != self.identity
                or len(left.encode()) > 6000 or len(right.encode()) > 6000):
            return None
        if self.scope_resolver is not None:
            query, source = self.scope_resolver.query(left), self.scope_resolver.source(right, source_scope)
            if not self.scope_resolver.compatible(query, source):
                return -1.0
            left, right = query.content, source.content
        a, b = self.vectors.get(left), self.vectors.get(right)
        if a is None or b is None or len(a) != len(b):
            return None
        return max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))

    def snapshot(self) -> dict:
        return {"embedding_id": self.identity, "dimension": self.dimension,
                "min_similarity": self.min_similarity,
                "relative_cutoff": self.relative_cutoff,
                "scope_min_similarity": self.scope_min_similarity,
                "scope_registry": self.scope_resolver.snapshot() if self.scope_resolver is not None else None,
                "cache_sha256": hashlib.sha256(json.dumps(
                    dict(self.vectors), sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()}


def semantic_index_from_environment(fallback_config: LLMConfig | None = None) -> QuestionSemanticIndex | None:
    model = str(os.environ.get("DUALBRAIN_MEMORY_EMBEDDING_MODEL") or "").strip()
    if not model:
        return None
    config = load_llm_config("MEMORY", default_model=model) or fallback_config
    if config is None:
        raise ValueError("Configured memory embedding model requires provider credentials")
    threshold = float(os.environ.get("DUALBRAIN_MEMORY_MIN_SIMILARITY", "0.70"))
    scope_threshold = float(os.environ.get("DUALBRAIN_MEMORY_SCOPE_MIN_SIMILARITY", "0.50"))
    return QuestionSemanticIndex(OpenAIQuestionEmbeddings(config, model), min_similarity=threshold,
                                 scope_min_similarity=scope_threshold)
