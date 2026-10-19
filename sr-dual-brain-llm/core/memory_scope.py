"""Declared memory identities and question content; no inferred translations."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import re
import unicodedata


def normalize_name(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _continues_name(character: str, alias_edge: str) -> bool:
    if not character:
        return False
    category = unicodedata.category(character)
    if category[0] == "M" or category == "Cf":
        return True
    # Japanese particles/wrappers can adjoin a Latin name (e.g. Caféの記録).
    # Other Unicode letters, numbers, marks and underscores extend the name.
    japanese = bool(re.fullmatch(r"[\u3040-\u30ff\u4e00-\u9fff]", character))
    word = character == "_" or category[0] in "LN"
    katakana_extension = (bool(re.fullmatch(r"[\u30a0-\u30ff]", alias_edge))
                          and bool(re.fullmatch(r"[\u30a0-\u30ff]", character)))
    return (word and not japanese) or katakana_extension


@dataclass(frozen=True)
class ScopeCue:
    scopes: frozenset[str]
    ambiguous: bool
    content: str


class MemoryScopeResolver:
    """Aliases are trusted caller metadata, never a verification of QA contents."""

    def __init__(self, traces=()):
        traces = tuple(traces)
        aliases: dict[str, set[str]] = {}
        sources: dict[str, set[str]] = {}
        seed_counts = Counter(trace.seed_id for trace in traces if trace.seed_id)
        qid_counts = Counter(trace.qid for trace in traces if trace.qid)
        self.seed_sources = {
            (trace.seed_id, trace.qid): ((trace.question, trace.retrieval_scope)
                                       if seed_counts[trace.seed_id] == qid_counts[trace.qid] == 1 else None)
            for trace in traces if trace.seed_id and trace.qid
        }
        for trace in traces:
            if not trace.retrieval_scope:
                continue
            sources.setdefault(trace.question, set()).add(trace.retrieval_scope)
            for alias in trace.scope_aliases:
                aliases.setdefault(normalize_name(alias), set()).add(trace.retrieval_scope)
        self.aliases = {alias: frozenset(scopes) for alias, scopes in aliases.items()}
        self.sources = {question: frozenset(scopes) for question, scopes in sources.items()}
        self.trie = {}
        for alias in aliases:
            node = self.trie
            for character in alias:
                node = node.setdefault(character, {})
            node[""] = alias

    def _matches(self, text: str) -> list[tuple[int, int, str]]:
        """Leftmost, longest *valid* alias; a rejected prefix consumes no text."""
        matches = []
        start = 0
        while start < len(text):
            node, end, best = self.trie, start, None
            while end < len(text) and text[end] in node:
                node = node[text[end]]
                end += 1
                alias = node.get("")
                if (alias is not None
                        and not _continues_name(text[start - 1] if start else "", alias[0])
                        and not _continues_name(text[end] if end < len(text) else "", alias[-1])):
                    best = (start, end, alias)
            if best is None:
                start += 1
            else:
                matches.append(best)
                start = best[1]
        return matches

    def query(self, question: str) -> ScopeCue:
        if not self.trie:
            return ScopeCue(frozenset(), False, question)
        text = normalize_name(question)
        matches = self._matches(text)
        if not matches:
            return ScopeCue(frozenset(), False, question)
        groups = [self.aliases[alias] for _, _, alias in matches]
        scopes = frozenset().union(*groups)
        parts, end = [], 0
        for start, stop, _ in matches:
            parts.extend((text[end:start], " "))
            end = stop
        content = "".join([*parts, text[end:]])
        content = re.sub(r"\bproject\b|プロジェクト|案件|[’']s\b", " ", content)
        content = " ".join(content.split()).strip(" ?？:：,、。")
        return ScopeCue(scopes, any(len(group) != 1 for group in groups), content)

    def source(self, question: str, scope: str | None = None) -> ScopeCue:
        cue = self.query(question)
        scopes = frozenset({scope}) if scope else self.sources.get(question, cue.scopes)
        return ScopeCue(scopes, len(scopes) > 1 and scope is None, cue.content)

    def source_for_seed(self, question: str, seed_id: str, source_qid: str | None) -> ScopeCue:
        key = (seed_id, source_qid)
        if key not in self.seed_sources:
            # Legacy seeds can lack a persisted identity binding. A question-text
            # fallback must still reject multiple possible scopes.
            return self.source(question)
        identity = self.seed_sources[key]
        cue = self.query(question)
        if identity is None or identity[0] != question:
            return ScopeCue(frozenset(), True, cue.content)
        scope = identity[1]
        return ScopeCue(frozenset({scope}) if scope else frozenset(), False, cue.content)

    @staticmethod
    def compatible(query: ScopeCue, source: ScopeCue) -> bool:
        return (not query.ambiguous and not source.ambiguous
                and (bool(query.scopes & source.scopes) if query.scopes or source.scopes else True))

    def allows_trace(self, query: ScopeCue, trace) -> bool:
        # A scope-specific query must not silently borrow a global/other-scope fact.
        source = ScopeCue(frozenset({trace.retrieval_scope}) if trace.retrieval_scope else frozenset(),
                          False, "")
        return self.compatible(query, source)

    def snapshot(self) -> dict:
        data = {"aliases": {key: sorted(value) for key, value in self.aliases.items()},
                "sources": {key: sorted(value) for key, value in self.sources.items()},
                "seed_sources": [[*key, value] for key, value in sorted(self.seed_sources.items())]}
        return {"scopes": len(set().union(*self.aliases.values())) if self.aliases else 0,
                "registry_sha256": hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
