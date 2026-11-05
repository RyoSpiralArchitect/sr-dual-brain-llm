"""Local whole-record quotation intent; never a factual support approval."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import unicodedata


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def literal_record_replay_requested(reference_content: str) -> bool:
    """Positive whole-record requests after declared aliases are removed."""
    text = unicodedata.normalize('NFKC', reference_content).casefold().strip()
    text = re.sub(r'\s+', ' ', text).strip(' .。!?？')
    text = re.sub(r'\b(?:for|of)\s*(?=verbatim\b|exactly\b|$)', '', text).strip()
    if re.fullmatch(r'(?:please )?(?:repeat|reproduce|quote) (?:(?:the|that) )?(?:recorded|stored) '
                    r'(?:answer|text|record) (?:verbatim|exactly)', text):
        return True
    text = re.sub(r'^の\s*', '', text)
    text = re.sub(r'^その(?=記録|保存済み)', '', text)
    return text in {
        '記録をそのまま再掲して', '記録をそのまま再掲してください',
        '記録された回答をそのまま再掲して', '記録された回答をそのまま再掲してください',
        '保存済みの回答をそのまま再掲して', '記録文をそのまま引用して',
        '記録された回答を一字一句繰り返して',
    }


@dataclass(frozen=True)
class RecordReplayIntent:
    """Trusted caller binding, recomputed from the current query/complete source."""
    query_sha256: str
    source_sha256: str
    source_qid: str
    scope: str

    def __post_init__(self):
        if (any(type(value) is not str or re.fullmatch(r'[0-9a-f]{64}', value) is None
                for value in (self.query_sha256, self.source_sha256))
                or type(self.source_qid) is not str or not 0 < len(self.source_qid) <= 256
                or type(self.scope) is not str or not self.scope.strip() or len(self.scope) > 128):
            raise ValueError('Bounded exact replay binding required')

    @classmethod
    def bind(cls, question, source):
        return cls(_digest(question), _digest(source), source['qid'], source['retrieval_scope'])

    def matches(self, question, source):
        return (bool(self.source_qid) and bool(self.scope)
                and self == self.bind(question, source))

    def identity(self):
        return asdict(self)
