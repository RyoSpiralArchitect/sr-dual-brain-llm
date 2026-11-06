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


def _normalized_literal_request(reference_content: str) -> str:
    """Normalize recognition only; never change a query/source approval key."""
    text = unicodedata.normalize('NFKC', reference_content).casefold().strip()
    return re.sub(r'\s+', ' ', text).strip(' .。!?？')


def literal_record_replay_requested(reference_content: str) -> bool:
    """Positive whole-record requests after declared aliases are removed."""
    text = _normalized_literal_request(reference_content)
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


def literal_record_replay_followup_requested(question: str) -> bool:
    """The existing bounded English `that` form, with identical normalization."""
    return (literal_record_replay_requested(question)
            and re.match(r'^(?:please )?(?:repeat|reproduce|quote) that\b',
                         _normalized_literal_request(question)) is not None)


def scoped_record_replay_requested(question, resolver, *, allow_ambiguous=False):
    """One identity slot; optional ambiguity preserves a request, never a binding."""
    parts = resolver.single_alias_parts(question, allow_ambiguous=allow_ambiguous)
    if parts is None:
        return False
    before, after = parts
    prefix = before.strip()
    if prefix in {'', 'project', 'プロジェクト'}:
        # Prefix identity, Japanese possessive, or synthesized focus header.
        if not after.strip() or after.lstrip()[0] in '.。!?？;；':
            return False
        return literal_record_replay_requested(re.sub(r'^\s*[:：]\s*', '', after))
    marked_project = re.search(r'\bproject\s*$', before) is not None
    before = re.sub(r'\bproject\s*$', '', before).strip()
    if re.fullmatch(r'(?:please )?(?:repeat|reproduce|quote)', before):
        if not marked_project and re.match(r"^[’']s\s+", after.strip()) is None:
            return False
        return literal_record_replay_requested(before + ' ' + re.sub(r"^[’']s\s+", '', after.strip()))
    if re.search(r'\b(?:for|of)$', before):
        return literal_record_replay_requested(before + ' ' + after)
    return False


def record_replay_requested(question, resolver):
    """Recognize the existing whole-record grammar independently of resolution."""
    if not isinstance(question, str) or len(question) > 6000 or '\x00' in question:
        return False
    return (literal_record_replay_requested(question)
            or scoped_record_replay_requested(question, resolver, allow_ambiguous=True))


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


@dataclass(frozen=True)
class RecordReplayResolution:
    """A fresh local request/binding result, never a source support approval.

    ``bound`` means one current eligible exact source can be named. Delivery
    still requires that binding's separate complete replay-policy approval.
    """
    binding_status: str
    intent: RecordReplayIntent | None = None

    def __post_init__(self):
        if (type(self.binding_status) is not str or self.binding_status not in {
                'not_requested', 'unresolved_reference', 'missing_record',
                'ambiguous_record', 'unavailable_binding', 'invalid_selection', 'stale_selection', 'bound'}
                or (self.binding_status == 'bound' and type(self.intent) is not RecordReplayIntent)
                or (self.binding_status != 'bound' and self.intent is not None)):
            raise ValueError('Replay resolution requires a consistent exact binding state')

    @property
    def requested(self):
        return self.binding_status != 'not_requested'
