# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, Sequence


_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u4e00-\u9fff]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "not",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}

_INTERNAL_TOKENS: tuple[str, ...] = (
    "qid",
    "telemetry",
    "metrics",
    "trace",
    "architecture path",
    "hemisphere",
    "psychoid",
    "corpus callosum",
    "left brain",
    "right brain",
)

_UNASKED_SENSING_MARKERS: tuple[str, ...] = (
    "weather",
    "天気",
)


def _tokenise(text: str) -> list[str]:
    lowered = (text or "").replace("\n", " ").lower()
    tokens: list[str] = []
    for tok in _WORD_RE.findall(lowered):
        tok = tok.strip()
        if not tok:
            continue
        tokens.append(tok)

        # For CJK, add short character n-grams so that overlap can be detected even
        # when there are no spaces (useful for Japanese).
        if _CJK_RE.search(tok) and len(tok) >= 4:
            grams_added = 0
            for i in range(len(tok) - 1):
                tokens.append(tok[i : i + 2])
                grams_added += 1
                if grams_added >= 32:
                    break
    return tokens


def _keywords(tokens: Sequence[str], *, limit: int = 8) -> tuple[str, ...]:
    seen: list[str] = []
    for tok in tokens:
        if len(tok) < 2:
            continue
        if tok.isascii() and tok.isalpha() and tok in _EN_STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= limit:
            break
    return tuple(seen)


def _split_sentences(text: str) -> list[str]:
    chunks: list[str] = []
    buf: list[str] = []
    for ch in str(text or ""):
        buf.append(ch)
        if ch in ".!?。！？":
            chunks.append("".join(buf))
            buf = []
    if buf:
        chunks.append("".join(buf))
    return [chunk for chunk in chunks if chunk.strip()]


def _strip_sentences_with_markers(text: str, *, markers: Sequence[str]) -> tuple[str, bool]:
    if not text:
        return "", False
    markers_norm = [m.lower() for m in markers if m]
    kept: list[str] = []
    removed_any = False
    for sent in _split_sentences(text):
        low = sent.lower()
        if any(m in low for m in markers_norm):
            removed_any = True
            continue
        kept.append(sent)
    cleaned = "".join(kept).strip()
    return cleaned, removed_any


def _strip_lines_with_internal_tokens(text: str, *, question: str) -> tuple[str, bool]:
    if not text:
        return "", False
    q_low = str(question or "").lower()
    removed = False
    kept: list[str] = []
    for line in str(text).splitlines():
        low = line.lower()
        hits = [tok for tok in _INTERNAL_TOKENS if tok in low]
        if hits and any(tok not in q_low for tok in hits):
            removed = True
            continue
        kept.append(line)
    return "\n".join(kept).strip(), removed


def _extract_last_assistant_answer(working_memory_context: str) -> str:
    last = ""
    for line in str(working_memory_context or "").splitlines():
        if line.startswith("A:"):
            last = line[2:].strip()
    return last


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b).ratio()

class Auditor:
    def __init__(self, banned_tokens=None):
        self.banned = banned_tokens or ["<SECRET_KEY>", "PASSWORD"]

    def check(
        self,
        answer: str,
        *,
        question: str | None = None,
        focus_keywords: Sequence[str] | None = None,
        working_memory_context: str = "",
        is_trivial_chat: bool = False,
        allow_debug: bool = False,
    ) -> Dict[str, Any]:
        question_text = str(question or "")
        for b in self.banned:
            if b in answer:
                return {"ok": False, "reason": "PII or banned token"}
        if len(answer) > 5000:
            return {"ok": False, "reason": "too_long"}

        if allow_debug:
            return {"ok": True}

        revised = str(answer or "").strip()
        flags: list[str] = []
        action = "accept"
        clarifying_question: str | None = None
        clarifying_replace = False

        revised, removed_internal = _strip_lines_with_internal_tokens(revised, question=question_text)
        if removed_internal:
            flags.append("internal_leak")
            action = "clean"

        if revised:
            q_low = question_text.lower()
            if not any(marker.lower() in q_low for marker in _UNASKED_SENSING_MARKERS):
                cleaned, removed = _strip_sentences_with_markers(
                    revised, markers=_UNASKED_SENSING_MARKERS
                )
                if removed:
                    revised = cleaned
                    flags.append("unsupported_sensing")
                    action = "clean"

        prev_answer = _extract_last_assistant_answer(working_memory_context)
        repetition = _similarity(prev_answer, revised)
        if repetition >= 0.935 and len(revised) >= 80 and prev_answer:
            flags.append("repetition")
            action = "shorten"
            if any(ch in question_text for ch in "？?"):
                # Keep it short and prompt the user to steer.
                revised = "うん、了解。続ける？それとも別の話にする？" if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", question_text) else "Got it. Want to keep going, or switch topics?"
            else:
                revised = "了解。続ける？それとも別の話にする？" if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", question_text) else "Understood. Keep going, or switch topics?"

        if is_trivial_chat and len(revised) > 360:
            flags.append("verbosity")
            action = "trim"
            revised = revised[:357].rstrip() + "..."

        coverage = None
        keyword_set: set[str] = set()
        if focus_keywords:
            focus_tokens: list[str] = []
            for kw in focus_keywords:
                if not kw:
                    continue
                focus_tokens.extend(_tokenise(str(kw)))
                if len(focus_tokens) >= 96:
                    break
            keyword_set.update(_keywords(focus_tokens, limit=12))
        else:
            keyword_set.update(_keywords(_tokenise(question_text)))
        answer_tokens = set(_tokenise(revised))
        if keyword_set:
            coverage = len(answer_tokens & keyword_set) / max(len(keyword_set), 1)

        question_substantial = bool(question_text.strip()) and (
            len(question_text) >= 12 or len(_tokenise(question_text)) >= 4
        )
        has_qmark = ("?" in question_text) or ("？" in question_text)
        declarative_long_form = bool(
            not has_qmark
            and (
                ("\n" in question_text and len(question_text) >= 48)
                or len(question_text) >= 80
                or len(_tokenise(question_text)) >= 14
            )
        )
        min_answer_len = 24 if has_qmark else 24
        answer_has_qmark = ("?" in revised) or ("？" in revised)
        should_check_drift = bool(
            question_substantial
            and len(revised) >= min_answer_len
            and not is_trivial_chat
            and not declarative_long_form
            and not (answer_has_qmark and len(revised) <= 240)
        )
        if should_check_drift and coverage is not None and coverage < 0.06:
            flags.append("topic_drift")
            action = "clarify"
            seeds: list[str] = []
            if focus_keywords:
                for kw in focus_keywords:
                    if not kw:
                        continue
                    kw_text = str(kw).strip()
                    if not kw_text or kw_text in seeds:
                        continue
                    seeds.append(kw_text)
                    if len(seeds) >= 2:
                        break
            if not seeds:
                extracted = _keywords(_tokenise(question_text), limit=2)
                seeds = list(extracted)
            seed = " ".join(seeds).strip()
            jp = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", question_text))
            if seed:
                clarifying_question = (
                    f"確認：{seed}のどの部分が知りたい？（概要／具体例／手順など）"
                    if jp
                    else f"Quick check: what part of '{seed}' do you want—overview, examples, or steps?"
                )
            else:
                clarifying_question = (
                    "ごめん、意図を取り違えそう。もう少しだけ具体的に教えて？"
                    if jp
                    else "Sorry—I might be misreading you. Could you clarify what you mean?"
                )
            clarifying_replace = coverage < 0.03

        if not revised:
            jp = bool(re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", question_text))
            revised = (
                "ごめん、うまく受け取れなかった。もう一言だけ補足してくれる？"
                if jp
                else "Sorry—I didn't catch that. Could you add one more detail?"
            )
            flags.append("empty_after_cleaning")
            action = "clarify"
            if clarifying_question is None:
                clarifying_question = revised

        metacognition: Dict[str, Any] = {
            "action": action,
            "flags": flags,
            "coverage": coverage,
            "repetition": round(repetition, 4),
            "question_len": len(question_text),
            "answer_len": len(str(answer or "")),
            "revised_len": len(revised),
            "clarifying_question": (clarifying_question[:240] if clarifying_question else None),
            "clarifying_replace": bool(clarifying_replace),
        }

        out: Dict[str, Any] = {"ok": True, "metacognition": metacognition}
        if revised != str(answer or "").strip():
            out["revised_answer"] = revised
        return out
