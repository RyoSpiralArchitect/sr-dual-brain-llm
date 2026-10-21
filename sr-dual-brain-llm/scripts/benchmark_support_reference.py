"""Benchmark-only reference for the whole-candidate batching rule before PR #60."""
from core.memory_answerability import AnswerabilityGate


class WholeBatchReference(AnswerabilityGate):
    """Use the production parser, rechecking every candidate on any cache miss.

    An isolated preparation sends the complete candidate batch without deleting
    earlier decisions on failure. Successful results replace the corresponding
    decisions, as in integrated main 74d552b. Benchmarks run this sequentially.
    """

    async def prepare(self, question, sources):
        if (str(self.backend.identity) != self.identity or not isinstance(question, str)
                or not question.strip() or len(question.encode()) > 6000):
            return await super().prepare(question, sources)
        eligible = [s for s in sources[:self.candidate_limit] if self.eligible(s)]
        keys = [self._key(question, source) for source in eligible]
        if all(key in self.cache for key in keys):
            return await super().prepare(question, sources)
        batch = AnswerabilityGate(self.backend, candidate_limit=self.candidate_limit,
                                  max_items=self.max_items,
                                  max_question_chars=self.max_question_chars,
                                  max_answer_chars=self.max_answer_chars)
        batch.identity = self.identity
        result = await batch.prepare(question, sources)
        if result["status"] == "ready":
            for key, decision in batch.cache.items():
                self.cache[key] = decision
                self.cache.move_to_end(key)
            while len(self.cache) > self.max_items:
                self.cache.popitem(last=False)
        return result
