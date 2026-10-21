"""Request-local provenance for answers eligible to become primary memory."""
from contextvars import ContextVar
from functools import wraps


_completed: ContextVar[set[str] | None] = ContextVar("source_completions", default=None)


def source_completion_required():
    return _completed.get() is not None


def record_completed_answer(text):
    completed = _completed.get()
    if completed is not None and isinstance(text, str) and text.strip():
        completed.add(text.strip())


def require_completed_answer(text):
    completed = _completed.get()
    if completed is None or not text.strip() or text.strip() not in completed:
        raise RuntimeError("Primary source answer lacks confirmed final completion")


def guard_source_completions(process):
    @wraps(process)
    async def guarded(self, *args, **kwargs):
        metadata = kwargs.get("memory_context")
        primary = (getattr(self.memory, "conversation_context", False)
                   and isinstance(metadata, dict) and metadata.get("remember") is True)
        if not primary:
            return await process(self, *args, **kwargs)
        token = _completed.set(set())
        try:
            return await process(self, *args, **kwargs)
        finally:
            _completed.reset(token)
    return guarded
