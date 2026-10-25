"""Shared right-worker pump with per-consultation lifetime management."""
from .callosum import CallosumClosed, RequestAbandoned
from .memory_evidence import request_memory_context


async def _respond(req, memory, model):
    context = request_memory_context(req)
    if context is not None:
        req = {**req, "context": context}
    qid, kind = req["qid"], req.get("type")
    if kind == "ASK_DETAIL":
        detail = await model.deepen(qid, req["question"], req.get("draft_sum", ""), memory,
            temperature=req.get("temperature", .7), budget=req.get("budget", "small"),
            context=req.get("context"), psychoid_projection=req.get("psychoid_attention_bias"))
        return {"notes_sum": detail["notes_sum"], "confidence_r": detail["confidence_r"]}
    if kind == "ASK_CRITIC":
        critique = await model.criticise_reasoning(qid, req.get("question", ""),
            req.get("draft", "") or req.get("draft_sum", "") or "",
            temperature=req.get("temperature", .2), context=req.get("context"),
            psychoid_projection=req.get("psychoid_attention_bias"))
        return {key: critique.get(key) for key in ("verdict", "issues", "fixes", "critic_sum",
                                                  "confidence_r", "critic_kind", "critic_status")}
    if kind == "ASK_LEAD":
        notes = await model.generate_lead(req.get("question", ""), req.get("context", ""),
                                         temperature=float(req.get("temperature", .85)))
        return {"lead_notes": notes}
    return {"error": "UnsupportedRequestType"}


async def right_worker(callosum, memory, right_model):
    while True:
        try:
            req = await callosum.recv_request()
        except CallosumClosed:
            return
        try:
            response = await callosum.run_request(req, lambda: _respond(req, memory, right_model))
        except RequestAbandoned:
            continue
        except Exception as exc:
            # Provider exceptions can include credentials or private source text.
            response = {"error": type(exc).__name__}
        await callosum.publish_response(req["request_id"], response)
