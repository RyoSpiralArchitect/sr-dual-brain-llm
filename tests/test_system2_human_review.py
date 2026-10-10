import copy
import json
import stat
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sr-dual-brain-llm/scripts"))
from serve_system2_review import ReviewStore, make_server  # noqa: E402


@pytest.fixture
def store(tmp_path):
    packets = {"schema": "system2-blind-pairs-v1", "rubric": {}, "packets": [
        {"packet_id": "fixture-one", "question": "2 + 2?", "answer_a": "4", "answer_b": "4"},
        {"packet_id": "fixture-two", "question": "Why?", "answer_a": "Unknown", "answer_b": "Unknown"},
    ]}
    path = tmp_path / "packets.json"
    path.write_text(json.dumps(packets))
    return ReviewStore(path, tmp_path / "review")


def ratings(store):
    return {"reviewer": "Fixture reviewer", "exposure": "no_prior_results", "ratings": [
        {"packet_id": id_, "answer_a": {"correctness": 2, "completeness": 2, "unsupported_claims": 0},
         "answer_b": {"correctness": 2, "completeness": 2, "unsupported_claims": 0},
         "preferred": "tie", "rationale": ""} for id_ in store.ids]}


def test_resume_bound_to_input_and_lock_immutable(store, tmp_path):
    data = ratings(store)
    store.save(data)
    assert json.loads(store.draft.read_text())["ratings"] == data["ratings"]
    store.save(data, final=True)
    original = store.locked.read_bytes()
    assert stat.S_IMODE(store.locked.stat().st_mode) == 0o400
    with pytest.raises(FileExistsError):
        store.save(data)
    assert store.locked.read_bytes() == original
    altered = json.loads((tmp_path / "packets.json").read_text())
    altered["packets"][0]["answer_a"] = "different answer"
    (tmp_path / "different.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="different packet"):
        ReviewStore(tmp_path / "different.json", store.output)


def test_no_unscored_or_duplicate_packets_can_be_locked(store):
    data = ratings(store)
    data["ratings"].pop()
    store.save(data)
    with pytest.raises(ValueError, match="Every packet"):
        store.save(data, final=True)
    data["ratings"].append(copy.deepcopy(data["ratings"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        store.validate(data)


def test_unfilled_scores_require_explicit_uncertainty_and_reason(store):
    data = ratings(store)
    data["ratings"][0]["answer_a"]["correctness"] = None
    with pytest.raises(ValueError, match="Complete every"):
        store.validate(data, final=True)
    data["ratings"][0]["preferred"] = "uncertain"
    with pytest.raises(ValueError, match="Complete every"):
        store.validate(data, final=True)
    data["ratings"][0]["rationale"] = "I cannot verify the answer."
    store.validate(data, final=True)
    data["ratings"][0]["answer_a"]["correctness"] = True
    with pytest.raises(ValueError, match="Scores must"):
        store.validate(data)


def test_mode_metadata_rejected_in_packets(tmp_path):
    path = tmp_path / "packets.json"
    path.write_text(json.dumps({"schema": "system2-blind-pairs-v1", "rubric": {}, "packets": [
        {"packet_id": "fixture", "question": "Q", "answer_a": "A", "answer_b": "B", "answer_a_mode": "off"}]}))
    with pytest.raises(ValueError, match="mode or run metadata"):
        ReviewStore(path, tmp_path / "review")


def test_server_exposes_only_blind_data_and_requires_local_session(store):
    server = make_server(store, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(origin + "/api/session") as response:
            session = json.load(response)
        assert session["packets"] == store.packets["packets"]
        with pytest.raises(HTTPError) as missing:
            urlopen(origin + "/../reveal_key.json")
        assert missing.value.code == 404
        body = json.dumps(ratings(store)).encode()
        with pytest.raises(HTTPError) as forbidden:
            urlopen(Request(origin + "/api/draft", data=body))
        assert forbidden.value.code == 403
        headers = {"Origin": origin, "X-Review-Token": session["token"]}
        with urlopen(Request(origin + "/api/lock", data=body, headers=headers)) as response:
            assert json.load(response)["state"] == "locked"
        with pytest.raises(HTTPError) as immutable:
            urlopen(Request(origin + "/api/draft", data=body, headers=headers))
        assert immutable.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
