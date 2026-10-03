import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_download_contains_reproducible_report_and_attachment_headers():
    response = client.get("/api/research/export")
    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="frostline-research-results.json"'
    assert response.json()["systems"] == client.get("/api/research").json()["systems"]


def test_report_reproduces_same_policy_without_test_tuning():
    current = client.get("/api/research")
    assert current.status_code == 200
    rerun = client.post("/api/experiments/run")
    assert rerun.status_code == 200
    assert current.json()["policy_id"] == rerun.json()["policy_id"]
    assert current.json()["systems"] == rerun.json()["systems"]
    assert len(current.json()["selection"]["trials"]) == 24


def test_seed_scenarios_have_explicit_provenance_and_evidence():
    summaries = client.get("/api/scenarios").json()
    for summary in summaries:
        replay = client.get(f"/api/scenarios/{summary['id']}")
        assert replay.status_code == 200
        content = replay.json()
        assert "frames" not in summary
        assert len(content["frames"]) == summary["frame_count"]
        assert content["provenance"]
        assert all(len(f["evidence"]) == 4 and f["decision"] in ["ALERT", "WATCH", "DISMISS"] for f in content["frames"])


def test_missing_readings_are_json_null_not_nan():
    response = client.get("/api/scenarios/seed-missing")
    assert response.status_code == 200
    assert response.json()["frames"][20]["sensors"]["pressure_bar"] is None
    assert "NaN" not in response.text


def test_bad_scenario_and_stream_bounds():
    assert client.get("/api/scenarios/nope").status_code == 404
    assert client.get("/stream/nope").status_code == 404
    assert client.get("/stream/seed-test?start=999999").status_code == 422
    assert client.get("/stream/seed-test?start=-1").status_code == 422


def test_sse_emits_parseable_tick_decision_and_completion():
    response = client.get("/stream/seed-test?start=238")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    data = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]
    assert len(data) == 5
    assert data[0]["type"] == "tick"
    assert data[1]["decision"] in ["ALERT", "WATCH", "DISMISS"]
    assert data[-1] == {"complete": True}


def test_optional_voice_reports_unavailable_explicitly():
    assert client.post("/tts").status_code == 503
    assert client.get("/api/health").json()["status"] == "ok"
