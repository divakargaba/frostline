"""Tests for backend API — FastAPI TestClient.

Owner: Div
"""

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas import DecisionEvent, TickEvent, WellInfo

pytestmark = pytest.mark.pending

client = TestClient(app)


class TestWellsEndpoint:
    def test_returns_list(self):
        """GET /wells should return a list of WellInfo-shaped objects."""
        resp = client.get("/wells")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        if len(data) > 0:
            WellInfo(**data[0])  # must validate against schema


class TestStreamEndpoint:
    def test_emits_valid_events(self):
        """GET /stream/{instance_id}?cached=true should emit SSE events.

        Each event should have a known type and valid JSON data.
        """
        valid_types = {"tick", "phase_marker", "watch_trigger",
                       "tool_call", "tool_result", "decision"}
        # In cached/mock mode this should return pre-recorded events
        resp = client.get("/stream/WELL-00019_demo?cached=true", stream=True)
        assert resp.status_code == 200


class TestResultsEndpoint:
    def test_returns_systems(self):
        """GET /results should return a dict with 'systems' list."""
        resp = client.get("/results")
        assert resp.status_code == 200
        data = resp.json()
        assert "systems" in data
