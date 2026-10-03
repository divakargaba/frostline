"""Tests for src/tools.py — agent tool definitions.

Owner: Div
"""

import json

import pytest

pytestmark = pytest.mark.pending


class TestJSONSerializable:
    """Every tool must return JSON-serializable output."""

    def test_hydrate_margin_serializable(self):
        from src.tools import hydrate_margin

        result = hydrate_margin(p_bar=200.0, t_c=25.0)
        json.dumps(result)  # must not raise

    def test_classify_event_serializable(self):
        from src.tools import classify_event

        result = classify_event(well_id="WELL-00019", minutes=60)
        json.dumps(result)

    def test_forecast_onset_serializable(self):
        from src.tools import forecast_onset

        result = forecast_onset(well_id="WELL-00019", minutes=60)
        json.dumps(result)

    def test_methanol_dose_serializable(self):
        from src.tools import methanol_dose

        result = methanol_dose(delta_t_c=10.0)
        json.dumps(result)

    def test_well_history_serializable(self):
        from src.tools import well_history

        result = well_history(well_id="WELL-00019")
        json.dumps(result)

    def test_search_playbook_serializable(self):
        from src.tools import search_playbook

        result = search_playbook(query="hydrate response")
        json.dumps(result)


class TestGetWindowContract:
    def test_returns_dict_not_dataframe(self):
        """get_window should return JSON-friendly data, not raw DataFrame rows."""
        from src.tools import get_window

        result = get_window(well_id="WELL-00019", minutes=60)
        # Should be serializable (dict or list, not DataFrame)
        json.dumps(result)


class TestSchemaRegistry:
    def test_all_tools_have_schemas(self):
        """TOOL_SCHEMAS should have an entry for each tool function."""
        from src.tools import TOOL_SCHEMAS

        tool_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        expected = {
            "get_window", "hydrate_margin", "classify_event",
            "forecast_onset", "methanol_dose", "well_history", "search_playbook",
        }
        assert expected == tool_names
