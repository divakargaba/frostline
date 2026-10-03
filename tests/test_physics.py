"""Tests for src/physics.py — hydrate thermodynamic models.

Owner: Physics+Eval
"""

import math

import numpy as np
import pytest

from src.physics import hammerschmidt_dose, hydrate_margin, hydrate_temp_towler_mokhatab

pytestmark = pytest.mark.pending


class TestTowlerMokhatab:
    def test_known_reference_value(self):
        """Verify against a textbook reference point.

        TODO: Fill in expected values from Towler & Mokhatab (2005) or
        Sloan & Koh (2007) Table X.Y. Use a pressure and SG with a
        published equilibrium temperature.
        """
        # Example: at 1000 psia, SG=0.6, expected T_hyd ~ ??? F
        # t = hydrate_temp_towler_mokhatab(1000.0, 0.6)
        # assert abs(t - EXPECTED) < 1.0  # within 1 F
        raise NotImplementedError("Fill from cited source")

    def test_monotonic_with_pressure(self):
        """Hydrate formation temperature must increase with pressure."""
        sg = 0.65
        t_low = hydrate_temp_towler_mokhatab(500.0, sg)
        t_high = hydrate_temp_towler_mokhatab(2000.0, sg)
        assert t_high > t_low


class TestHydrateMargin:
    def test_positive_margin_safe(self):
        """Well above equilibrium -> positive margin (safe)."""
        # At moderate pressure, if T is very high, margin should be positive
        m = hydrate_margin(p_bar=200.0, t_c=30.0, sg=0.65)
        assert m > 0

    def test_negative_margin_danger(self):
        """Well below equilibrium -> negative margin (danger)."""
        # At high pressure with low temperature
        m = hydrate_margin(p_bar=300.0, t_c=5.0, sg=0.65)
        assert m < 0

    def test_nan_when_temp_missing(self):
        """If temperature is NaN, margin should be NaN."""
        m = hydrate_margin(p_bar=200.0, t_c=float("nan"), sg=0.65)
        assert math.isnan(m)


class TestHammerschmidt:
    def test_methanol_dose(self):
        """Hammerschmidt for methanol: K=1297, M=32.

        TODO: Fill in expected dose from a textbook example.
        Formula: w = 100 * M * dT / (K + M * dT)
        For dT=10 C, methanol: w = 100 * 32 * 10 / (1297 + 32*10) = 19.8 wt%
        """
        dose = hammerschmidt_dose(10.0, inhibitor="methanol")
        assert abs(dose - 19.8) < 0.5

    def test_meg_dose(self):
        """Hammerschmidt for MEG: K=1297, M=62.

        For dT=10 C, MEG: w = 100 * 62 * 10 / (1297 + 62*10) = 32.3 wt%
        """
        dose = hammerschmidt_dose(10.0, inhibitor="meg")
        assert abs(dose - 32.3) < 0.5

    def test_inverse_round_trip(self):
        """Compute dose for dT, then back-calculate dT and verify round-trip.

        Inverse: dT = K * w / (M * (100 - w))
        """
        dt_input = 15.0
        dose = hammerschmidt_dose(dt_input, inhibitor="methanol")
        # Back-calculate
        K, M = 1297.0, 32.0
        dt_back = K * dose / (M * (100 - dose))
        assert abs(dt_back - dt_input) < 0.1

    def test_out_of_range_flag(self):
        """Very large dT should flag that dose > 25 wt% is out of range.

        TODO: Verify the function returns or raises an out-of-range indicator
        when the computed dose exceeds ~20-25 wt%.
        """
        dose = hammerschmidt_dose(20.0, inhibitor="methanol")
        # At dT=20 C methanol: w = 100*32*20/(1297+640) = 33.0 -> out of range
        assert dose > 25.0
