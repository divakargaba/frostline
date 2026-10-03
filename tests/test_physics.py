"""Tests for physics module.

Owner: Physics+Eval

TODO:
  - Add tests for hydrate_temp_towler_mokhatab with known reference values
  - Add tests for hydrate_margin at safe and dangerous conditions
  - Add tests for hammerschmidt_dose for methanol and MEG
"""

import pytest

from src.physics import hammerschmidt_dose, hydrate_margin, hydrate_temp_towler_mokhatab


def test_hydrate_temp_known_value():
    """Verify Towler-Mokhatab against a known reference point."""
    raise NotImplementedError


def test_hydrate_margin_safe():
    """Verify positive margin when temperature is well above equilibrium."""
    raise NotImplementedError


def test_hydrate_margin_danger():
    """Verify negative margin when temperature is below equilibrium."""
    raise NotImplementedError


def test_hammerschmidt_methanol():
    """Verify methanol dose for a known temperature depression."""
    raise NotImplementedError


def test_hammerschmidt_meg():
    """Verify MEG dose for a known temperature depression."""
    raise NotImplementedError
