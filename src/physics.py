"""Hydrate thermodynamic models.

Owner: Physics+Eval

Implements hydrate equilibrium temperature estimation (Towler & Mokhatab
correlation), subcooling margin calculation, and Hammerschmidt inhibitor
dosing for methanol and MEG.

TODO:
  - Implement Towler-Mokhatab correlation
  - Implement hydrate margin (subcooling) calculation
  - Implement Hammerschmidt inhibitor dosing
  - Add unit tests for known reference values
"""


def hydrate_temp_towler_mokhatab(p_psia: float, sg: float) -> float:
    """Estimate hydrate formation temperature using Towler-Mokhatab correlation.

    Args:
        p_psia: Pressure in psia.
        sg: Gas specific gravity (air = 1.0).

    Returns:
        Hydrate formation temperature in degrees Fahrenheit.
    """
    raise NotImplementedError


def hydrate_margin(p_bar: float, t_c: float, sg: float = 0.65) -> float:
    """Calculate subcooling margin (distance from hydrate equilibrium).

    A positive margin means the system is above the hydrate formation
    temperature (safe). A negative margin means hydrates may form.

    Args:
        p_bar: Pressure in bar.
        t_c: Current temperature in degrees Celsius.
        sg: Gas specific gravity (default 0.65).

    Returns:
        Subcooling margin in degrees Celsius.
    """
    raise NotImplementedError


def hammerschmidt_dose(delta_t_c: float, inhibitor: str = "methanol") -> float:
    """Calculate required inhibitor mass fraction using Hammerschmidt equation.

    Args:
        delta_t_c: Required temperature depression in degrees Celsius.
        inhibitor: Inhibitor type — 'methanol' or 'meg'.

    Returns:
        Required inhibitor mass fraction (weight percent, 0–100).
    """
    raise NotImplementedError
