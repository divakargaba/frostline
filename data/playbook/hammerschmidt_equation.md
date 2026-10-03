---
id: hammerschmidt_equation
title: "Hammerschmidt Equation: What It Estimates and Its Limits"
tags: [hydrate, hammerschmidt, inhibitor, dosing, equation]
sources:
  - https://en.wikipedia.org/wiki/Clathrate_hydrate#Removal
  - https://www.semanticscholar.org/paper/Formation-of-Gas-Hydrates-in-Natural-Gas-Lines-Hammerschmidt/d08fdd28b3859f1c82297b3ac683beb0643f68b4
  - https://doi.org/10.1021/ie50264a002
---

# Hammerschmidt Equation

## The equation

The Hammerschmidt equation (1934) estimates the temperature depression (shift in hydrate equilibrium temperature) achieved by a given concentration of thermodynamic inhibitor in the water phase:

    dT = K * w / (M * (100 - w))

Where:
- **dT** = temperature depression in degrees Celsius
- **K** = constant (1297 for most natural gas systems)
- **M** = molecular weight of the inhibitor (32 for methanol, 62 for MEG)
- **w** = weight percent of inhibitor in the aqueous phase

## Example

For methanol at 20 wt%: dT = 1297 * 20 / (32 * 80) = 10.1 C

This means a 20 wt% methanol solution shifts the hydrate curve down by about 10 C.

## Limits and caveats

1. **Accuracy:** The equation is an approximation. It is reasonably accurate for methanol up to ~25 wt% and MEG up to ~40 wt%, but overpredicts at higher concentrations.
2. **Gas composition dependence:** The constant K can vary with gas composition, though 1297 is a standard approximation for typical natural gas.
3. **Does not account for kinetics:** The equation predicts equilibrium shift only, not the time to dissociate an existing plug.
4. **High-dose warning:** Methanol concentrations above ~25 wt% may be impractical due to cost, handling, and diminishing returns. MEG above ~50 wt% can cause viscosity problems.
5. **Modern alternatives:** For precise design, thermodynamic simulation software (e.g., Multiflash, PVTsim) is preferred, but Hammerschmidt remains useful for quick field estimates and advisory systems.
