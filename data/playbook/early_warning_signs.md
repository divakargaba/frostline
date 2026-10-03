---
id: early_warning_signs
title: "Early Warning Signs of Hydrate Formation in Operating Data"
tags: [hydrate, warning, sensors, detection, pressure, temperature]
sources:
  - https://en.wikipedia.org/wiki/Clathrate_hydrate
  - https://doi.org/10.1016/j.petrol.2019.106223
  - https://github.com/petrobras/3W
---

# Early Warning Signs of Hydrate Formation in Operating Data

## Key sensor signatures

Hydrate formation produces characteristic patterns in well sensor data. The earliest signs typically appear in pressure and temperature trends before a full plug develops.

### Pressure indicators

- **Gradual pressure decline at the tree (P-PDG, P-TPT):** As hydrate crystals restrict the flowline, upstream pressure builds while downstream pressure drops. P-TPT often shows the earliest decline.
- **Increasing differential pressure:** The difference between P-PDG (permanent downhole gauge) and P-MON-CKP (monitoring choke pressure) widens as restrictions develop.
- **Pressure oscillations:** Partial blockages cause irregular flow, visible as pressure fluctuations before a smooth blockage.

### Temperature indicators

- **Temperature decline toward equilibrium:** T-TPT dropping toward the hydrate equilibrium temperature at current pressure is a direct precursor. This distinguishes hydrate from many look-alikes.
- **Coupled pressure-temperature drop:** The hallmark of hydrate formation is simultaneous pressure and temperature decline — not just one.

### Flow indicators

- **Declining flow rate (QGL):** Gas-lift flow rate drops as the flowline restriction increases.
- **Choke opening increases:** Operators may open the choke (ABER-CKP) to compensate for declining flow, masking the underlying restriction.

## Timing

In the Petrobras 3W dataset, the "forming" phase (transient, class 108/109) often lasts tens of hours — median ~63 hours for class 8 events. This gives operators a meaningful window for intervention if early warning signs are caught.

## What to watch for

An operator or automated system should trigger investigation when:
1. Both pressure and temperature show sustained declining trends (10-30 minute slopes both negative).
2. The subcooling margin is shrinking toward zero or is already positive.
3. The ML classifier probability for hydrate exceeds 50% for 3+ consecutive minutes.
