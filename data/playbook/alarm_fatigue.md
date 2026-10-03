---
id: alarm_fatigue
title: "Alarm Fatigue: Why False Alarms Matter"
tags: [alarm, fatigue, safety, advisory, operations, human_factors]
sources:
  - https://en.wikipedia.org/wiki/Alarm_fatigue
  - https://doi.org/10.1016/j.psep.2018.05.029
  - https://www.aiche.org/ccps/resources/publications
---

# Alarm Fatigue: Why False Alarms Matter

## The problem

Alarm fatigue occurs when operators are exposed to so many alarms that they become desensitized — leading to delayed or missed responses to real events. In process industries, alarm overload is a recognized contributor to major incidents.

Studies of industrial control rooms show that operators can effectively respond to roughly 6-12 alarms per hour. Above this rate, important alarms get buried in noise, response times increase, and operators may silence or ignore alarms entirely.

## Relevance to hydrate detection

A hydrate early warning system that produces too many false alarms is counterproductive:

- **Operator trust erodes:** After several false alarms, operators discount or ignore subsequent alerts — including real ones.
- **Resource waste:** Each false alarm may trigger unnecessary inhibitor injection (costly), well interventions, or production curtailment.
- **The baseline trap:** The simplest "baseline" system (always say normal) has zero false alarms — but catches nothing. The challenge is catching real events while keeping false alarms low enough for operators to trust the system.

## Advisory vs control

For safety-critical systems, there is an important distinction:

- **Advisory systems** (like Frostline) recommend actions for human decision-making. The operator retains authority. This is appropriate when false positives are possible and the consequences of action are reversible (e.g., injecting inhibitor).
- **Control systems** take automatic action (e.g., shutting a valve). These require much higher confidence thresholds because false positives cause direct operational disruption.

Frostline operates in advisory mode: it recommends ALERT, WATCH, or DISMISS, with evidence and confidence scores, but never takes automatic control actions.

## Design implications

- Tune alarm thresholds to keep false alarm rates below 1-2 per 24 hours.
- Use graduated responses (WATCH before ALERT) to avoid cry-wolf effects.
- Provide evidence with every alarm so operators can quickly assess credibility.
- Track and display false alarm rates alongside detection rates to maintain calibration.
