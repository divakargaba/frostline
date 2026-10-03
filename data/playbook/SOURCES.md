# Playbook Sources

All playbook documents are grounded in publicly available sources.
Every URL below was verified on 2026-10-03.

## Web references

| Source | URL | Status | Used in |
|--------|-----|--------|---------|
| Wikipedia: Clathrate hydrate | https://en.wikipedia.org/wiki/Clathrate_hydrate | VERIFIED | what_are_hydrates, equilibrium_subcooling, early_warning_signs, cost_of_hydrate_events, hydrate_response, low_dosage_inhibitors, prevention_methods, scaling_vs_hydrate, shutdown_restart_risk, choke_flow_instability |
| Wikipedia: Clathrate hydrate #Prevention | https://en.wikipedia.org/wiki/Clathrate_hydrate#Prevention | VERIFIED | thermodynamic_inhibitors, low_dosage_inhibitors, prevention_methods |
| Wikipedia: Clathrate hydrate #Removal | https://en.wikipedia.org/wiki/Clathrate_hydrate#Removal | VERIFIED | hammerschmidt_equation, hydrate_response, plug_remediation, thermodynamic_inhibitors |
| Wikipedia: Alarm fatigue | https://en.wikipedia.org/wiki/Alarm_fatigue | VERIFIED | alarm_fatigue |
| Petrobras 3W dataset (GitHub) | https://github.com/petrobras/3W | VERIFIED | early_warning_signs, scaling_vs_hydrate, choke_flow_instability |
| ScienceDirect: Gas Hydrate topic | https://www.sciencedirect.com/topics/engineering/gas-hydrate | VERIFIED | what_are_hydrates |
| IEEE YP Hackathon Case 9 README | context/hackathon/case9_README.md | VERIFIED (local) | cost_of_hydrate_events |

## Books and journal articles (via DOI)

| Reference | DOI / URL | Status | Used in |
|-----------|-----------|--------|---------|
| Sloan & Koh, *Clathrate Hydrates of Natural Gases*, 3rd ed., CRC Press, 2007, ISBN 978-0849390784 | https://doi.org/10.1201/9781420008494 | VERIFIED | equilibrium_subcooling, thermodynamic_inhibitors, plug_remediation, prevention_methods, shutdown_restart_risk, hydrate_response |
| Vargas et al., "A realistic and public dataset with rare undesirable real events in oil wells", *J. Petrol. Sci. Eng.*, 2019 | https://doi.org/10.1016/j.petrol.2019.106223 | VERIFIED | early_warning_signs, scaling_vs_hydrate, choke_flow_instability, cost_of_hydrate_events |
| Hammerschmidt, "Formation of Gas Hydrates in Natural Gas Transmission Lines", *Ind. Eng. Chem.*, vol. 26, p. 851, 1934 | https://doi.org/10.1021/ie50264a002 | VERIFIED (DOI resolves; ACS paywall) | hammerschmidt_equation |
| Hammerschmidt 1934 (Semantic Scholar page) | https://www.semanticscholar.org/paper/Formation-of-Gas-Hydrates-in-Natural-Gas-Lines-Hammerschmidt/d08fdd28b3859f1c82297b3ac683beb0643f68b4 | VERIFIED | hammerschmidt_equation |
| Amodu, "Assessment of hydrates inhibition in deepwater production systems using LDHI and MEG", *J. Petrol. Explor. Prod. Technol.*, 2019 | https://link.springer.com/article/10.1007/s13202-019-00812-4 | VERIFIED | low_dosage_inhibitors |

## Industry standards

| Source | URL | Status | Used in |
|--------|-----|--------|---------|
| ISA-18 Alarm Management Standards | https://www.isa.org/standards-and-publications/isa-standards/isa-18-series-of-standards | VERIFIED | alarm_fatigue |
| AIChE/CCPS Publications | https://www.aiche.org/ccps/publications | VERIFIED (403 to bots; confirmed via search) | alarm_fatigue |

## Removed citations

The following citations from the original docs were incorrect DOIs or dead URLs and have been replaced:

| Original URL | Issue | Replacement |
|--------------|-------|-------------|
| `https://petrowiki.spe.org/Gas_hydrates` | PetroWiki migrated to OnePetro Jan 2025; all old URLs redirect to generic landing page | Wikipedia Clathrate hydrate |
| `https://petrowiki.spe.org/Hydrate_plug_remediation` | Same migration; dead redirect | Wikipedia Clathrate hydrate #Removal |
| `https://en.wikipedia.org/wiki/Hammerschmidt_equation` | 404 — page never existed | Semantic Scholar + ACS DOI for original 1934 paper |
| `https://doi.org/10.1016/B978-0-12-382182-9.00001-3` | Resolves to "Overview of Sustainability of Water Quality" — wrong paper | Correct Sloan & Koh DOI: 10.1201/9781420008494 |
| `https://doi.org/10.1016/j.petrol.2018.09.057` | Resolves to "CO2 circulating in geothermal well" — wrong paper | Correct 3W paper DOI: 10.1016/j.petrol.2019.106223 |
| `https://doi.org/10.1016/j.fuel.2019.116423` | Resolves to "Pore-scale diffusion in nanoporous matter" — wrong paper | Springer LDHI assessment paper |
| `https://doi.org/10.1016/j.psep.2018.05.029` | Resolves to "JSA and revised Petri net" — wrong paper | ISA-18 alarm management standard |
| `https://www.aiche.org/ccps/resources/publications` | Correct content but wrong path | Fixed to /ccps/publications |
