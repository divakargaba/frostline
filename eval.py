"""Reproduce the frozen seed experiment; optionally run the real 3W pilot."""
import argparse
import json
from src.research import ROOT, experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Also run grouped real-well evaluation (download first)")
    args = parser.parse_args()
    report = experiment()
    out = ROOT / "data/processed/research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "seed_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print("Frozen final test: Jan 21-30; detection delay is AFTER the label onset.")
    print(f"{'System':34} {'Bad hours':>10} {'FP hours':>10} {'Delay(h)':>10}")
    for system in report["systems"]:
        m = system["test"]
        print(f"{system['name']:34} {m['true_positive']:>7}/{m['bad_hours']:<2} {m['false_positive']:>10} {str(m['delay_hours']):>10}")
    print(f"Policy: {report['policy_id']} | selection: {report['selection']['selected_trial']}/24 | 1 test incident")
    if args.real:
        from src.real_pilot import run
        run()


if __name__ == "__main__":
    main()
