"""Build the frozen four-well replay bundle from the existing verified subset."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fleet_model import BUNDLE_PATH, REPORT_PATH, train_bundle

if __name__ == "__main__":
    report = train_bundle()
    print(json.dumps({"bundle": str(BUNDLE_PATH), "report": str(REPORT_PATH),
                      "model_id": report["model_id"], "policy": report["selected_policy"],
                      "selected_candidate": report["selected_candidate"],
                      "model_comparison": [{key: item[key] for key in ["id", "qualifies", "selected", "policy", "validation"]}
                                           for item in report["model_comparison"]],
                      "stages": report["stages"]}, indent=2))
