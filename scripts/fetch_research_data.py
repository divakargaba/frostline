"""Download a reproducible, bounded REAL 3W pilot; never select by model score.

All real class 8 recordings; first recording per well for 6/7;
first recording from each of the first 12 normal wells (lexical order).
Upstream revision, selection rule and file hashes are persisted before training.
Raw recordings and outputs are gitignored. Dataset license: CC BY 4.0.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/raw/research"
PINNED_REVISION = "6a13bd21a02a2ce6ce23a02b1c2ba2f73770752b"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest_path = DEST / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        revision = PINNED_REVISION
        selected = []
        for label in [0, 6, 7, 8]:
            response = requests.get(f"https://api.github.com/repos/petrobras/3W/contents/dataset/{label}?ref={revision}", timeout=60)
            response.raise_for_status()
            files = sorted([f for f in response.json() if f["name"].startswith("WELL-") and f["name"].endswith(".parquet")], key=lambda x: x["name"])
            seen = set()
            for file in files:
                well = file["name"].split("_")[0]
                if label != 8 and well in seen:
                    continue
                if label == 0 and len(seen) >= 12:
                    break
                seen.add(well)
                selected.append({"class": label, "well": well, "name": file["name"], "bytes": file["size"], "url": file["download_url"]})
        manifest = {"revision": revision, "license": "CC BY 4.0", "source": "https://github.com/petrobras/3W", "selection": "All real class 8; first recording per well for class 6/7; first recording from first 12 normal wells, sorted by filename. No class 9 in this production-line pilot.", "files": selected}
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def fetch(item):
        path = DEST / str(item["class"]) / item["name"]
        path.parent.mkdir(exist_ok=True)
        expected = item.get("sha256")
        valid = path.exists() and path.stat().st_size == item["bytes"]
        if valid and expected:
            valid = hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if not valid:
            response = requests.get(item["url"], timeout=180)
            response.raise_for_status()
            if len(response.content) != item["bytes"]:
                raise ValueError(f"Size mismatch: {item['name']}")
            if expected and hashlib.sha256(response.content).hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {item['name']}")
            temp = path.with_suffix(".download")
            temp.write_bytes(response.content)
            temp.replace(path)
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return item

    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest["files"] = list(pool.map(fetch, manifest["files"]))
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Verified {len(manifest['files'])} recordings at {manifest['revision']}")


if __name__ == "__main__":
    main()
