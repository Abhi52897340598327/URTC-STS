"""Download the external fall-detection datasets used by the trainer."""

from pathlib import Path
import json

import kagglehub


DATASETS = {
    "sisfall": "adityavvvn/sisfall",
    "mobifall": "lucky01811/mobifall",
}


def main() -> None:
    paths = {name: kagglehub.dataset_download(handle) for name, handle in DATASETS.items()}
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    (output / "external_dataset_paths.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
