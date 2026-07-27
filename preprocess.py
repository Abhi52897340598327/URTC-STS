"""Feature extraction and leakage-safe loading for fall recordings."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


SIGNALS = ("magnitude", "x", "y", "z")


def _moments(values: np.ndarray) -> list[float]:
    """Return stable descriptive statistics for one signal."""
    series = pd.Series(values)
    return [
        float(np.mean(values)),
        float(np.std(values)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
        float(np.ptp(values)),
        float(np.sqrt(np.mean(np.square(values)))),
        float(series.skew()) if len(values) > 2 else 0.0,
        float(series.kurt()) if len(values) > 3 else 0.0,
    ]


def extract_array_features(axes: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Extract the common feature schema from an n-by-3 sensor trial."""
    axes = np.asarray(axes, dtype=float)
    axes = axes[np.isfinite(axes).all(axis=1)]
    if len(axes) < 5:
        raise ValueError("Trial contains fewer than five valid samples")
    frame = pd.DataFrame(axes, columns=["x", "y", "z"])
    frame["magnitude"] = np.linalg.norm(axes, axis=1)
    stat_names = ("mean", "std", "min", "max", "median", "q25", "q75", "range", "rms", "skew", "kurtosis")
    values: list[float] = []
    names: list[str] = []
    for signal in SIGNALS:
        values.extend(_moments(frame[signal].to_numpy(dtype=float)))
        names.extend(f"{signal}_{stat}" for stat in stat_names)

    magnitude = frame["magnitude"].to_numpy(dtype=float)
    jerk = np.diff(magnitude)
    extra = [
        float(np.mean(np.abs(jerk))),
        float(np.std(jerk)),
        float(np.max(np.abs(jerk))),
        float(np.corrcoef(axes[:, 0], axes[:, 1])[0, 1]),
        float(np.corrcoef(axes[:, 0], axes[:, 2])[0, 1]),
        float(np.corrcoef(axes[:, 1], axes[:, 2])[0, 1]),
    ]
    values.extend(extra)
    names.extend(("jerk_abs_mean", "jerk_std", "jerk_abs_max", "corr_xy", "corr_xz", "corr_yz"))
    return np.nan_to_num(np.asarray(values), nan=0.0, posinf=0.0, neginf=0.0), names


def extract_features(csv_path: str | Path) -> tuple[np.ndarray, list[str]]:
    """Convert one headerless time,magnitude,x,y,z CSV trial to one feature row."""
    frame = pd.read_csv(csv_path, header=None, names=["time", *SIGNALS])
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    try:
        return extract_array_features(frame[["x", "y", "z"]].to_numpy())
    except ValueError as exc:
        raise ValueError(f"{csv_path}: {exc}") from exc


def load_dataset(data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load every raw trial; labels are 1=fall and 0=no fall."""
    root = Path(data_dir)
    records: list[tuple[Path, int]] = []
    records += [(path, 1) for path in sorted((root / "raw-fall").glob("*.csv"))]
    records += [(path, 0) for path in sorted((root / "raw-no-fall").glob("*.csv"))]
    if not records:
        raise FileNotFoundError(f"No CSV recordings found below {root.resolve()}")
    rows, labels, files = [], [], []
    feature_names: list[str] = []
    for path, label in records:
        row, feature_names = extract_features(path)
        rows.append(row)
        labels.append(label)
        files.append(str(path))
    return np.vstack(rows), np.asarray(labels), files, feature_names


def _load_sisfall(root: Path) -> tuple[list[np.ndarray], list[int], list[dict], list[str]]:
    dataset_roots = list(root.glob("**/SisFall_dataset/SisFall_dataset"))
    if not dataset_roots:
        raise FileNotFoundError(f"Could not find SisFall_dataset below {root}")
    rows, labels, metadata = [], [], []
    names: list[str] = []
    for path in sorted(dataset_roots[0].glob("S[AE]*/[DF]*.txt")):
        # First three columns are ADXL345 acceleration; 0.00390625 converts counts to g.
        frame = pd.read_csv(path, header=None, sep=",", usecols=[0, 1, 2], engine="c")
        frame[2] = frame[2].astype(str).str.rstrip(";")
        axes = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float) / 256.0
        row, names = extract_array_features(axes)
        label = int(path.name.startswith("F"))
        rows.append(row)
        labels.append(label)
        metadata.append({"source": "sisfall", "recording": str(path), "group": f"sisfall:{path.parent.name}", "label": label})
    return rows, labels, metadata, names


def _load_mobifall(root: Path) -> tuple[list[np.ndarray], list[int], list[dict], list[str]]:
    csv_files = list(root.glob("**/mobifall_dataset.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Could not find mobifall_dataset.csv below {root}")
    frame = pd.read_csv(csv_files[0])
    rows, labels, metadata = [], [], []
    names: list[str] = []
    # Kaggle's aggregate CSV is shuffled. Sorting within subject/activity and splitting
    # at >100 ms timestamp gaps reconstructs the original continuous trials.
    for (subject, activity, label), group in frame.groupby(["subject", "activity", "label"], sort=True):
        group = group.sort_values("timestamp")
        segment_ids = group["timestamp"].diff().gt(100_000_000).cumsum()
        for segment_id, segment in group.groupby(segment_ids):
            if len(segment) < 100:
                continue
            # These exported x/y/z channels are orientation angles. Map degrees to a
            # bounded common numerical range while retaining motion shape.
            axes = segment[["x", "y", "z"]].to_numpy(dtype=float) / 180.0
            row, names = extract_array_features(axes)
            label_int = int(label)
            rows.append(row)
            labels.append(label_int)
            recording = f"subject-{subject}/{activity}/segment-{int(segment_id)}"
            metadata.append({"source": "mobifall", "recording": recording, "group": f"mobifall:{subject}", "label": label_int})
    return rows, labels, metadata, names


def load_combined_dataset(data_dir: str | Path = "data", paths_json: str | Path = "artifacts/external_dataset_paths.json", cache_path: str | Path = "data/processed/external_features.joblib"):
    """Load local, SisFall, and MobiFall trials with participant grouping metadata."""
    X, y, files, names = load_dataset(data_dir)
    metadata = [{"source": "local", "recording": file, "group": f"local:{Path(file).stem}", "label": int(label)} for file, label in zip(files, y)]
    paths_file = Path(paths_json)
    if not paths_file.exists():
        return X, y, metadata, names
    cache = Path(cache_path)
    if cache.exists() and cache.stat().st_mtime >= paths_file.stat().st_mtime:
        external = joblib.load(cache)
    else:
        paths = json.loads(paths_file.read_text(encoding="utf-8"))
        sis = _load_sisfall(Path(paths["sisfall"]))
        mobi = _load_mobifall(Path(paths["mobifall"]))
        external = {
            "X": np.vstack(sis[0] + mobi[0]),
            "y": np.asarray(sis[1] + mobi[1]),
            "metadata": sis[2] + mobi[2],
            "feature_names": sis[3],
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(external, cache)
    if external["feature_names"] != names:
        raise ValueError("External and local feature schemas differ")
    return np.vstack([X, external["X"]]), np.concatenate([y, external["y"]]), metadata + external["metadata"], names
