"""Export the trained scikit-learn SVM into a browser-readable JSON artifact."""

import argparse
import json
from pathlib import Path

import joblib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/fall_svm.joblib")
    parser.add_argument("--output", default="mobile-scaffold/src/models/fall-svm.json")
    args = parser.parse_args()

    artifact = joblib.load(args.input)
    pipeline = artifact["model"]
    scaler = pipeline.named_steps["scale"]
    svm = pipeline.named_steps["svm"]

    exported = {
        "modelType": "rbf_svc",
        "classes": svm.classes_.tolist(),
        "gamma": float(svm._gamma),
        "intercept": svm.intercept_.astype(float).tolist(),
        "dualCoef": svm.dual_coef_.astype(float).tolist(),
        "supportVectors": svm.support_vectors_.astype(float).tolist(),
        "scalerMean": scaler.mean_.astype(float).tolist(),
        "scalerScale": scaler.scale_.astype(float).tolist(),
        "featureNames": artifact["feature_names"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(exported, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "supportVectorCount": len(exported["supportVectors"]),
        "featureCount": len(exported["featureNames"]),
        "classes": exported["classes"],
        "gamma": exported["gamma"],
        "sizeBytes": output.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
