"""Classify a new raw accelerometer recording with the trained model."""

import argparse
import joblib
import math

from preprocess import extract_features


def fall_score(model, features):
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(features)[0, 1])
    if hasattr(model, "decision_function"):
        decision = float(model.decision_function(features)[0])
        return 1.0 / (1.0 + math.exp(-decision))
    return float(model.predict(features)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Raw headerless time,magnitude,x,y,z CSV")
    parser.add_argument("--model", default="artifacts/fall_svm.joblib")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    features, names = extract_features(args.csv)
    if names != bundle["feature_names"]:
        raise ValueError("Feature schema does not match the saved model")
    probability = fall_score(bundle["model"], features.reshape(1, -1))
    print(f"Prediction: {'FALL' if probability >= 0.5 else 'NO FALL'}")
    print(f"Fall probability: {probability:.3f}")


if __name__ == "__main__":
    main()
