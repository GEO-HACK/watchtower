import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Import diagnostics utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diagnostics.schema_checker import inspect_model_file


def load_maybe_dict_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
    else:
        model = obj
    return model, obj


def predict_with_model(model, X):
    # If model has predict_proba, return probs and preds
    preds = None
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            preds = np.argmax(proba, axis=1)
        else:
            preds = model.predict(X)
    except Exception:
        preds = model.predict(X)
    return preds, proba


def validate_schema_compatibility(model1_info, model2_info, data_shape):
    """Validate that both models expect the same number of features.
    
    Returns:
        (is_compatible: bool, warnings: list, expected_n_features: int)
    """
    warnings = []
    fatal = False
    n1 = model1_info.get('n_features_in_')
    n2 = model2_info.get('n_features_in_')
    n_actual = data_shape[1] if len(data_shape) > 1 else None
    
    if n1 is None or n2 is None:
        warnings.append("One or both models did not expose n_features_in_; schema validation limited.")
        expected_n = n1 or n2 or n_actual
    else:
        if n1 != n2:
            warnings.append(f"CRITICAL: Model feature mismatch! Model1 expects {n1} features, Model2 expects {n2}. Retrain one model.")
            fatal = True
            return False, warnings, None
        expected_n = n1
    
    if n_actual is not None and expected_n is not None and n_actual != expected_n:
        warnings.append(f"Data feature count ({n_actual}) != expected ({expected_n}). Alignment may be needed.")
    
    return (not fatal), warnings, expected_n


def combine_predictions(preds1, preds2, proba1=None, proba2=None, strategy="majority"):
    """Fusion layer: combine predictions from two models using various strategies."""
    if strategy == "or":
        # If any model flags a non-zero (non-benign) class, choose that label.
        combined = []
        for a, b in zip(preds1, preds2):
            if a != 0 and b != 0:
                combined.append(a)
            elif a != 0:
                combined.append(a)
            elif b != 0:
                combined.append(b)
            else:
                combined.append(0)
        return np.array(combined)

    if strategy == "avg_proba" and proba1 is not None and proba2 is not None:
        # average predicted probabilities then argmax
        avg = (proba1 + proba2) / 2.0
        return np.argmax(avg, axis=1)
    
    if strategy == "confidence_weighted" and proba1 is not None and proba2 is not None:
        # Weight predictions by max confidence per sample
        c1 = np.max(proba1, axis=1)
        c2 = np.max(proba2, axis=1)
        combined = []
        for i, (a, b) in enumerate(zip(preds1, preds2)):
            if c1[i] > c2[i]:
                combined.append(a)
            else:
                combined.append(b)
        return np.array(combined)
    
    if strategy == "unanimous_or_majority":
        # If both models agree, use that; else if either flags anomaly (non-zero), use anomaly
        combined = []
        for a, b in zip(preds1, preds2):
            if a == b:
                combined.append(a)
            elif a != 0 or b != 0:
                combined.append(max(a, b) if a != 0 else b if b != 0 else 0)
            else:
                combined.append(a)
        return np.array(combined)

    # default majority: pick the label agreed by both, else pick preds1
    combined = []
    for a, b in zip(preds1, preds2):
        if a == b:
            combined.append(a)
        else:
            combined.append(a)
    return np.array(combined)


def load_live_csv_features(csv_path):
    """Load live stream CSV and return numeric feature matrix plus preview dataframe."""
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Live CSV is empty: {csv_path}")

    preview_cols = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "timestamp"]
    available_preview = [c for c in preview_cols if c in df.columns]
    preview_df = df[available_preview].copy() if available_preview else pd.DataFrame(index=df.index)

    drop_cols = ["src_ip", "dst_ip", "timestamp"]
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return feature_df.to_numpy(), preview_df


def main():
    p = argparse.ArgumentParser(description="Hybrid detector combining two saved models (unified feature schema)")
    p.add_argument("--models_dir", default="./src/models", help="Directory where models live (local)")
    p.add_argument("--data_dir", default="./src/data", help="Directory where test data lives (local)")
    p.add_argument("--model1", default=None, help="Path to first model (joblib, e.g., random_forest.pkl)")
    p.add_argument("--model2", default=None, help="Path to second model (joblib, e.g., xgboost_model.pkl)")
    p.add_argument("--pipeline", default=None, help="Path to preprocessing pipeline (joblib)")
    p.add_argument("--x_test", default=None, help="Path to X_test (.npy)")
    p.add_argument("--y_test", default=None, help="Path to y_test (.npy)")
    p.add_argument("--live_csv", default=None, help="Path to live stream CSV (e.g., src/data/live_stream.csv)")
    p.add_argument("--top", type=int, default=10, help="Number of sample rows to print for live CSV mode")
    p.add_argument("--strategy", choices=["majority","or","avg_proba","confidence_weighted","unanimous_or_majority"], default="majority", help="Fusion strategy")
    args = p.parse_args()

    models_dir = args.models_dir
    data_dir = args.data_dir
    model1_path = args.model1 or os.path.join(models_dir, "random_forest.pkl")
    model2_path = args.model2 or os.path.join(models_dir, "xgboost_model.pkl")
    pipeline_path = args.pipeline or os.path.join(models_dir, "preprocessing_pipeline.pkl")
    x_test_path = args.x_test or os.path.join(data_dir, "X_test.npy")
    y_test_path = args.y_test or os.path.join(data_dir, "y_test.npy")

    print("Loading models and artifacts...")
    m1, raw1 = load_maybe_dict_model(model1_path)
    m2, raw2 = load_maybe_dict_model(model2_path)
    
    # Inspect models for schema diagnostics
    model1_info = inspect_model_file(model1_path)
    model2_info = inspect_model_file(model2_path)
    
    print(f"\n[Model 1] {os.path.basename(model1_path)}: {model1_info.get('n_features_in_')} features, type={model1_info.get('model_type')}")
    print(f"[Model 2] {os.path.basename(model2_path)}: {model2_info.get('n_features_in_')} features, type={model2_info.get('model_type')}")

    pipeline = None
    if os.path.exists(pipeline_path):
        try:
            pipeline = joblib.load(pipeline_path)
            print(f"[Preprocessing] Loaded pipeline from {os.path.basename(pipeline_path)}")
        except Exception as e:
            print(f"[Preprocessing] Could not load pipeline: {e}")

    live_mode = args.live_csv is not None
    live_preview = None

    if live_mode:
        if not os.path.exists(args.live_csv):
            raise FileNotFoundError(f"Live CSV not found: {args.live_csv}")
        X_test, live_preview = load_live_csv_features(args.live_csv)
        y_test = None
        print(f"\n[Data] Live CSV mode: {args.live_csv}")
        print(f"[Data] X_live shape: {X_test.shape}")
    else:
        if not os.path.exists(x_test_path) or not os.path.exists(y_test_path):
            # Check for y_test with space in filename
            alt_y_test_path = os.path.join(data_dir, "y_test .npy")
            if os.path.exists(x_test_path) and os.path.exists(alt_y_test_path):
                y_test_path = alt_y_test_path
            else:
                raise FileNotFoundError(f"Missing test arrays at {x_test_path} or {y_test_path}")

        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        print(f"\n[Data] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Validate schema compatibility between models
    is_compatible, schema_warnings, expected_n = validate_schema_compatibility(model1_info, model2_info, X_test.shape)
    if schema_warnings:
        for w in schema_warnings:
            print(f"[Schema Warning] {w}")
    if not is_compatible:
        print("\n[FATAL] Schema incompatibility detected. Both models must have the same feature count.")
        print("Action: Retrain the model with fewer features to match the expanded dataset.")
        sys.exit(1)
    
    print(f"\n[Schema] ✓ Both models expect {expected_n} features; data has {X_test.shape[1]} features")

    # Apply preprocessing if available
    X_in = X_test
    if pipeline is not None:
        try:
            X_in = pipeline.transform(X_test)
            print(f"[Preprocessing] Applied pipeline; output shape: {X_in.shape}")
        except Exception as e:
            print(f"[Preprocessing] Pipeline not applicable ({e}); using X_test as-is")

    # With unified schema, both models expect the same feature count
    n1 = model1_info.get('n_features_in_')
    n2 = model2_info.get('n_features_in_')
    
    if n1 is not None and X_in.shape[1] != n1:
        print(f"[Warning] Model1 expects {n1} features but data has {X_in.shape[1]}; truncating")
        X_in = X_in[:, :n1]
    
    X_for_m1 = X_in
    X_for_m2 = X_in

    print(f"\n[Inference] Running both models on unified {X_in.shape[1]}-feature dataset...")
    preds1, proba1 = predict_with_model(m1, X_for_m1)
    preds2, proba2 = predict_with_model(m2, X_for_m2)

    combined = combine_predictions(preds1, preds2, proba1, proba2, strategy=args.strategy)

    if live_mode:
        print("\n" + "=" * 70)
        print(f"LIVE STREAM RESULTS (Fusion Strategy: {args.strategy})")
        print("=" * 70)

        def show_dist(name, arr):
            labels, counts = np.unique(arr, return_counts=True)
            print(f"\n[{name}] Prediction distribution:")
            for lbl, cnt in zip(labels, counts):
                pct = (cnt / len(arr)) * 100
                print(f"  Class {int(lbl)}: {int(cnt)} ({pct:.2f}%)")

        show_dist("Model 1", preds1)
        show_dist("Model 2", preds2)
        show_dist("Fused", combined)

        agreement = np.mean(preds1 == preds2)
        print(f"\n[Agreement] Model1 vs Model2: {agreement:.4f} ({int(np.sum(preds1 == preds2))}/{len(preds1)})")

        n_show = min(args.top, len(combined))
        if n_show > 0:
            print(f"\n[Sample Predictions] First {n_show} rows")
            for i in range(n_show):
                ctx = ""
                if live_preview is not None and not live_preview.empty:
                    row = live_preview.iloc[i].to_dict()
                    src_ip = row.get("src_ip", "?")
                    dst_ip = row.get("dst_ip", "?")
                    src_port = row.get("src_port", "?")
                    dst_port = row.get("dst_port", "?")
                    proto = row.get("protocol", "?")
                    ctx = f"{src_ip}:{src_port} -> {dst_ip}:{dst_port} proto={proto}"
                print(
                    f"  Row {i}: M1={int(preds1[i])}, M2={int(preds2[i])}, Fused={int(combined[i])}"
                    + (f" | {ctx}" if ctx else "")
                )

        print("=" * 70)
        return

    print("\n" + "="*70)
    print(f"RESULTS (Fusion Strategy: {args.strategy})")
    print("="*70)
    print("\n[Model 1 - Intrusion Model] Metrics:")
    print(classification_report(y_test, preds1, zero_division=0))
    
    print("\n[Model 2 - XGBoost Model] Metrics:")
    print(classification_report(y_test, preds2, zero_division=0))
    
    print(f"\n[Fusion Layer - {args.strategy} Strategy] Metrics:")
    print(classification_report(y_test, combined, zero_division=0))

    print("\n[Summary]")
    print(f"  Model1 Accuracy: {accuracy_score(y_test, preds1):.4f}")
    print(f"  Model2 Accuracy: {accuracy_score(y_test, preds2):.4f}")
    print(f"  Fused Accuracy:  {accuracy_score(y_test, combined):.4f}")
    print(f"  Fused Macro F1:  {f1_score(y_test, combined, average='macro'):.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
