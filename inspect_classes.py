#!/usr/bin/env python3
"""Inspect model classes and their meanings."""
import os
import sys
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

models = [
    "src/models/random_forest.pkl",
    "src/models/xgboost_model.pkl",
]

print("=" * 80)
print("MODEL CLASS INSPECTION")
print("=" * 80)

for path in models:
    if not os.path.exists(path):
        print(f"\n{path}: NOT FOUND")
        continue
    
    obj = joblib.load(path)
    model = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    
    print(f"\n{os.path.basename(path)}")
    print("-" * 60)
    print(f"  Model type: {type(model).__name__}")
    print(f"  Features: {getattr(model, 'n_features_in_', 'unknown')}")
    
    # Extract classes
    classes = getattr(model, 'classes_', None)
    if classes is not None:
        print(f"  Classes: {list(classes)}")
        print(f"  Number of classes: {len(classes)}")
        for i, cls in enumerate(classes):
            print(f"    Class {i}: {cls}")
    else:
        print(f"  Classes: NOT AVAILABLE (model doesn't expose classes_)")
    
    # Check if object has metadata
    if isinstance(obj, dict):
        print(f"\n  Extra metadata in saved object:")
        for key, val in obj.items():
            if key != 'model':
                if isinstance(val, (str, int, float, bool)):
                    print(f"    {key}: {val}")
                elif isinstance(val, (list, dict)):
                    print(f"    {key}: {type(val).__name__} ({len(val)} items)")
                else:
                    print(f"    {key}: {type(val).__name__}")

# Try to infer from y_test if available
print("\n" + "=" * 80)
print("GROUND TRUTH CLASS DISTRIBUTION")
print("=" * 80)

y_test_candidates = [
    "src/data/y_test.npy",
    "src/data/y_test .npy",
]

for path in y_test_candidates:
    if os.path.exists(path):
        import numpy as np
        y = np.load(path)
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nFrom {os.path.basename(path)}:")
        print(f"  Total samples: {len(y)}")
        print(f"  Unique classes: {sorted(unique)}")
        for cls, cnt in sorted(zip(unique, counts)):
            pct = (cnt / len(y)) * 100
            print(f"    Class {int(cls)}: {int(cnt)} samples ({pct:.2f}%)")
        break
else:
    print("\nNo y_test file found")

print("\n" + "=" * 80)
print("NOTE: Class 0 typically represents 'BENIGN' traffic.")
print("Non-zero classes typically represent different types of ATTACKS.")
print("Run models on actual data to see what attacks are predicted.")
print("=" * 80)
