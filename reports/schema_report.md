# Model Compatibility & Feature Schema Report

## Data Source

- Loaded inference dataset info: {"source": "./src/data/X_test.npy", "note": "No feature name metadata found; columns are positional indices"}

## Model: xgboost_model.pkl

- Model type: XGBClassifier
- Raw object type: XGBClassifier
- n_features_in_: 70
- preprocessing_attached: None
- encoders: []

### Comparison against inference dataset
- expected count: 70
- actual count: 70
- missing features (0): []
- extra features (0): []
- misordered: False

## Model: intrusion_model.pkl

- Model type: RandomForestClassifier
- Raw object type: dict
- n_features_in_: 68
- preprocessing_attached: True
- encoders: [{'name': 'pipeline', 'type': 'Pipeline'}]

### Comparison against inference dataset
- expected count: 68
- actual count: 70
- missing features (0): []
- extra features (0): []
- misordered: False

## Recommendations

- If models expose full `feature_names`, align inference DataFrame columns to that order before prediction.
- Attach and version preprocessing pipelines (save them alongside models) so inference can reuse identical transforms.
- For models lacking feature names, prefer retraining with a pipeline that stores column metadata (or wrap into a `Pipeline`).
- Add a strict validator that checks column names, dtypes, and ordering before prediction (see utilities in `src/diagnostics/schema_checker.py`).
