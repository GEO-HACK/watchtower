import os
import json
import joblib
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

try:
    import xgboost as xgb
except Exception:
    xgb = None


def find_model_files(models_dir: str) -> List[str]:
    if not os.path.exists(models_dir):
        return []
    return [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.lower().endswith(('.pkl', '.joblib', '.model'))]


def load_artifact(path: str) -> Any:
    # Try joblib then pickle
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None


def extract_feature_names_from_estimator(est) -> Optional[List[str]]:
    # Best-effort heuristics to extract feature names
    if est is None:
        return None
    # Scikit-learn common
    if hasattr(est, 'feature_names_in_'):
        try:
            return list(getattr(est, 'feature_names_in_'))
        except Exception:
            pass
    if hasattr(est, 'get_feature_names_out'):
        try:
            names = est.get_feature_names_out()
            return list(names)
        except Exception:
            pass
    # Pipeline: look for named_steps and ColumnTransformer
    if hasattr(est, 'named_steps'):
        # try to find a ColumnTransformer inside pipeline
        for name, step in est.named_steps.items():
            if step is None:
                continue
            # ColumnTransformer
            if hasattr(step, 'transformers_'):
                # transformers_ is list of (name, transformer, columns)
                cols = []
                for tr_name, transformer, columns in step.transformers_:
                    if isinstance(columns, (list, tuple, np.ndarray)):
                        cols.extend(list(columns))
                if cols:
                    return [str(c) for c in cols]
            # If transformer itself exposes feature names
            names = extract_feature_names_from_estimator(step)
            if names:
                return names
    # XGBoost native
    if xgb is not None:
        try:
            if isinstance(est, xgb.Booster):
                return list(est.feature_names) if est.feature_names is not None else None
            # sklearn wrapper
            if hasattr(est, 'get_booster'):
                booster = est.get_booster()
                if booster is not None and getattr(booster, 'feature_names', None) is not None:
                    return list(booster.feature_names)
        except Exception:
            pass
    # Fallback: check for metadata dict
    if isinstance(est, dict):
        for key in ['feature_names', 'feature_order', 'features']:
            if key in est:
                return list(est[key])
    return None


def extract_n_features(est) -> Optional[int]:
    if est is None:
        return None
    if hasattr(est, 'n_features_in_'):
        try:
            return int(getattr(est, 'n_features_in_'))
        except Exception:
            pass
    names = extract_feature_names_from_estimator(est)
    if names is not None:
        return len(names)
    # xgboost Booster
    if xgb is not None:
        try:
            if isinstance(est, xgb.Booster):
                return len(est.feature_names) if est.feature_names is not None else None
        except Exception:
            pass
    if isinstance(est, dict) and 'n_features' in est:
        try:
            return int(est['n_features'])
        except Exception:
            pass
    return None


def inspect_model_file(path: str) -> Dict[str, Any]:
    obj = load_artifact(path)
    model = None
    if isinstance(obj, dict) and 'model' in obj:
        model = obj['model']
    else:
        model = obj

    info: Dict[str, Any] = {
        'path': path,
        'raw_object_type': type(obj).__name__ if obj is not None else None,
        'model_type': type(model).__name__ if model is not None else None,
        'n_features_in_': extract_n_features(model),
        'feature_names': extract_feature_names_from_estimator(model),
        'expects_transformed': False,
        'preprocessing_attached': None,
        'encoders': [],
        'notes': [],
    }

    # Heuristics: if object is Pipeline or contains named_steps, we consider preprocess attached
    if hasattr(model, 'named_steps'):
        info['preprocessing_attached'] = True
        info['expects_transformed'] = True
        # inspect steps for encoders
        try:
            for name, step in model.named_steps.items():
                step_type = type(step).__name__
                info['encoders'].append({'name': name, 'type': step_type})
        except Exception:
            pass
    else:
        # If raw object dict includes preprocessing key
        if isinstance(obj, dict):
            for key in ['preprocessor', 'preprocessing', 'pipeline']:
                if key in obj:
                    info['preprocessing_attached'] = True
                    info['expects_transformed'] = True
                    info['encoders'].append({'name': key, 'type': type(obj[key]).__name__})

    # XGBoost heuristics: if Booster exists, feature names from booster
    try:
        if xgb is not None and hasattr(model, 'get_booster'):
            booster = model.get_booster()
            if booster is not None and getattr(booster, 'feature_names', None) is not None:
                info['feature_names'] = list(booster.feature_names)
                info['n_features_in_'] = len(info['feature_names'])
    except Exception:
        pass

    # Data types: not usually stored, but try to infer from training metadata dict
    if isinstance(obj, dict):
        if 'dtypes' in obj:
            info['dtypes'] = obj['dtypes']
        if 'feature_types' in obj:
            info['feature_types'] = obj['feature_types']

    return info


def load_inference_dataset(data_dir: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    # Try to find a tabular dataset or numpy arrays
    info = {}
    x_path = os.path.join(data_dir, 'X_test.npy')
    features_path = os.path.join(data_dir, 'feature_names.json')
    if os.path.exists(x_path):
        arr = np.load(x_path, allow_pickle=True)
        if arr.dtype.names is not None:
            # structured array with field names
            df = pd.DataFrame(arr)
            info['source'] = x_path
            return df, info
        # Try to load feature names
        if os.path.exists(features_path):
            try:
                with open(features_path, 'r') as f:
                    names = json.load(f)
                df = pd.DataFrame(arr, columns=names)
                info['source'] = x_path
                info['feature_names_file'] = features_path
                return df, info
            except Exception:
                pass
        # If no column names, return DataFrame with integer columns
        df = pd.DataFrame(arr)
        info['source'] = x_path
        info['note'] = 'No feature name metadata found; columns are positional indices'
        return df, info
    # fallback: try CSV or parquet
    for ext in ('.csv', '.parquet', '.pkl'):
        for f in os.listdir(data_dir):
            if f.lower().endswith(ext):
                try:
                    p = os.path.join(data_dir, f)
                    if ext == '.csv':
                        df = pd.read_csv(p)
                    elif ext == '.parquet':
                        df = pd.read_parquet(p)
                    else:
                        df = pd.read_pickle(p)
                    info['source'] = p
                    return df, info
                except Exception:
                    continue
    return None, info


def compare_schemas(df: pd.DataFrame, model_info: Dict[str, Any]) -> Dict[str, Any]:
    report = {'model_path': model_info.get('path')}
    expected_names = model_info.get('feature_names')
    n_expected = model_info.get('n_features_in_')
    report['n_expected'] = n_expected
    report['n_actual'] = None
    if df is None:
        report['error'] = 'No inference dataframe available'
        return report
    report['n_actual'] = df.shape[1]
    actual_cols = list(df.columns.astype(str))
    report['actual_columns'] = actual_cols

    if expected_names is None:
        # try best-effort: compare counts
        report['matching'] = (n_expected == df.shape[1]) if n_expected is not None else None
        report['missing'] = []
        report['extra'] = []
        report['misordered'] = False
        report['notes'] = ['Model did not expose feature names; comparison is positional']
        return report

    report['expected_columns'] = [str(x) for x in expected_names]
    set_expected = set(map(str, expected_names))
    set_actual = set(actual_cols)
    report['missing'] = sorted(list(set_expected - set_actual))
    report['extra'] = sorted(list(set_actual - set_expected))
    # misordered: when both sets equal but order differs
    if set_expected == set_actual:
        report['misordered'] = (list(map(str, expected_names)) != actual_cols)
    else:
        report['misordered'] = None

    # dtype mismatches: if model stored dtypes
    dtypes_issues = []
    if 'dtypes' in model_info and isinstance(model_info['dtypes'], dict):
        for k, exp_dt in model_info['dtypes'].items():
            if k in df.columns:
                act = str(df[k].dtype)
                if exp_dt != act:
                    dtypes_issues.append({'column': k, 'expected': exp_dt, 'actual': act})
    report['dtype_issues'] = dtypes_issues

    # detect categorical unseen values: if encoder attached and categorical columns present
    unseen_values = {}
    if model_info.get('preprocessing_attached') and 'encoders' in model_info:
        # Best-effort: look for OneHotEncoder or LabelEncoder presence; if model metadata includes categories
        if isinstance(model_info.get('feature_names'), list):
            for enc in model_info.get('encoders', []):
                if 'OneHot' in enc.get('type', '') or 'OneHotEncoder' in enc.get('type', ''):
                    # if categories available in model_info dict
                    if isinstance(model_info.get('raw_object_type'), str):
                        pass
    report['unseen_categorical'] = unseen_values
    return report


def generate_diagnostic_report(models_dir: str, data_dir: str, out_path: str = 'reports/schema_report.md') -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    models = find_model_files(models_dir)
    summary = {'models_examined': [], 'data_source': None, 'comparisons': []}

    df, data_info = load_inference_dataset(data_dir)
    summary['data_source'] = data_info

    for m in models:
        info = inspect_model_file(m)
        # store raw object if available
        obj = load_artifact(m)
        if isinstance(obj, dict):
            info['raw_meta_keys'] = list(obj.keys())
        summary['models_examined'].append(info)
        comp = compare_schemas(df, info)
        summary['comparisons'].append(comp)

    # Write human-readable markdown
    lines = []
    lines.append('# Model Compatibility & Feature Schema Report')
    lines.append('')
    lines.append('## Data Source')
    lines.append('')
    lines.append(f'- Loaded inference dataset info: {json.dumps(data_info)}')
    lines.append('')

    for info, comp in zip(summary['models_examined'], summary['comparisons']):
        lines.append(f"## Model: {os.path.basename(info.get('path',''))}")
        lines.append('')
        lines.append(f"- Model type: {info.get('model_type')}")
        lines.append(f"- Raw object type: {info.get('raw_object_type')}")
        lines.append(f"- n_features_in_: {info.get('n_features_in_')}")
        lines.append(f"- preprocessing_attached: {info.get('preprocessing_attached')}")
        lines.append(f"- encoders: {info.get('encoders')}")
        if info.get('feature_names') is not None:
            lines.append(f"- feature_names ({len(info.get('feature_names'))}): {info.get('feature_names')}")
        lines.append('')
        lines.append('### Comparison against inference dataset')
        if 'error' in comp:
            lines.append(f"- Error: {comp['error']}")
        else:
            lines.append(f"- expected count: {comp.get('n_expected')}")
            lines.append(f"- actual count: {comp.get('n_actual')}")
            lines.append(f"- missing features ({len(comp.get('missing', []))}): {comp.get('missing')}")
            lines.append(f"- extra features ({len(comp.get('extra', []))}): {comp.get('extra')}")
            lines.append(f"- misordered: {comp.get('misordered')}")
            if comp.get('dtype_issues'):
                lines.append(f"- dtype issues: {comp.get('dtype_issues')}")
        lines.append('')

    # Add recommendations
    lines.append('## Recommendations')
    lines.append('')
    lines.append('- If models expose full `feature_names`, align inference DataFrame columns to that order before prediction.')
    lines.append('- Attach and version preprocessing pipelines (save them alongside models) so inference can reuse identical transforms.')
    lines.append('- For models lacking feature names, prefer retraining with a pipeline that stores column metadata (or wrap into a `Pipeline`).')
    lines.append('- Add a strict validator that checks column names, dtypes, and ordering before prediction (see utilities in `src/diagnostics/schema_checker.py`).')
    lines.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    # Return summary for programmatic use
    summary['report_path'] = out_path
    return summary


# ----------------- Utility functions exposed for user ----------------- #
def print_model_expectations(models_dir: str):
    models = find_model_files(models_dir)
    for m in models:
        info = inspect_model_file(m)
        print('Model:', os.path.basename(m))
        print('  type:', info.get('model_type'))
        print('  n_features_in_:', info.get('n_features_in_'))
        print('  preprocessing_attached:', info.get('preprocessing_attached'))
        print('  sample feature names:', (info.get('feature_names') or [])[:20])
        print()


def validate_inference_df(df: pd.DataFrame, model_info: Dict[str, Any]) -> Tuple[bool, List[str]]:
    # Returns (is_valid, list_of_warnings)
    warnings = []
    if df is None:
        return False, ['No dataframe provided']
    expected = model_info.get('feature_names')
    if expected is None:
        # Only check counts
        n = model_info.get('n_features_in_')
        if n is not None and df.shape[1] != n:
            warnings.append(f"Expected {n} features but dataframe has {df.shape[1]}")
            return False, warnings
        return True, warnings
    # check missing
    missing = [c for c in expected if str(c) not in df.columns.astype(str).tolist()]
    extra = [c for c in df.columns.astype(str).tolist() if c not in list(map(str, expected))]
    if missing:
        warnings.append(f"Missing features: {missing}")
    if extra:
        warnings.append(f"Extra features: {extra}")
    # dtype checks
    if 'feature_types' in model_info:
        for col, exp_t in model_info['feature_types'].items():
            if col in df.columns and str(df[col].dtype) != exp_t:
                warnings.append(f"Column {col} dtype mismatch: expected {exp_t} got {df[col].dtype}")
    return (len(warnings) == 0), warnings


def align_columns(df: pd.DataFrame, model_info: Dict[str, Any], fill_value=0) -> pd.DataFrame:
    expected = model_info.get('feature_names')
    if expected is None:
        # If no expected names, try to match by count
        n = model_info.get('n_features_in_')
        if n is None:
            return df
        cur = df.shape[1]
        if cur == n:
            return df
        if cur < n:
            # pad extra columns
            for i in range(n - cur):
                df[f'_pad_{i}'] = fill_value
            return df
        return df.iloc[:, :n]
    # Reorder and add missing columns
    cols = list(map(str, expected))
    out = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            out[c] = df[c]
        else:
            out[c] = fill_value
    return out


def warn_schema_mismatch(df: pd.DataFrame, model_info: Dict[str, Any]) -> None:
    valid, warnings = validate_inference_df(df, model_info)
    if valid:
        print('Schema OK for model:', os.path.basename(model_info.get('path','')))
    else:
        print('Schema WARNINGS for model:', os.path.basename(model_info.get('path','')))
        for w in warnings:
            print(' -', w)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Run schema diagnostics for saved models')
    p.add_argument('--models_dir', default='./src/models')
    p.add_argument('--data_dir', default='./src/data')
    p.add_argument('--out', default='reports/schema_report.md')
    args = p.parse_args()

    print('Inspecting models in', args.models_dir)
    summary = generate_diagnostic_report(args.models_dir, args.data_dir, args.out)
    print('Report written to', summary.get('report_path'))
