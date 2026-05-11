import os
import sys
import argparse
import joblib
import logging
import numpy as np
import pandas as pd

from preprocessing.packet_capture import capture_from_pcap, get_packet_metadata, save_packets_for_flows
from preprocessing.flow_aggregator import FlowAggregator
from diagnostics.schema_checker import inspect_model_file, align_columns
from sklearn.metrics import classification_report, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_maybe_dict_model(path):
    obj = joblib.load(path)
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
    else:
        model = obj
    return model, obj


def predict_with_model(model, X):
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


def pcap_to_flow_features(pcap_path, max_packets=None, skip_packets=0):
    """Read a PCAP file and return a pandas.DataFrame of flow feature dicts."""
    aggregator = FlowAggregator()

    def handle_packet(packet):
        metadata = get_packet_metadata(packet)
        if metadata:
            aggregator.process_packet(metadata)

    capture_from_pcap(
        pcap_path,
        handle_packet,
        max_packets=max_packets,
        skip_packets=skip_packets,
    )
    aggregator.flush()
    flows = aggregator.get_completed_flows()
    if not flows:
        return pd.DataFrame()
    df = pd.DataFrame(flows)
    return df


def align_and_transform(df, model_info, pipeline_path=None):
    """Align and transform flow features for model input."""
    if df.empty:
        return df
    aligned = align_columns(df, model_info, fill_value=0)

    if pipeline_path and os.path.exists(pipeline_path):
        try:
            pipe = joblib.load(pipeline_path)
            try:
                transformed = pipe.transform(aligned)
                return transformed
            except Exception:
                return pipe.transform(aligned.values)
        except Exception as e:
            logger.warning(f'Could not load pipeline: {e}')

    if isinstance(aligned, pd.DataFrame):
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors='coerce')
        aligned = aligned.fillna(0)
        return aligned.values

    return aligned


def combine_predictions(preds1, preds2, proba1=None, proba2=None, strategy="majority"):
    """Fusion layer: combine predictions from two models."""
    if strategy == "or":
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
        avg = (proba1 + proba2) / 2.0
        return np.argmax(avg, axis=1)
    
    if strategy == "confidence_weighted" and proba1 is not None and proba2 is not None:
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
        combined = []
        for a, b in zip(preds1, preds2):
            if a == b:
                combined.append(a)
            elif a != 0 or b != 0:
                combined.append(max(a, b) if a != 0 else b if b != 0 else 0)
            else:
                combined.append(a)
        return np.array(combined)

    # default majority
    combined = []
    for a, b in zip(preds1, preds2):
        if a == b:
            combined.append(a)
        else:
            combined.append(a)
    return np.array(combined)


def main():
    parser = argparse.ArgumentParser(description='Run hybrid detection on flow features extracted from a PCAP')
    parser.add_argument('--pcap', default=os.path.join('src', 'data', 'Friday-WorkingHours.pcap'), help='Path to input PCAP file')
    parser.add_argument('--max-packets', type=int, default=1000, help='Process only this many packets from the PCAP for faster testing (default: 1000)')
    parser.add_argument('--skip-packets', type=int, default=0, help='Skip this many initial packets before processing')
    parser.add_argument('--export-misclassified', default=None, help='Directory to write per-flow PCAPs for misclassified flows')
    parser.add_argument('--export-which', choices=['model1','model2','fused'], default='model1', help='Which predictions to consider when exporting misclassified flows')
    parser.add_argument('--pad-seconds', type=float, default=0.5, help='Seconds to pad flow time window when exporting packets')
    args = parser.parse_args()

    if args.max_packets is not None and args.max_packets <= 0:
        logger.error('--max-packets must be > 0 when provided')
        sys.exit(1)

    if args.skip_packets < 0:
        logger.error('--skip-packets must be >= 0')
        sys.exit(1)

    # Setup paths
    pcap_path = args.pcap
    model1_path = os.path.join('src', 'models', 'random_forest.pkl')
    model2_path = os.path.join('src', 'models', 'xgboost_model.pkl')
    pipeline_path = os.path.join('src', 'models', 'preprocessing_pipeline.pkl')
    y_test_path = os.path.join('src', 'data', 'y_test.npy')

    # Check if y_test exists (with possible space in filename)
    y_test = None
    if not os.path.exists(y_test_path):
        alt_y_test_path = os.path.join('src', 'data', 'y_test .npy')
        if os.path.exists(alt_y_test_path):
            y_test_path = alt_y_test_path

    if os.path.exists(y_test_path):
        y_test = np.load(y_test_path)
        logger.info('Loaded ground truth labels: %d samples', len(y_test))
    else:
        logger.warning('No ground truth labels found; will show predictions only')

    # Convert PCAP to flows
    logger.info(
        'Converting PCAP to flows (skip=%d, max=%s)...',
        args.skip_packets,
        str(args.max_packets) if args.max_packets is not None else 'all',
    )
    df = pcap_to_flow_features(
        pcap_path,
        max_packets=args.max_packets,
        skip_packets=args.skip_packets,
    )
    logger.info('Flows generated: %d', len(df))

    if df.empty:
        logger.error('No flows extracted from PCAP')
        sys.exit(1)

    # Load models
    logger.info('Loading models...')
    model1, raw1 = load_maybe_dict_model(model1_path)
    model2, raw2 = load_maybe_dict_model(model2_path)

    # Inspect models
    model1_info = inspect_model_file(model1_path)
    model2_info = inspect_model_file(model2_path)

    logger.info('Model1 (%s): %s features', os.path.basename(model1_path), model1_info.get('n_features_in_'))
    logger.info('Model2 (%s): %s features', os.path.basename(model2_path), model2_info.get('n_features_in_'))

    # Align and transform features
    X = align_and_transform(df, model1_info, pipeline_path=pipeline_path)
    if isinstance(X, pd.DataFrame):
        X = X.values

    if X is None or getattr(X, 'size', 0) == 0:
        logger.error('No feature data available for inference')
        sys.exit(1)

    logger.info('Input shape for models: %s', X.shape)

    # Run inference with both models
    logger.info('Running Model1 inference...')
    preds1, proba1 = predict_with_model(model1, X)
    
    logger.info('Running Model2 inference...')
    preds2, proba2 = predict_with_model(model2, X)

    # Apply fusion strategies
    strategies = ["majority", "or", "confidence_weighted", "unanimous_or_majority"]
    fused_results = {}
    
    logger.info('Applying fusion strategies...')
    for strategy in strategies:
        fused_results[strategy] = combine_predictions(preds1, preds2, proba1, proba2, strategy=strategy)

    # Optionally export misclassified flows to PCAPs
    if args.export_misclassified:
        if y_test is None:
            logger.error('Cannot export misclassified flows: ground truth (y_test) not available')
        else:
            # choose prediction vector
            if args.export_which == 'model1':
                chosen_preds = preds1
            elif args.export_which == 'model2':
                chosen_preds = preds2
            else:
                # fused majority
                chosen_preds = fused_results['majority']

            # align y_test length with preds
            y_aligned = y_test[:len(chosen_preds)]
            mis_idx = np.where(chosen_preds != y_aligned)[0]
            logger.info('Found %d misclassified flows (exporting to %s)', len(mis_idx), args.export_misclassified)
            if len(mis_idx) > 0:
                flows_records = df.to_dict('records')
                results = save_packets_for_flows(pcap_path, flows_records, mis_idx, args.export_misclassified, pad_seconds=args.pad_seconds)
                logger.info('Exported packets for %d flows', len([v for v in results.values() if v > 0]))

    # Display results
    print("\n" + "="*80)
    print("HYBRID DETECTOR: FRIDAY-WORKINGHOURS PCAP ANALYSIS")
    print("="*80)
    print(f"\nDataset: {len(df)} flows extracted from Friday-WorkingHours.pcap")
    print(f"Features: {X.shape[1]} aligned features")

    print("\n" + "-"*80)
    print("INDEPENDENT MODEL PREDICTIONS")
    print("-"*80)
    
    unique1, counts1 = np.unique(preds1, return_counts=True)
    unique2, counts2 = np.unique(preds2, return_counts=True)
    
    print(f"\nModel1 ({os.path.basename(model1_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique1), map(int, counts1)))}")
    
    print(f"\nModel2 ({os.path.basename(model2_path)}):")
    print(f"  Prediction distribution: {dict(zip(map(int, unique2), map(int, counts2)))}")
    
    # Model agreement
    agree = np.mean(preds1 == preds2)
    print(f"\nModel Agreement: {agree:.2%} ({int(np.sum(preds1 == preds2))}/{len(preds1)})")

    print("\n" + "-"*80)
    print("FUSION LAYER RESULTS")
    print("-"*80)
    
    for strategy in strategies:
        fused = fused_results[strategy]
        unique_f, counts_f = np.unique(fused, return_counts=True)
        print(f"\n[{strategy.upper()}]")
        print(f"  Prediction distribution: {dict(zip(map(int, unique_f), map(int, counts_f)))}")
        
        if y_test is not None:
            # Check sample size match
            if len(y_test) >= len(fused):
                y_test_aligned = y_test[:len(fused)]
                acc = accuracy_score(y_test_aligned, fused)
                f1_macro = f1_score(y_test_aligned, fused, average='macro', zero_division=0)
                print(f"  Accuracy vs Ground Truth: {acc:.4f}")
                print(f"  Macro F1 Score: {f1_macro:.4f}")
            else:
                logger.warning(f"Ground truth has {len(y_test)} samples but generated {len(fused)} flows")

    # Detailed metrics if ground truth available
    if y_test is not None and len(y_test) >= len(fused):
        print("\n" + "-"*80)
        print("DETAILED CLASSIFICATION METRICS (vs Ground Truth)")
        print("-"*80)
        
        y_test_aligned = y_test[:len(fused)]
        
        print(f"\nModel1 Classification Report:")
        print(classification_report(y_test_aligned, preds1, zero_division=0))
        
        print(f"\nModel2 Classification Report:")
        print(classification_report(y_test_aligned, preds2, zero_division=0))
        
        print(f"\nFused (MAJORITY) Classification Report:")
        print(classification_report(y_test_aligned, fused_results["majority"], zero_division=0))

    # Sample predictions
    print("\n" + "-"*80)
    print("SAMPLE FLOW DETECTIONS (First 10)")
    print("-"*80)
    
    n_samples = min(10, len(df))
    for i in range(n_samples):
        row = df.iloc[i].to_dict()
        print(f"\n[Flow {i+1}]")
        print(f"  {row.get('src_ip')} : {row.get('src_port')} -> {row.get('dst_ip')} : {row.get('dst_port')}")
        print(f"  Model1: {int(preds1[i])}, Model2: {int(preds2[i])}, Fused: {int(fused_results['majority'][i])}")
        if y_test is not None and i < len(y_test):
            print(f"  Ground Truth: {int(y_test[i])}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
