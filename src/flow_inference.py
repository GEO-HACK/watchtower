import os
import sys
import argparse
import joblib
import logging
import numpy as np
import pandas as pd

from preprocessing.packet_capture import capture_from_pcap, get_packet_metadata
from preprocessing.flow_aggregator import FlowAggregator
from diagnostics.schema_checker import inspect_model_file, align_columns

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
    # First align columns if model exposes feature names
    if df.empty:
        return df
    aligned = align_columns(df, model_info, fill_value=0)

    # If a preprocessing pipeline is provided, apply it
    if pipeline_path and os.path.exists(pipeline_path):
        try:
            pipe = joblib.load(pipeline_path)
            try:
                transformed = pipe.transform(aligned)
                return transformed
            except Exception:
                # If pipeline expects numpy input
                return pipe.transform(aligned.values)
        except Exception as e:
            logger.warning(f'Could not load pipeline: {e}')

    # Coerce aligned dataframe to numeric features only (drop/convert identity strings)
    if isinstance(aligned, pd.DataFrame):
        for col in aligned.columns:
            aligned[col] = pd.to_numeric(aligned[col], errors='coerce')
        aligned = aligned.fillna(0)
        return aligned.values

    return aligned


def main():
    p = argparse.ArgumentParser(description='Run flow-based inference from a PCAP')
    p.add_argument('--pcap', default=os.path.join('src','data','Friday-WorkingHours.pcap'), help='Path to input PCAP file')
    p.add_argument('--model', required=True, help='Path to saved model (joblib/pkl)')
    p.add_argument('--model2', default=os.path.join('src','models','xgboost_model.pkl'), help='Optional second model for comparison')
    p.add_argument('--pipeline', default=None, help='Optional preprocessing pipeline path')
    p.add_argument('--top', type=int, default=5, help='Show top-N flows and predictions')
    p.add_argument('--max-packets', type=int, default=1000, help='Process only this many packets from the PCAP for faster testing (default: 1000)')
    p.add_argument('--skip-packets', type=int, default=0, help='Skip this many initial packets before processing')
    p.add_argument('--show-disagreements', action='store_true', help='Print sample flows where models disagree')
    args = p.parse_args()

    if not os.path.exists(args.pcap):
        logger.error('PCAP not found: %s', args.pcap)
        sys.exit(1)

    if not os.path.exists(args.model):
        logger.error('Model not found: %s', args.model)
        sys.exit(1)

    if args.max_packets is not None and args.max_packets <= 0:
        logger.error('--max-packets must be > 0 when provided')
        sys.exit(1)

    if args.skip_packets < 0:
        logger.error('--skip-packets must be >= 0')
        sys.exit(1)

    logger.info(
        'Converting PCAP -> flows (skip=%d, max=%s)...',
        args.skip_packets,
        str(args.max_packets) if args.max_packets is not None else 'all',
    )
    df = pcap_to_flow_features(
        args.pcap,
        max_packets=args.max_packets,
        skip_packets=args.skip_packets,
    )
    logger.info('Flows generated: %d', len(df))

    model_info = inspect_model_file(args.model)
    logger.info('Model expects %s features', model_info.get('n_features_in_'))

    X = align_and_transform(df, model_info, pipeline_path=args.pipeline)
    if isinstance(X, pd.DataFrame):
        X = X.values

    if X is None or getattr(X, 'size', 0) == 0:
        logger.error('No feature data available for inference')
        sys.exit(1)

    model, raw = load_maybe_dict_model(args.model)

    preds, proba = predict_with_model(model, X)

    # Optionally load and run a second model for comparison
    preds2 = None
    proba2 = None
    model2_path = args.model2
    if model2_path and os.path.exists(model2_path):
        try:
            model_b, raw_b = load_maybe_dict_model(model2_path)
            preds2, proba2 = predict_with_model(model_b, X)
        except Exception as e:
            logger.warning('Could not run second model: %s', e)
    else:
        logger.info('No second model found at %s; skipping comparison', model2_path)

    # Present basic output
    unique, counts = np.unique(preds, return_counts=True)
    logger.info('Model1 prediction distribution: %s', dict(zip(map(int, unique), map(int, counts))))

    if preds2 is not None:
        unique2, counts2 = np.unique(preds2, return_counts=True)
        logger.info('Model2 prediction distribution: %s', dict(zip(map(int, unique2), map(int, counts2))))

        # Agreement summary
        agree = np.mean(preds == preds2)
        total = len(preds)
        disagree_idx = np.where(preds != preds2)[0]
        logger.info('Models agreement: %.3f (%d/%d)', agree, int(np.sum(preds == preds2)), total)

        # Build simple contingency table for labels
        cont = {}
        for a, b in zip(preds, preds2):
            cont[(int(a), int(b))] = cont.get((int(a), int(b)), 0) + 1
        logger.info('Contingency (model1_label -> model2_label): %s', cont)

    # Print top-N sample flows with predictions
    n = min(args.top, len(df))
    if n > 0:
        print('\nTop %d flow predictions:' % n)
        for i in range(n):
            row = df.iloc[i].to_dict()
            print('---')
            print('Flow:', row.get('src_ip'), '->', row.get('dst_ip'), 'ports:', row.get('src_port'), '->', row.get('dst_port'))
            print('Model1 Pred:', int(preds[i]))
            if preds2 is not None:
                print('Model2 Pred:', int(preds2[i]))

    # Optionally show sample disagreements
    if args.show_disagreements and preds2 is not None and len(disagree_idx) > 0:
        m = min(len(disagree_idx), args.top)
        print(f"\nShowing {m} sample disagreements (model1 != model2):")
        for j in range(m):
            i = disagree_idx[j]
            row = df.iloc[i].to_dict()
            print('---')
            print('Flow:', row.get('src_ip'), '->', row.get('dst_ip'), 'ports:', row.get('src_port'), '->', row.get('dst_port'))
            print('Model1:', int(preds[i]), 'Model2:', int(preds2[i]))


if __name__ == '__main__':
    main()
