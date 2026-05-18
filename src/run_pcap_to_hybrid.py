#!/usr/bin/env python3
"""Aggregate flows from a PCAP and run the hybrid detector in live-CSV mode.

Usage:
  python src/run_pcap_to_hybrid.py <pcap_path> [--out_csv path] [--max-packets N]

This script reads packets from a PCAP using the existing packet_capture utilities,
builds flows with FlowAggregator, writes a feature CSV compatible with
`hybrid_detector.py --live_csv`, and calls the hybrid detector.
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Ensure imports use package-relative paths in this repo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocessing.packet_capture import capture_from_pcap, get_packet_metadata
from preprocessing.flow_aggregator import FlowAggregator


def aggregate_pcap_to_flows(pcap_path, max_packets=None, skip_packets=0):
    aggregator = FlowAggregator()

    def handler(pkt):
        meta = get_packet_metadata(pkt)
        if meta is None:
            return
        aggregator.process_packet(meta)

    capture_from_pcap(pcap_path, handler, max_packets=max_packets, skip_packets=skip_packets)
    aggregator.flush()
    flows = aggregator.get_completed_flows()
    return flows


def flows_to_csv(flows, out_csv):
    if not flows:
        raise ValueError('No flows generated from PCAP')
    df = pd.DataFrame(flows)
    # Ensure timestamp columns are present for hybrid_detector preview
    if 'start_time' not in df.columns and 'flow_duration' in df.columns:
        df['start_time'] = 0
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def run_hybrid_detector(csv_path, strategy='majority', top=10):
    # Invoke hybrid_detector as a subprocess to preserve CLI behavior
    import subprocess
    cmd = [sys.executable, os.path.join('src', 'hybrid_detector.py'), '--live_csv', csv_path, '--strategy', strategy, '--top', str(top)]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description='Aggregate PCAP->flows then run hybrid detector')
    p.add_argument('pcap', help='Path to input PCAP file')
    p.add_argument('--out_csv', default=os.path.join('src', 'data', 'flows_from_pcap.csv'), help='Output CSV path for aggregated flows')
    p.add_argument('--max-packets', type=int, default=None, help='Maximum packets to process from PCAP')
    p.add_argument('--skip-packets', type=int, default=0, help='Packets to skip at start of PCAP')
    p.add_argument('--strategy', default='majority', help='Fusion strategy for hybrid detector')
    p.add_argument('--top', type=int, default=10, help='Number of sample rows to print')
    args = p.parse_args()

    pcap_path = args.pcap
    if not os.path.exists(pcap_path):
        print('PCAP not found:', pcap_path)
        sys.exit(1)

    print('Aggregating flows from PCAP...')
    flows = aggregate_pcap_to_flows(pcap_path, max_packets=args.max_packets, skip_packets=args.skip_packets)
    print(f'Generated {len(flows)} flows')

    out_csv = args.out_csv
    print('Writing flows to CSV:', out_csv)
    flows_to_csv(flows, out_csv)

    print('Invoking hybrid detector on generated CSV')
    run_hybrid_detector(out_csv, strategy=args.strategy, top=args.top)


if __name__ == '__main__':
    main()
