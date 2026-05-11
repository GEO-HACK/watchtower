from scapy.all import sniff, PcapReader, PcapWriter
import os
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_five_tuple(packet):
    """
    Extracts the five-tuple from a packet.
    Returns (src_ip, dst_ip, src_port, dst_port, protocol) or None
    if the packet is not TCP/UDP.
    """
    try:
        if packet.haslayer('IP'):
            src_ip = packet['IP'].src
            dst_ip = packet['IP'].dst
            protocol = packet['IP'].proto

            src_port = None
            dst_port = None

            if packet.haslayer('TCP'):
                src_port = packet['TCP'].sport
                dst_port = packet['TCP'].dport
            elif packet.haslayer('UDP'):
                src_port = packet['UDP'].sport
                dst_port = packet['UDP'].dport
            else:
                # Not TCP or UDP — skip
                return None

            return (src_ip, dst_ip, src_port, dst_port, protocol)
    except Exception as e:
        logger.warning(f'Error extracting five-tuple: {e}')
    return None


def get_packet_metadata(packet):
    """
    Extracts metadata from a packet needed for flow feature computation.
    Returns a dict with timestamp, size, flags, and five-tuple.
    """
    five_tuple = extract_five_tuple(packet)
    if five_tuple is None:
        return None

    metadata = {
        'five_tuple': five_tuple,
        'timestamp': float(packet.time),
        'size': len(packet),
        'flags': None,
        'direction': 'forward'
    }

    # Extract TCP flags if present
    if packet.haslayer('TCP'):
        metadata['flags'] = packet['TCP'].flags

    return metadata


def capture_live(interface, packet_handler, count=0, timeout=None):
    """
    Captures packets live from a network interface.
    
    Args:
        interface: Network interface name (e.g. 'eth0')
        packet_handler: Callback function to process each packet
        count: Number of packets to capture (0 = infinite)
        timeout: Stop after this many seconds (None = no timeout)
    """
    logger.info(f'Starting live capture on interface: {interface}')
    sniff(
        iface=interface,
        prn=packet_handler,
        count=count,
        timeout=timeout,
        store=False  # Don't store packets in memory
    )


def capture_from_pcap(pcap_path, packet_handler, max_packets=None, skip_packets=0, progress_interval=1000):
    """
    Reads packets from a PCAP file and passes each to packet_handler.
    Used for testing without live traffic.
    
    Args:
        pcap_path: Path to the .pcap file
        packet_handler: Callback function to process each packet
        max_packets: Maximum packets to process after skipping (None = all)
        skip_packets: Number of initial packets to skip before processing
    """
    logger.info(f'Reading packets from PCAP: {pcap_path}')
    processed = 0
    skipped = 0

    with PcapReader(pcap_path) as reader:
        for packet in reader:
            if skipped < skip_packets:
                skipped += 1
                # Log skipped progress occasionally for long skips
                if skipped % progress_interval == 0:
                    logger.info('Skipped %d packets so far...', skipped)
                continue

            packet_handler(packet)
            processed += 1

            # Periodic progress reporting while processing
            if progress_interval and processed % progress_interval == 0:
                logger.info('Processed %d packets (skipped=%d)...', processed, skipped)

            if max_packets is not None and processed >= max_packets:
                logger.info('Reached max_packets=%d, stopping read.', max_packets)
                break

    logger.info('PCAP processing complete: skipped=%d processed=%d', skipped, processed)
    return processed


def canonical_flow_key(five_tuple):
    """Return a direction-independent canonical flow key for a five-tuple."""
    src_ip, dst_ip, src_port, dst_port, protocol = five_tuple
    forward = (src_ip, dst_ip, src_port, dst_port, protocol)
    backward = (dst_ip, src_ip, dst_port, src_port, protocol)
    return min(forward, backward)


def save_packets_for_flows(pcap_path, flows, indices, out_dir, pad_seconds=0.0, progress_interval=1000):
    """Extract packets matching selected flows and write per-flow PCAPs.

    Args:
        pcap_path: input PCAP file
        flows: list-like of flow dicts (must include src_ip,dst_ip,src_port,dst_port,protocol,start_time,end_time)
        indices: iterable of integer indices (indexes into flows) to export
        out_dir: directory to write per-flow PCAP files
        pad_seconds: float seconds to pad start/end times to include nearby packets
    Returns:
        dict mapping flow_index -> number of packets written
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prepare flow matchers
    matchers = {}
    for i in indices:
        if i < 0 or i >= len(flows):
            continue
        row = flows[i]
        key = canonical_flow_key((row.get('src_ip'), row.get('dst_ip'), row.get('src_port'), row.get('dst_port'), row.get('protocol')))
        start = float(row.get('start_time', row.get('flow_start', 0))) - pad_seconds
        end = float(row.get('end_time', row.get('last_seen', row.get('start_time', 0)))) + pad_seconds
        fname = os.path.join(out_dir, f'flow_{i}_{row.get("src_ip")}_{row.get("dst_ip")}_{int(start)}-{int(end)}.pcap')
        matchers[i] = {
            'key': key,
            'start': start,
            'end': end,
            'path': fname,
            'writer': PcapWriter(fname, append=False, sync=True),
            'count': 0,
        }

    processed = 0
    with PcapReader(pcap_path) as reader:
        for packet in reader:
            processed += 1
            if progress_interval and processed % progress_interval == 0:
                logger.info('Export scan processed %d packets...', processed)

            meta = get_packet_metadata(packet)
            if meta is None:
                continue
            pkt_key = canonical_flow_key(meta['five_tuple'])
            t = float(meta['timestamp'])

            for idx, m in matchers.items():
                if pkt_key == m['key'] and t >= m['start'] and t <= m['end']:
                    m['writer'].write(packet)
                    m['count'] += 1

    # Close writers and report
    results = {}
    for idx, m in matchers.items():
        try:
            m['writer'].close()
        except Exception:
            pass
        results[idx] = m['count']
        logger.info('Flow %d -> wrote %d packets to %s', idx, m['count'], m['path'])

    logger.info('Export complete: scanned %d packets, exported %d flows', processed, len(results))
    return results


if __name__ == '__main__':
    # Quick test — print five-tuple of first 10 packets from a PCAP
    # Replace with actual PCAP path when testing
    import sys

    if len(sys.argv) < 2:
        print('Usage: python3 packet_capture.py <path_to_pcap>')
        sys.exit(1)

    pcap_path = sys.argv[1]
    count = [0]

    def test_handler(packet):
        metadata = get_packet_metadata(packet)
        if metadata and count[0] < 10:
            print(f"Packet {count[0]+1}: {metadata['five_tuple']} "
                  f"size={metadata['size']} "
                  f"time={metadata['timestamp']}")
            count[0] += 1

    capture_from_pcap(pcap_path, test_handler)
