from scapy.all import sniff, rdpcap
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


def capture_from_pcap(pcap_path, packet_handler):
    """
    Reads packets from a PCAP file and passes each to packet_handler.
    Used for testing without live traffic.
    
    Args:
        pcap_path: Path to the .pcap file
        packet_handler: Callback function to process each packet
    """
    logger.info(f'Reading packets from PCAP: {pcap_path}')
    packets = rdpcap(pcap_path)
    logger.info(f'Loaded {len(packets)} packets')
    for packet in packets:
        packet_handler(packet)


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
