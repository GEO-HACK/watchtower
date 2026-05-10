from datetime import datetime
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flow timeout settings (seconds)
ACTIVE_TIMEOUT = 120   # Terminate flow after 2 minutes of activity
IDLE_TIMEOUT = 30      # Terminate flow after 30 seconds of inactivity


class Flow:
    """
    Represents a bidirectional network flow identified by a five-tuple.
    Accumulates packet metadata and computes flow-level features.
    """

    def __init__(self, five_tuple, timestamp):
        src_ip, dst_ip, src_port, dst_port, protocol = five_tuple

        # Canonical five-tuple (forward direction)
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol

        # Timestamps
        self.start_time = timestamp
        self.last_seen = timestamp

        # Forward packets (src -> dst)
        self.fwd_packets = []
        self.fwd_sizes = []
        self.fwd_timestamps = []
        self.fwd_flags = []

        # Backward packets (dst -> src)
        self.bwd_packets = []
        self.bwd_sizes = []
        self.bwd_timestamps = []
        self.bwd_flags = []

    def add_packet(self, metadata):
        """
        Adds a packet to the flow in the correct direction.
        """
        five_tuple = metadata['five_tuple']
        src_ip = five_tuple[0]
        timestamp = metadata['timestamp']
        size = metadata['size']
        flags = metadata['flags']

        self.last_seen = timestamp

        if src_ip == self.src_ip:
            # Forward direction
            self.fwd_packets.append(metadata)
            self.fwd_sizes.append(size)
            self.fwd_timestamps.append(timestamp)
            if flags is not None:
                self.fwd_flags.append(int(flags))
        else:
            # Backward direction
            self.bwd_packets.append(metadata)
            self.bwd_sizes.append(size)
            self.bwd_timestamps.append(timestamp)
            if flags is not None:
                self.bwd_flags.append(int(flags))

    def compute_features(self):
        """
        Computes flow-level features aligned with CICIDS2017 feature set.
        Returns a dict of features.
        """
        duration = self.last_seen - self.start_time

        fwd_count = len(self.fwd_packets)
        bwd_count = len(self.bwd_packets)
        total_count = fwd_count + bwd_count

        fwd_bytes = sum(self.fwd_sizes)
        bwd_bytes = sum(self.bwd_sizes)

        # Inter-arrival times
        all_timestamps = sorted(self.fwd_timestamps + self.bwd_timestamps)
        iats = np.diff(all_timestamps) if len(all_timestamps) > 1 else [0]

        fwd_iats = np.diff(self.fwd_timestamps) if len(
            self.fwd_timestamps) > 1 else [0]
        bwd_iats = np.diff(self.bwd_timestamps) if len(
            self.bwd_timestamps) > 1 else [0]
        
        # IAT totals
        fwd_iat_total = float(np.sum(fwd_iats))
        bwd_iat_total = float(np.sum(bwd_iats))

        # Packet length stats
        all_sizes = self.fwd_sizes + self.bwd_sizes
        fwd_sizes = self.fwd_sizes if self.fwd_sizes else [0]
        bwd_sizes = self.bwd_sizes if self.bwd_sizes else [0]

        # Minimum packet length across all packets
        min_packet_length = float(np.min(all_sizes)) if all_sizes else 0.0

        # Forward packets with actual data payload (size > 40 bytes = IP+TCP header)
        act_data_pkt_fwd = sum(1 for s in self.fwd_sizes if s > 40)

        # TCP flags
        syn_count  = sum(1 for f in self.fwd_flags if f & 0x02)
        ack_count  = sum(1 for f in self.fwd_flags if f & 0x10)
        fin_count  = sum(1 for f in self.fwd_flags if f & 0x01)
        rst_count  = sum(1 for f in self.fwd_flags if f & 0x04)
        psh_count  = sum(1 for f in self.fwd_flags if f & 0x08)
        urg_count  = sum(1 for f in self.fwd_flags if f & 0x20)
        fwd_psh    = sum(1 for f in self.fwd_flags if f & 0x08)

        # Connection fail ratio
        cfr = 1.0 if bwd_count == 0 else 0.0

        # Rates — avoid division by zero
        duration_safe = duration if duration > 0 else 1e-6
        flow_bytes_per_s   = (fwd_bytes + bwd_bytes) / duration_safe
        flow_packets_per_s = total_count / duration_safe
        fwd_packets_per_s  = fwd_count / duration_safe
        bwd_packets_per_s  = bwd_count / duration_safe

        # Active and idle times
        # Active = periods where packets are being sent
        # Idle = gaps between active periods
        active_times = []
        idle_times = []
        if len(all_timestamps) > 1:
            gaps = np.diff(sorted(all_timestamps))
            # Threshold: gaps > 1 second are idle periods
            IDLE_THRESHOLD = 1.0
            current_active = 0.0
            for gap in gaps:
                if gap < IDLE_THRESHOLD:
                    current_active += gap
                else:
                    if current_active > 0:
                        active_times.append(current_active)
                    idle_times.append(gap)
                    current_active = 0.0
            if current_active > 0:
                active_times.append(current_active)

        active_times = active_times if active_times else [0]
        idle_times   = idle_times   if idle_times   else [0]

        # Init window bytes — first packet size in each direction
        init_win_fwd = self.fwd_sizes[0] if self.fwd_sizes else 0
        init_win_bwd = self.bwd_sizes[0] if self.bwd_sizes else 0

        # Down/Up ratio
        down_up_ratio = bwd_bytes / fwd_bytes if fwd_bytes > 0 else 0

        features = {
            # Identity
            'src_ip':    self.src_ip,
            'dst_ip':    self.dst_ip,
            'src_port':  self.src_port,
            'dst_port':  self.dst_port,
            'protocol':  self.protocol,

            # Temporal
            'flow_duration':    duration,
            'flow_iat_mean':    float(np.mean(iats)),
            'flow_iat_std':     float(np.std(iats)),
            'flow_iat_max':     float(np.max(iats)),
            'flow_iat_min':     float(np.min(iats)),
            'fwd_iat_mean':     float(np.mean(fwd_iats)),
            'fwd_iat_std':      float(np.std(fwd_iats)),
            'fwd_iat_min':      float(np.min(fwd_iats)),
            'fwd_iat_max':      float(np.max(fwd_iats)),
            'bwd_iat_mean':     float(np.mean(bwd_iats)),
            'bwd_iat_std':      float(np.std(bwd_iats)),
            'bwd_iat_min':      float(np.min(bwd_iats)),
            'bwd_iat_max':      float(np.max(bwd_iats)),

            # Active/Idle times
            'active_mean':  float(np.mean(active_times)),
            'active_std':   float(np.std(active_times)),
            'active_max':   float(np.max(active_times)),
            'active_min':   float(np.min(active_times)),
            'idle_mean':    float(np.mean(idle_times)),
            'idle_std':     float(np.std(idle_times)),
            'idle_max':     float(np.max(idle_times)),
            'idle_min':     float(np.min(idle_times)),

            # Volumetric
            'total_fwd_packets':        fwd_count,
            'total_bwd_packets':        bwd_count,
            'total_packets':            total_count,
            'total_fwd_bytes':          fwd_bytes,
            'total_bwd_bytes':          bwd_bytes,
            'fwd_packet_length_mean':   float(np.mean(fwd_sizes)),
            'fwd_packet_length_std':    float(np.std(fwd_sizes)),
            'fwd_packet_length_max':    float(np.max(fwd_sizes)),
            'fwd_packet_length_min':    float(np.min(fwd_sizes)),
            'bwd_packet_length_mean':   float(np.mean(bwd_sizes)),
            'bwd_packet_length_std':    float(np.std(bwd_sizes)),
            'bwd_packet_length_max':    float(np.max(bwd_sizes)),
            'bwd_packet_length_min':    float(np.min(bwd_sizes)),
            'packet_length_mean':       float(np.mean(all_sizes)),
            'packet_length_std':        float(np.std(all_sizes)),
            'packet_length_variance':   float(np.var(all_sizes)),
            'max_packet_length':        float(np.max(all_sizes)),
            'avg_bwd_segment_size':     float(np.mean(bwd_sizes)),
            'avg_fwd_segment_size':     float(np.mean(fwd_sizes)),
            'average_packet_size':      float(np.mean(all_sizes)),

            # Rates
            'flow_bytes_per_s':     flow_bytes_per_s,
            'flow_packets_per_s':   flow_packets_per_s,
            'fwd_packets_per_s':    fwd_packets_per_s,
            'bwd_packets_per_s':    bwd_packets_per_s,

            # Connection behaviour
            'syn_flag_count':       syn_count,
            'ack_flag_count':       ack_count,
            'fin_flag_count':       fin_count,
            'rst_flag_count':       rst_count,
            'psh_flag_count':       psh_count,
            'urg_flag_count':       urg_count,
            'fwd_psh_flags':        fwd_psh,
            'connection_fail_ratio': cfr,
            'init_win_bytes_forward':  init_win_fwd,
            'init_win_bytes_backward': init_win_bwd,

            # Relational
            'fwd_bwd_packet_ratio': fwd_count / bwd_count if bwd_count > 0 else fwd_count,
            'fwd_bwd_bytes_ratio':  fwd_bytes / bwd_bytes if bwd_bytes > 0 else fwd_bytes,
            'down_up_ratio':        down_up_ratio,

            # Subflow features (single-subflow model — mirrors main flow counts)
            'subflow_fwd_packets':  fwd_count,
            'subflow_fwd_bytes':    fwd_bytes,
            'subflow_bwd_packets':  bwd_count,
            'subflow_bwd_bytes':    bwd_bytes,

            # Additional features
            'fwd_iat_total':        fwd_iat_total,
            'bwd_iat_total':        bwd_iat_total,
            'min_packet_length':    min_packet_length,
            'act_data_pkt_fwd':     act_data_pkt_fwd,
        }

        # Add CICIDS2017-compatible aliases for signature matching
        features['total_fwd_bytes'] = fwd_bytes
        features['total_bwd_bytes'] = bwd_bytes
        features['total_length_of_fwd_packets'] = fwd_bytes  # For SSH-Patator/Bot
        features['total_length_of_bwd_packets'] = bwd_bytes
            
        # Ensure all flag counts exist (default to 0 if no TCP)
        features['fin_flag_count'] = fin_count
        features['ack_flag_count'] = ack_count
        features['rst_flag_count'] = rst_count
        features['psh_flag_count'] = psh_count
        features['urg_flag_count'] = urg_count

        return features

    def is_expired(self, current_time):
        """
        Returns True if the flow has exceeded active or idle timeout.
        """
        if current_time - self.start_time > ACTIVE_TIMEOUT:
            return True
        if current_time - self.last_seen > IDLE_TIMEOUT:
            return True
        return False


class FlowAggregator:
    """
    Manages active flows and produces completed flow records.
    """

    def __init__(self, flow_callback=None):
        self.active_flows = {}
        self.completed_flows = []
        self.flow_callback = flow_callback  # Called when a flow completes

    def get_flow_key(self, five_tuple):
        """
        Returns a canonical flow key that is direction-independent.
        (A->B and B->A belong to the same flow)
        """
        src_ip, dst_ip, src_port, dst_port, protocol = five_tuple
        # Sort to make key direction-independent
        forward = (src_ip, dst_ip, src_port, dst_port, protocol)
        backward = (dst_ip, src_ip, dst_port, src_port, protocol)
        return min(forward, backward)

    def process_packet(self, metadata):
        """
        Processes a packet metadata dict and adds it to the correct flow.
        Expires old flows as new packets arrive.
        """
        if metadata is None:
            return

        five_tuple = metadata['five_tuple']
        timestamp = metadata['timestamp']
        flow_key = self.get_flow_key(five_tuple)

        # Check for expired flows
        self._expire_flows(timestamp)

        # Add to existing flow or create new one
        if flow_key in self.active_flows:
            self.active_flows[flow_key].add_packet(metadata)
        else:
            flow = Flow(five_tuple, timestamp)
            flow.add_packet(metadata)
            self.active_flows[flow_key] = flow

    def _expire_flows(self, current_time):
        """
        Checks all active flows and completes any that have timed out.
        """
        expired_keys = [
            key for key, flow in self.active_flows.items()
            if flow.is_expired(current_time)
        ]
        for key in expired_keys:
            self._complete_flow(key)

    def _complete_flow(self, flow_key):
        """
        Moves a flow from active to completed and computes its features.
        """
        flow = self.active_flows.pop(flow_key)
        features = flow.compute_features()
        self.completed_flows.append(features)
        if self.flow_callback:
            self.flow_callback(features)

    def flush(self):
        """
        Forces completion of all remaining active flows.
        Call this at the end of a capture session.
        """
        for key in list(self.active_flows.keys()):
            self._complete_flow(key)
        logger.info(f'Flushed {len(self.completed_flows)} completed flows')

    def get_completed_flows(self):
        return self.completed_flows


if __name__ == '__main__':
    # Quick test using packet_capture module
    import sys
    from packet_capture import capture_from_pcap, get_packet_metadata

    if len(sys.argv) < 2:
        print('Usage: python3 flow_aggregator.py <path_to_pcap>')
        sys.exit(1)

    pcap_path = sys.argv[1]
    aggregator = FlowAggregator()

    def handle_packet(packet):
        metadata = get_packet_metadata(packet)
        aggregator.process_packet(metadata)

    capture_from_pcap(pcap_path, handle_packet)
    aggregator.flush()

    flows = aggregator.get_completed_flows()
    print(f'\nTotal flows generated: {len(flows)}')
    if flows:
        print('\nSample flow features:')
        for k, v in list(flows[0].items()):
            print(f'  {k}: {v}')
