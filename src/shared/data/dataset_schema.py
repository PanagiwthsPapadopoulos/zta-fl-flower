"""Class definitions and the numeric feature schema of the full source datasets.

This module has no ML dependencies, so the download/cleaning command can use the
same class definitions as the training loader without importing Torch.
"""

EDGE_FULL_LABELS = {
    "Normal": 0, "DDoS_TCP": 1, "DDoS_UDP": 2, "DDoS_HTTP": 3,
    "MITM": 4, "Fingerprinting": 5, "Password": 6,
    "Port_Scanning": 7, "Ransomware": 8, "Backdoor": 9,
    "Vulnerability_scanner": 10, "Uploading": 11, "SQL_injection": 12,
    "XSS": 13, "DDoS_ICMP": 14,
}

CIC_IDS2017_LABELS = {
    "BENIGN": 0, "DoS Hulk": 1, "DDoS": 2, "PortScan": 3,
    "Bot": 4, "Web Attack – Brute Force": 5, "Web Attack – XSS": 6,
    "Web Attack – Sql Injection": 7, "Infiltration": 8, "Heartbleed": 9,
    "FTP-Patator": 10, "SSH-Patator": 11, "DoS slowloris": 12,
    "DoS Slowhttptest": 13, "DoS GoldenEye": 14,
}

UNSW_NB15_LABELS = {
    "Normal": 0, "Generic": 1, "Exploits": 2, "Fuzzers": 3,
    "DoS": 4, "Reconnaissance": 5, "Backdoor": 6, "Analysis": 7,
    "Shellcode": 8, "Worms": 9,
}

EDGE_SOURCE_COLUMNS = (
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4",
    "arp.opcode", "arp.hw.size", "arp.src.proto_ipv4", "icmp.checksum",
    "icmp.seq_le", "icmp.transmit_timestamp", "icmp.unused", "http.file_data",
    "http.content_length", "http.request.uri.query", "http.request.method",
    "http.referer", "http.request.full_uri", "http.request.version",
    "http.response", "http.tls_port", "tcp.ack", "tcp.ack_raw", "tcp.checksum",
    "tcp.connection.fin", "tcp.connection.rst", "tcp.connection.syn",
    "tcp.connection.synack", "tcp.dstport", "tcp.flags", "tcp.flags.ack",
    "tcp.len", "tcp.options", "tcp.payload", "tcp.seq", "tcp.srcport",
    "udp.port", "udp.stream", "udp.time_delta", "dns.qry.name",
    "dns.qry.name.len", "dns.qry.qu", "dns.qry.type", "dns.retransmission",
    "dns.retransmit_request", "dns.retransmit_request_in", "mqtt.conack.flags",
    "mqtt.conflag.cleansess", "mqtt.conflags", "mqtt.hdrflags", "mqtt.len",
    "mqtt.msg_decoded_as", "mqtt.msg", "mqtt.msgtype", "mqtt.proto_len",
    "mqtt.protoname", "mqtt.topic", "mqtt.topic_len", "mqtt.ver", "mbtcp.len",
    "mbtcp.trans_id", "mbtcp.unit_id", "Attack_label", "Attack_type",
)

EDGE_EXCLUDED_COLUMNS = frozenset({
    "frame.time", "ip.src_host", "ip.dst_host", "arp.dst.proto_ipv4",
    "arp.src.proto_ipv4", "http.file_data", "http.request.uri.query",
    "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "tcp.options", "tcp.payload", "dns.qry.name",
    "mqtt.conack.flags", "mqtt.msg_decoded_as", "mqtt.msg", "mqtt.protoname",
    "mqtt.topic", "Attack_label", "Attack_type",
})
EDGE_FULL_FEATURES = tuple(c for c in EDGE_SOURCE_COLUMNS if c not in EDGE_EXCLUDED_COLUMNS)

UNSW_SOURCE_COLUMNS = (
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime", "Sintpkt",
    "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "attack_cat", "Label",
)
UNSW_EXCLUDED_COLUMNS = frozenset({"srcip", "dstip", "proto", "state", "service", "attack_cat", "Label"})

CIC_SOURCE_COLUMNS = (
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
    "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean",
    "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
    "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
)
