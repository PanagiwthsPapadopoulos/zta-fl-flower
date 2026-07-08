import logging
import json
import os


class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "node": getattr(record, "node", "UNKNOWN"),
            "round": getattr(record, "round", 0),
            "message": record.getMessage()
        })


class FederatedContextFilter(logging.Filter):
    def __init__(self, node_id):
        super().__init__()
        self.node_id = node_id

    def filter(self, record):
        record.node = self.node_id
        if not hasattr(record, "round"):
            record.round = 0
        return True


def setup_logger(node_id: str):
    log_base_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_base_dir, exist_ok=True)
    log_file = os.path.join(log_base_dir, f"{node_id.replace('[', '').replace(']', '').replace(' ', '_').lower()}.jsonl")

    logger = logging.getLogger(node_id)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if not any(isinstance(f, FederatedContextFilter) for f in logger.filters):
        logger.addFilter(FederatedContextFilter(node_id))
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(JSONFormatter())
        logger.addHandler(stream_handler)
    return logger