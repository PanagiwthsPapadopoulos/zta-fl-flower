import logging
import json
import os

class JSONFormatter(logging.Formatter):
    """
    Extends format mapping directives producing explicit dictionaries tailored for parsing operations.
    """
    def __init__(self, node_id):
        super().__init__()
        self.node_id = node_id

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "node": self.node_id,
            "round": getattr(record, 'round', None),
            "message": record.getMessage()
        }
        return json.dumps(log_record)

def setup_logger(node_id: str):
    """
    Constructs an explicit pipeline routing analytical logs specifically tracking system milestones toward local stores
    while broadcasting surface alerts toward general visualization streams.
    """
    os.makedirs("logs/nodes", exist_ok=True)
    
    safe_id = node_id.replace("[", "").replace("]", "").replace(" ", "_").lower()
    log_file = f"logs/nodes/{safe_id}.jsonl"

    logger = logging.getLogger(node_id)
    
    # Authorizes total propagation access toward backend parsing operations.
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        json_formatter = JSONFormatter(node_id)

        # Connects exhaustive tracking toward diagnostic output sources.
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

        # Filters operational metrics away from primary environment monitoring streams.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(json_formatter)
        logger.addHandler(stream_handler)

    return logger