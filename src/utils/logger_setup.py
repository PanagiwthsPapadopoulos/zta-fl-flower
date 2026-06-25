import logging
import json
import os

class JSONFormatter(logging.Formatter):
    """
    Extends format mapping directives producing explicit dictionaries tailored for parsing operations.
    """
    def format(self, record):
        # The filter guarantees that 'node' and 'round' fields are present on the record natively
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "node": getattr(record, "node", "UNKNOWN"),
            "round": getattr(record, "round", 0),
            "message": record.getMessage()
        }
        return json.dumps(log_record)

class FederatedContextFilter(logging.Filter):
    """
    A filter attached directly to the logger that guarantees node identity 
    and round parameters are safely injected into every LogRecord without 
    breaking standard Logger object compatibility.
    """
    def __init__(self, node_id):
        super().__init__()
        self.node_id = node_id

    def filter(self, record):
        # Inject standard guaranteed variables
        record.node = self.node_id
        
        # Inject default round if omitted, while preserving any dynamic extras
        if not hasattr(record, "round"):
            record.round = 0
            
        return True

def setup_logger(node_id: str):
    """
    Constructs an explicit pipeline routing analytical logs specifically tracking system milestones toward local stores
    while broadcasting surface alerts toward general visualization streams.
    """
    # 1. Use a relative path based on the current working directory
    current_dir = os.getcwd()
    log_base_dir = os.path.join(current_dir, "logs", "nodes")
    os.makedirs(log_base_dir, exist_ok=True)
    
    safe_id = node_id.replace("[", "").replace("]", "").replace(" ", "_").lower()
    
    # 2. Bind the filename explicitly to the new dynamic base directory
    log_file = os.path.join(log_base_dir, f"{safe_id}.jsonl")

    # Diagnostic print to guarantee visibility of the exact save location
    print(f"[LOGGER SETUP] Booting {node_id}. Writing JSON logs to: {log_file}")

    logger = logging.getLogger(node_id)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    # Apply the context filter
    if not any(isinstance(f, FederatedContextFilter) for f in logger.filters):
        logger.addFilter(FederatedContextFilter(node_id))
    
    if not logger.handlers:
        # Pass the specific node_id to the formatter if your init requires it, 
        # or leave blank if using the Filter approach from the last step.
        json_formatter = JSONFormatter()

        # Connects exhaustive tracking toward diagnostic output sources.
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

        # Filters operational metrics away from primary environment monitoring streams.
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(json_formatter)
        logger.addHandler(stream_handler)

    return logger