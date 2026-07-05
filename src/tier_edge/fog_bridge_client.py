import os
import time
import socket
import logging
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

from src.shared.network.ipc import send_msg, recv_msg


class FogBridgeClient:
    """A network client component for managing IPC socket flows and coordinating training rounds with the regional hub."""
    def __init__(self, logger: logging.Logger, log_prefix: str, ipc_port: int, socket_timeout: float = 600.0):
        """Initializes the Fog Bridge Client for IPC communication over the designated socket."""
        self.logger = logger
        self.log_prefix = log_prefix
        self.ipc_port = ipc_port
        self.socket_timeout = socket_timeout
        self.target_host = os.getenv("FOG_SERVER_HOST", "127.0.0.1")

    def _connect_with_retries(self, current_round: int) -> Optional[socket.socket]:
        """Attempts to establish a socket connection with the Fog Server, actively retrying on failure."""
        self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Attempting to connect to Fog Server at {self.target_host}:{self.ipc_port}...", extra={"round": current_round})
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.socket_timeout)
                sock.connect((self.target_host, self.ipc_port))
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Socket connection successfully established.", extra={"round": current_round})
                return sock
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                sock.close()
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Server unavailable. Retrying in 0.5s...", extra={"round": current_round})
                time.sleep(0.5)
            except Exception as e:
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Non-fatal socket initialization error: {e}", extra={"round": current_round})
                return None

    def execute_round(self, current_round: int) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Executes a single federated learning round by dispatching a START signal and awaiting aggregated weights."""
        try:
            sock = self._connect_with_retries(current_round)
            if not sock:
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Aborting round. Could not establish socket.", extra={"round": current_round})
                return [], 0, {"status": "crashed"}
            
            self.logger.info(f"{self.log_prefix} [IPC CLIENT] Connected! Sending START signal for round {current_round}.", extra={"round": current_round})
            send_msg(sock, {"cmd": "START", "round": current_round})
            
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] START dispatched. Blocking until Fog Server returns aggregated weights...", extra={"round": current_round})
            aggregated_ndarrays = recv_msg(sock)
            
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Data received. Closing IPC socket.", extra={"round": current_round})
            sock.close()
            
            if not aggregated_ndarrays:
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Empty weight array received. Returning neutralized status.", extra={"round": current_round})
                return [], 0, {"status": "neutralized_attack", "node_name": self.log_prefix}
                
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Successfully received {len(aggregated_ndarrays)} tensor layers.", extra={"round": current_round})
            return aggregated_ndarrays, 1, {"node_name": self.log_prefix}
            
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Non-fatal runtime exception during execute_round: {e}", extra={"round": current_round})
            return [], 0, {"status": "crashed"}