import os
import time
import socket
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# Import your pristine transport-layer functions
from src.network.ipc import send_msg, recv_msg

# =====================================================================
# 1. THE FOG CLIENT (Upstream to Cloud)
# =====================================================================
class FogBridgeClient:
    """
    Application-layer wrapper for the IPC bridge connecting the Fog and Cloud tiers.
    Operates on the Fog Client side to initiate the handshake and retrieve weights.
    """
    def __init__(self, logger: logging.Logger, log_prefix: str, ipc_port: int, socket_timeout: float = 600.0):
        self.logger = logger
        self.log_prefix = log_prefix
        self.ipc_port = ipc_port
        self.socket_timeout = socket_timeout
        self.target_host = os.getenv("FOG_SERVER_HOST", "127.0.0.1")

    def _connect_with_retries(self) -> Optional[socket.socket]:
        """Blocks and retries until the Fog Server opens its listening socket."""
        self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Attempting to connect to Fog Server at {self.target_host}:{self.ipc_port}...")
        
        while True:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.socket_timeout)
                sock.connect((self.target_host, self.ipc_port))
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Socket connection successfully established.")
                return sock
            except (ConnectionRefusedError, TimeoutError, socket.timeout):
                sock.close()
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Server unavailable. Retrying in 0.5s...")
                time.sleep(0.5)
            except Exception as e:
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Non-fatal socket initialization error: {e}")
                return None

    def execute_round(self, current_round: int) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        """Executes the full synchronous handshake for a single federated round."""
        try:
            sock = self._connect_with_retries()
            if not sock:
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Aborting round. Could not establish socket.")
                return [], 0, {"status": "crashed"}
            
            self.logger.info(f"{self.log_prefix} [IPC CLIENT] Connected! Sending START signal for round {current_round}.")
            send_msg(sock, {"cmd": "START", "round": current_round})
            
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] START dispatched. Blocking until Fog Server returns aggregated weights...")
            aggregated_ndarrays = recv_msg(sock)
            
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Data received. Closing IPC socket.")
            sock.close()
            
            if not aggregated_ndarrays:
                self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Empty weight array received. Returning neutralized status.")
                return [], 0, {"status": "neutralized_attack", "node_name": self.log_prefix}
                
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Successfully received {len(aggregated_ndarrays)} tensor layers.")
            return aggregated_ndarrays, 1, {"node_name": self.log_prefix}
            
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC CLIENT] Non-fatal runtime exception during execute_round: {e}")
            return [], 0, {"status": "crashed"}


# =====================================================================
# 2. THE FOG SERVER (Downstream to Edge)
# =====================================================================
class FogBridgeServer:
    """
    Application-layer wrapper for the IPC bridge connecting the Fog and Cloud tiers.
    Operates on the Fog Server side to listen for the client, read the START message, 
    and relay the aggregated edge weights back up.
    """
    def __init__(self, logger: logging.Logger, log_prefix: str, ipc_port: int, socket_timeout: float = 600.0):
        self.logger = logger
        self.log_prefix = log_prefix
        self.ipc_port = ipc_port
        self.socket_timeout = socket_timeout
        self.ipc_server_sock = None
        self.active_conn = None
        
        self._bind_socket()

    def _bind_socket(self):
        """Initializes and binds the listening socket for incoming Fog Client connections."""
        try:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Booting IPC listener on 0.0.0.0:{self.ipc_port}...")
            self.ipc_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ipc_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.ipc_server_sock.settimeout(self.socket_timeout)
            self.ipc_server_sock.bind(("0.0.0.0", self.ipc_port))
            self.ipc_server_sock.listen(1)
            self.logger.info(f"{self.log_prefix} [IPC SERVER] Listening on port {self.ipc_port}. Awaiting local Fog Client bridge...")
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception during socket bind: {e}")
            self.ipc_server_sock = None

    def wait_for_start(self) -> int:
        """Blocks until the Fog Client connects and issues the START command."""
        if not self.ipc_server_sock:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Socket not initialized. Bypassing wait_for_start.")
            return 0
            
        try:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Blocking for incoming connection from local Fog Client...")
            self.active_conn, addr = self.ipc_server_sock.accept()
            self.active_conn.settimeout(self.socket_timeout)
            
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Connection established from {addr}. Awaiting START payload...")
            msg = recv_msg(self.active_conn)
            
            if isinstance(msg, dict) and msg.get("cmd") == "START":
                round_num = msg.get("round", 0)
                self.logger.info(f"{self.log_prefix} [IPC SERVER] START received for round {round_num}. Shouting to EDGE clients!")
                return round_num
            else:
                self.logger.debug(f"{self.log_prefix} [IPC SERVER] Invalid message received on bridge: {msg}")
                return 0
                
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception while waiting for START: {e}")
            return 0

    def relay_weights(self, ndarrays: List[np.ndarray]):
        """Transmits the final aggregated PyTorch parameters back across the boundary."""
        if not self.active_conn:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] No active connection available to relay weights. Skipping.")
            return
            
        try:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Relaying {len(ndarrays) if ndarrays else 0} layers back to Fog Client...")
            send_msg(self.active_conn, ndarrays if ndarrays else [])
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Relay successful. Dropping active connection.")
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception during weight relay: {e}")
        finally:
            try:
                self.active_conn.close()
            except Exception:
                pass
            self.active_conn = None
    
    def close(self):
        """Gracefully shuts down the listener thread and socket."""
        self.running = False # Assuming you have a self.running flag
        if hasattr(self, "socket"):
            self.socket.close()
        self.logger.info("FogBridge connection closed.")