import socket
import logging
import numpy as np
from typing import List

from src.shared.network.ipc import send_msg, recv_msg


class FogBridgeServer:
    """An IPC socket listener for intercepting, synchronizing, and distributing parameters across local connections."""
    def __init__(self, logger: logging.Logger, log_prefix: str, ipc_port: int, socket_timeout: float = 600.0):
        """Initializes the Fog Bridge Server to listen for and manage incoming local IPC connections."""
        self.logger = logger
        self.log_prefix = log_prefix
        self.ipc_port = ipc_port
        self.socket_timeout = socket_timeout
        self.ipc_server_sock = None
        self.active_conn = None
        self.current_round = 0 
        
        self._bind_socket()

    def _bind_socket(self):
        """Binds the server socket to the designated IPC port and initiates the listening state."""
        try:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Booting IPC listener on 0.0.0.0:{self.ipc_port}...", extra={"round": 0})
            self.ipc_server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.ipc_server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.ipc_server_sock.settimeout(self.socket_timeout)
            self.ipc_server_sock.bind(("0.0.0.0", self.ipc_port))
            self.ipc_server_sock.listen(1)
            self.logger.info(f"{self.log_prefix} [IPC SERVER] Listening on port {self.ipc_port}. Awaiting local Fog Client bridge...", extra={"round": 0})
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception during socket bind: {e}", extra={"round": 0})
            self.ipc_server_sock = None

    def wait_for_start(self) -> int:
        """Blocks execution to await an incoming START payload from a connected Fog Bridge Client."""
        if not self.ipc_server_sock:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Socket not initialized. Bypassing wait_for_start.", extra={"round": self.current_round})
            return 0
            
        try:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Blocking for incoming connection from local Fog Client...", extra={"round": self.current_round})
            self.active_conn, addr = self.ipc_server_sock.accept()
            self.active_conn.settimeout(self.socket_timeout)
            
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Connection established from {addr}. Awaiting START payload...", extra={"round": self.current_round})
            msg = recv_msg(self.active_conn)
            
            if isinstance(msg, dict) and msg.get("cmd") == "START":
                self.current_round = msg.get("round", 0)
                self.logger.info(f"{self.log_prefix} [IPC SERVER] START received for round {self.current_round}. Shouting to EDGE clients!", extra={"round": self.current_round})
                return self.current_round
            else:
                self.logger.debug(f"{self.log_prefix} [IPC SERVER] Invalid message received on bridge: {msg}", extra={"round": self.current_round})
                return 0
                
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception while waiting for START: {e}", extra={"round": self.current_round})
            return 0

    def relay_weights(self, ndarrays: List[np.ndarray]):
        """Relays the aggregated global model weights back to the actively connected Fog Bridge Client."""
        if not self.active_conn:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] No active connection available to relay weights. Skipping.", extra={"round": self.current_round})
            return
            
        try:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Relaying {len(ndarrays) if ndarrays else 0} layers back to Fog Client...", extra={"round": self.current_round})
            send_msg(self.active_conn, ndarrays if ndarrays else [])
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Relay successful. Dropping active connection.", extra={"round": self.current_round})
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception during weight relay: {e}", extra={"round": self.current_round})
        finally:
            try:
                self.active_conn.close()
            except Exception:
                pass
            self.active_conn = None