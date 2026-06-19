import os
import time
import socket
import logging
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

from src.network.ipc import send_msg, recv_msg

class FogBridgeClient:
    def __init__(self, logger: logging.Logger, log_prefix: str, ipc_port: int, socket_timeout: float = 600.0):
        self.logger = logger
        self.log_prefix = log_prefix
        self.ipc_port = ipc_port
        self.socket_timeout = socket_timeout
        self.target_host = os.getenv("FOG_SERVER_HOST", "127.0.0.1")

    def _connect_with_retries(self, current_round: int) -> Optional[socket.socket]:
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


class FogBridgeServer:
    def __init__(self, logger: logging.Logger, log_prefix: str, ipc_port: int, socket_timeout: float = 600.0):
        self.logger = logger
        self.log_prefix = log_prefix
        self.ipc_port = ipc_port
        self.socket_timeout = socket_timeout
        self.ipc_server_sock = None
        self.active_conn = None
        self.current_round = 0  # 🚨 Added state to track the round
        
        self._bind_socket()

    def _bind_socket(self):
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
        if not self.ipc_server_sock:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Socket not initialized. Bypassing wait_for_start.", extra={"round": self.current_round})
            return 0
            
        try:
            # We use current_round here so that while it blocks, it prints the round it is currently in/finishing
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Blocking for incoming connection from local Fog Client...", extra={"round": self.current_round})
            self.active_conn, addr = self.ipc_server_sock.accept()
            self.active_conn.settimeout(self.socket_timeout)
            
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Connection established from {addr}. Awaiting START payload...", extra={"round": self.current_round})
            msg = recv_msg(self.active_conn)
            
            if isinstance(msg, dict) and msg.get("cmd") == "START":
                self.current_round = msg.get("round", 0) # 🚨 Update state!
                self.logger.info(f"{self.log_prefix} [IPC SERVER] START received for round {self.current_round}. Shouting to EDGE clients!", extra={"round": self.current_round})
                return self.current_round
            else:
                self.logger.debug(f"{self.log_prefix} [IPC SERVER] Invalid message received on bridge: {msg}", extra={"round": self.current_round})
                return 0
                
        except Exception as e:
            self.logger.debug(f"{self.log_prefix} [IPC SERVER] Non-fatal exception while waiting for START: {e}", extra={"round": self.current_round})
            return 0

    def relay_weights(self, ndarrays: List[np.ndarray]):
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