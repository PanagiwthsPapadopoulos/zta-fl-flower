import logging
from typing import Dict, TypedDict

class NodeState(TypedDict):
    score: float
    is_quarantined: bool
    recovery_streak: int

class TrustDatabase:
    """
    Stateful trust evaluation engine for federated edge nodes.
    Enforces dynamic penalties, rewards, and quarantine boundaries
    as defined by the Zero-Trust Agentic Federated Learning (ZTA-FL) architecture.
    """
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._db: Dict[str, NodeState] = {}
        
        # Architectural Constants
        self.INIT_SCORE = 0.7
        self.MIN_THRESHOLD = 0.6
        self.REWARD_STEP = 0.02
        self.PENALTY_MULTIPLIER = 0.5
        self.RECOVERY_REQUIREMENT = 5
        self.RECOVERY_RESET_SCORE = 0.65

    def register_node(self, node_id: str) -> None:
        """Initializes a newly connected node into the trust ledger."""
        if node_id not in self._db:
            self._db[node_id] = {
                "score": self.INIT_SCORE,
                "is_quarantined": False,
                "recovery_streak": 0
            }
            self.logger.debug(f"TrustDB: Node {node_id[:16]} registered with initial score {self.INIT_SCORE}.")

    def process_attestation(self, node_id: str, is_valid: bool) -> None:
        """
        Updates the trust metric based on attestation or SHAP verification results.
        Enforces penalty drops, reward clamping, and quarantine mechanics.
        """
        if node_id not in self._db:
            self.register_node(node_id)

        state = self._db[node_id]

        if not is_valid:
            self._apply_penalty(node_id, state)
        else:
            if state["is_quarantined"]:
                self._process_recovery(node_id, state)
            else:
                self._apply_reward(node_id, state)

    def _apply_penalty(self, node_id: str, state: NodeState) -> None:
        """Halves the trust score and triggers quarantine if below threshold."""
        old_score = state["score"]
        state["score"] *= self.PENALTY_MULTIPLIER
        state["recovery_streak"] = 0 # Reset any partial recovery
        
        self.logger.warning(f"TrustDB: Node {node_id[:16]} penalized. Score dropped: {old_score:.3f} -> {state['score']:.3f}")

        if state["score"] < self.MIN_THRESHOLD and not state["is_quarantined"]:
            state["is_quarantined"] = True
            self.logger.error(f"TrustDB: Node {node_id[:16]} fell below {self.MIN_THRESHOLD} threshold. QUARANTINE ENGAGED.")

    def _apply_reward(self, node_id: str, state: NodeState) -> None:
        """Increments the trust score, capped at 1.0."""
        old_score = state["score"]
        state["score"] = min(1.0, state["score"] + self.REWARD_STEP)
        self.logger.debug(f"TrustDB: Node {node_id[:16]} rewarded. Score increased: {old_score:.3f} -> {state['score']:.3f}")

    def _process_recovery(self, node_id: str, state: NodeState) -> None:
        """Tracks consecutive valid attestations for quarantined nodes."""
        state["recovery_streak"] += 1
        self.logger.info(f"TrustDB: Quarantined node {node_id[:16]} logged valid attestation. Recovery streak: {state['recovery_streak']}/{self.RECOVERY_REQUIREMENT}")

        if state["recovery_streak"] >= self.RECOVERY_REQUIREMENT:
            state["is_quarantined"] = False
            state["score"] = self.RECOVERY_RESET_SCORE
            state["recovery_streak"] = 0
            self.logger.info(f"TrustDB: Node {node_id[:16]} successfully completed rehabilitation. QUARANTINE LIFTED. Score reset to {self.RECOVERY_RESET_SCORE}.")

    def is_quarantined(self, node_id: str) -> bool:
        """Returns True if the node is actively quarantined or unregistered."""
        if node_id not in self._db:
            return True
        return self._db[node_id]["is_quarantined"]

    def get_score(self, node_id: str) -> float:
        """Returns the current trust score, or 0.0 if unregistered."""
        return self._db.get(node_id, {}).get("score", 0.0)