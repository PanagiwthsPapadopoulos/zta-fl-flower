import logging
from typing import Dict, TypedDict


class NodeState(TypedDict):
    score: float
    is_quarantined: bool
    recovery_streak: int


class TrustDatabase:
    """A stateful ledger and evaluation engine for tracking persistent risk levels and applying quarantine penalties."""
    def __init__(self, logger: logging.Logger):
        """Initializes the TrustDatabase with default scoring mechanics and quarantine parameters."""
        self.logger = logger
        self._db: Dict[str, NodeState] = {}
        
        self.INIT_SCORE = 0.7
        self.MIN_THRESHOLD = 0.6
        self.REWARD_STEP = 0.02
        self.PENALTY_MULTIPLIER = 0.5
        self.RECOVERY_REQUIREMENT = 5
        self.RECOVERY_RESET_SCORE = 0.65

    def register_node(self, node_id: str, display_name: str, round_num: int) -> None:
        """Registers a newly discovered node in the trust ledger with an initial baseline score."""
        if node_id not in self._db:
            self._db[node_id] = {
                "score": self.INIT_SCORE,
                "is_quarantined": False,
                "recovery_streak": 0
            }
            self.logger.debug(f"[TrustDB] Agent {display_name} registered into ledger. Init Score: {self.INIT_SCORE}", extra={"round": round_num})

    def process_attestation(self, node_id: str, display_name: str, is_valid: bool, round_num: int) -> None:
        """Processes a cryptographic attestation result, adjusting the node's score and quarantine boundaries."""
        self.logger.debug(f"[TrustDB-PROCESS] Ingesting Cryptographic Result for {display_name} | Valid Signature/Nonce: {is_valid}", extra={"round": round_num})
        
        if node_id not in self._db:
            self.register_node(node_id, display_name, round_num)

        state = self._db[node_id]

        if not is_valid:
            self._apply_penalty(node_id, display_name, state, round_num)
        else:
            if state["is_quarantined"]:
                self._process_recovery(node_id, display_name, state, round_num)
            
    def apply_behavioral_reward(self, node_id: str, display_name: str, round_num: int) -> None:
        """Public endpoint to grant a trust reward based on behavioral analysis (e.g., SHAP stability)."""
        if node_id in self._db and not self._db[node_id]["is_quarantined"]:
            self._apply_reward(node_id, display_name, self._db[node_id], round_num)

    def _apply_penalty(self, node_id: str, display_name: str, state: NodeState, round_num: int) -> None:
        """Applies a multiplicative penalty to a node's score following a failed security attestation."""
        old_score = state["score"]
        state["score"] *= self.PENALTY_MULTIPLIER
        state["recovery_streak"] = 0 
        self.logger.warning(f"TrustDB: Agent {display_name} penalized. Score dropped: {old_score:.3f} -> {state['score']:.3f}", extra={"round": round_num})

        if state["score"] < self.MIN_THRESHOLD and not state["is_quarantined"]:
            state["is_quarantined"] = True
            self.logger.error(f"TrustDB: Agent {display_name} fell below {self.MIN_THRESHOLD} threshold. QUARANTINE ENGAGED.", extra={"round": round_num})

    def _apply_reward(self, node_id: str, display_name: str, state: NodeState, round_num: int) -> None:
        """Increments a node's trust score following a successfully verified behavioral security evaluation."""
        old_score = state["score"]
        state["score"] = min(1.0, state["score"] + self.REWARD_STEP)
        self.logger.debug(f"TrustDB: Agent {display_name} rewarded. Score increased: {old_score:.3f} -> {state['score']:.3f}", extra={"round": round_num})

    def _process_recovery(self, node_id: str, display_name: str, state: NodeState, round_num: int) -> None:
        """Tracks consecutive recovery streaks for quarantined nodes, lifting restrictions upon completion."""
        state["recovery_streak"] += 1
        self.logger.info(f"TrustDB: Quarantined Agent {display_name} logged valid attestation. Recovery streak: {state['recovery_streak']}/{self.RECOVERY_REQUIREMENT}", extra={"round": round_num})

        if state["recovery_streak"] >= self.RECOVERY_REQUIREMENT:
            state["is_quarantined"] = False
            state["score"] = self.RECOVERY_RESET_SCORE
            state["recovery_streak"] = 0
            self.logger.info(f"TrustDB: Agent {display_name} successfully completed rehabilitation. QUARANTINE LIFTED.", extra={"round": round_num})

    def is_quarantined(self, node_id: str) -> bool:
        """Checks the internal ledger to determine whether a given node is currently in a quarantined state."""
        if node_id not in self._db:
            return True
        return self._db[node_id]["is_quarantined"]

    def get_score(self, node_id: str) -> float:
        """Retrieves the current trust evaluation score mapping to the specified node identifier."""
        return self._db.get(node_id, {}).get("score", 0.0)