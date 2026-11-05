import logging
from typing import Dict, TypedDict


class NodeState(TypedDict):
    score: float
    is_quarantined: bool
    recovery_streak: int
    last_attestation_passed: bool
    last_attestation_round: int
    last_penalty_round: int


class TrustDatabase:
    """Paper V-A: readmit on the fifth consecutive valid attestation round."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._db: Dict[str, NodeState] = {}
        self.INIT_SCORE = 0.7
        self.MIN_THRESHOLD = 0.6
        self.REWARD_STEP = 0.02
        self.PENALTY_MULTIPLIER = 0.5
        self.RECOVERY_REQUIREMENT = 5
        self.RECOVERY_RESET_SCORE = 0.65

    def register_node(self, node_id: str, display_name: str, round_num: int) -> None:
        """Called only after successful first authentication."""
        if node_id not in self._db:
            self._db[node_id] = {
                "score": self.INIT_SCORE, "is_quarantined": False,
                "recovery_streak": 0, "last_attestation_passed": True,
                "last_attestation_round": -1, "last_penalty_round": -1,
            }
            self.logger.debug(f"[TrustDB] Registered {display_name} after authentication.", extra={"round": round_num})

    def process_attestation(self, node_id: str, display_name: str, is_valid: bool, round_num: int) -> None:
        if node_id not in self._db:
            if not is_valid:
                return  # An unauthenticated claim cannot create a trusted identity.
            self.register_node(node_id, display_name, round_num)
        state = self._db[node_id]
        previous_round = state["last_attestation_round"]
        if round_num < previous_round:
            return
        if not is_valid:
            state["last_attestation_passed"] = False
            state["last_attestation_round"] = round_num
            self._apply_penalty(node_id, display_name, state, round_num)
            return
        if round_num == previous_round:
            return  # A repeated quote cannot advance recovery or undo a failure.
        consecutive = state["last_attestation_passed"] and previous_round == round_num - 1
        state["last_attestation_passed"] = True
        state["last_attestation_round"] = round_num
        if state["is_quarantined"]:
            state["recovery_streak"] = state["recovery_streak"] + 1 if consecutive else 1
            if state["recovery_streak"] >= self.RECOVERY_REQUIREMENT:
                state["is_quarantined"] = False
                state["score"] = self.RECOVERY_RESET_SCORE
                state["recovery_streak"] = 0
                self.logger.info(f"[TrustDB] {display_name} readmitted on the fifth consecutive valid attestation.", extra={"round": round_num})
            else:
                self.logger.info(f"[TrustDB] {display_name} recovery {state['recovery_streak']}/{self.RECOVERY_REQUIREMENT}; still excluded.", extra={"round": round_num})

    def record_missing_attestation(self, node_id: str, round_num: int) -> None:
        """A missing quote breaks recovery continuity without inventing a failed quote.

        A training failure after valid admission does not invalidate that quote.
        No trust reward or penalty is assigned to the missing response itself.
        """
        state = self._db.get(node_id)
        if state is not None and round_num > state["last_attestation_round"]:
            state["last_attestation_round"] = round_num
            state["last_attestation_passed"] = False
            state["recovery_streak"] = 0

    def apply_behavioral_penalty(self, node_id: str, display_name: str, round_num: int) -> None:
        if node_id in self._db:
            self._apply_penalty(node_id, display_name, self._db[node_id], round_num)

    def apply_behavioral_reward(self, node_id: str, display_name: str, round_num: int) -> None:
        if node_id in self._db and not self.is_quarantined(node_id):
            state = self._db[node_id]
            state["score"] = min(1.0, state["score"] + self.REWARD_STEP)

    def _apply_penalty(self, node_id: str, display_name: str, state: NodeState, round_num: int) -> None:
        # One failed round cannot be charged repeatedly through duplicate messages.
        if state["last_penalty_round"] != round_num:
            state["score"] *= self.PENALTY_MULTIPLIER
            state["last_penalty_round"] = round_num
        state["recovery_streak"] = 0
        if state["score"] < self.MIN_THRESHOLD:
            state["is_quarantined"] = True
        self.logger.warning(f"[TrustDB] {display_name} penalized; score={state['score']:.3f}.", extra={"round": round_num})

    def is_quarantined(self, node_id: str) -> bool:
        return self._db.get(node_id, {}).get("is_quarantined", True)

    def get_score(self, node_id: str) -> float:
        return self._db.get(node_id, {}).get("score", 0.0)
