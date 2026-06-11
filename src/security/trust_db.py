import logging

class TrustDB:
    """
    Maintains the Reputation and Attestation state of Edge Agents.
    Matches the TrustDB Policy from the ZTA-FL Paper.
    """
    def __init__(self, logger: logging.Logger, tau_min: float = 0.6):
        self.logger = logger
        # Maps node_id -> Trust Score (tau_i)
        self.agent_scores = {}
        # The quarantine threshold
        self.tau_min = tau_min 

    def get_trust_score(self, node_id: str) -> float:
        """New agents start at tau_i = 0.7 after first attestation."""
        if node_id not in self.agent_scores:
            self.agent_scores[node_id] = 0.7
            self.logger.info(f"[TrustDB] Initialized new agent {node_id} at 0.7")
        return self.agent_scores[node_id]

    def is_quarantined(self, node_id: str) -> bool:
        """Agents with tau_i < 0.6 enter quarantine."""
        return self.get_trust_score(node_id) < self.tau_min

    def penalize_agent(self, node_id: str, reason: str):
        """
        Failed attestation or SHAP filtering triggers a massive penalty: tau_i = tau_i * 0.5
        """
        current = self.get_trust_score(node_id)
        self.agent_scores[node_id] = current * 0.5
        self.logger.warning(f"[TrustDB] 🚨 PENALTY applied to {node_id} ({reason}). New score: {self.agent_scores[node_id]:.3f}")

    def reward_agent(self, node_id: str):
        """
        After each successful round with above-average SHAP stability, tau_i += 0.02 (Max 1.0)
        """
        current = self.get_trust_score(node_id)
        new_score = min(1.0, current + 0.02)
        self.agent_scores[node_id] = new_score
        self.logger.debug(f"[TrustDB] Reward applied to {node_id}. New score: {new_score:.3f}")