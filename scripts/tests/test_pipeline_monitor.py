"""Log compatibility for deployed and simulated bridge phases."""
import unittest

from scripts.ops.verify_pipeline import parse_round_state


class PipelineMonitorTests(unittest.TestCase):
    def test_bridge_events_in_both_modes(self):
        for prefix in ('IPC', 'SIMULATED IPC'):
            with self.subTest(prefix=prefix):
                logs = [
                    dict(node='[FOG 1 CLIENT]', message=f'[{prefix} CLIENT] Connected! Sending START signal for round 1.'),
                    dict(node='[FOG 1 SERVER]', message=f'[{prefix} SERVER] START received for round 1. Shouting to EDGE clients!'),
                ]
                state = parse_round_state(logs, {})
                self.assertEqual(state['fog_clients_started'], {'[FOG 1 CLIENT]'})
                self.assertEqual(state['fog_servers_started'], {'[FOG 1 SERVER]'})
                self.assertEqual(state['fog_servers_aggregated'], set())

    def test_waiting_and_failed_bridge_do_not_count_as_started(self):
        state = parse_round_state([
            dict(node='[FOG 1 SERVER]', message='[IPC SERVER] Failed to establish valid round sync.'),
            dict(node='[FOG 1 SERVER]', message='[IPC SERVER] Listening. Awaiting local Fog Client bridge...'),
        ], {})
        self.assertEqual(state['fog_servers_started'], set())
        self.assertEqual(state['fog_servers_aggregated'], set())
