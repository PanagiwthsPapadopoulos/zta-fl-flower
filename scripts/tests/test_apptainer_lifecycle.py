"""Regression coverage for orphaned helpers; no containers or real signals."""
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.ops import apptainer_lifecycle as lifecycle
from scripts.ops import batch_runs as batch


def process(pid, name="python3", ppid=1, sid=10, start=None):
    return dict(pid=pid, name=name, ppid=ppid, sid=sid, pgid=pid,
                start_time=pid * 100 if start is None else start, state="S")


class ApptainerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "results/attempt/process_inventory.json"
        self.inventory = lifecycle.ProcessInventory(self.path, "attempt")

    def begin(self, baseline=()):
        with patch.object(lifecycle, "snapshot", return_value=({r["pid"]: r for r in baseline}, set())):
            self.inventory.begin()

    def scan(self, records, marked=(), unreadable=()):
        with patch.object(lifecycle, "snapshot", return_value=({r["pid"]: r for r in records}, set(unreadable))):
            return self.inventory.scan(marked)

    def test_unmarked_helper_in_separate_group_remains_owned_after_reparenting_and_recovery(self):
        self.begin()
        shell = process(10, "bash")
        runtime = process(11, "Apptainer runtim", ppid=10)
        helper = process(12, "squashfuse_ll", ppid=11)
        self.scan([shell, runtime, helper], [shell])
        helper["ppid"] = 1
        recovered = lifecycle.ProcessInventory(self.path, "attempt")
        with patch.object(lifecycle, "snapshot", return_value=({12: helper}, set())):
            owned, unknown, _ = recovered.scan([])
            self.assertEqual([r["pid"] for r in owned], [12])
            self.assertEqual(unknown, [])
            with self.assertRaisesRegex(lifecycle.LifecycleError, "shutdown unconfirmed"):
                recovered.assert_clean([])

    def test_unobserved_orphan_blocks_completion_without_claiming_ownership(self):
        self.begin()
        orphan = process(12, "squashfuse_ll", sid=999)
        with patch.object(lifecycle, "snapshot", return_value=({12: orphan}, set())):
            owned, unknown, _ = self.inventory.scan([])
            self.assertEqual(owned, [])
            self.assertEqual(unknown, [orphan])
            with self.assertRaises(lifecycle.LifecycleError):
                self.inventory.assert_clean([])

    def test_preexisting_foreign_helpers_are_not_owned_or_signalled(self):
        foreign = process(80, "squashfuse_ll", sid=80)
        self.begin([foreign])
        with patch.object(lifecycle, "snapshot", return_value=({80: foreign}, set())):
            self.inventory.assert_clean([])
        self.assertEqual(self.inventory.known, {})

    def test_reused_pid_and_session_do_not_inherit_ownership(self):
        self.begin()
        original = process(10, "bash")
        self.scan([original], [original])
        reused = process(10, "bash", start=9999)
        foreign = process(12, "squashfuse_ll", ppid=10)
        owned, unknown, _ = self.scan([reused, foreign], [original])
        self.assertEqual(owned, [])
        self.assertEqual(unknown, [foreign])

    def test_unreadable_owned_process_is_not_treated_as_gone(self):
        self.begin()
        original = process(10)
        self.scan([original], [original])
        with self.assertRaisesRegex(lifecycle.LifecycleError, "Cannot inspect"):
            self.scan([], unreadable=[10])

    def test_pidfd_revalidation_skips_reused_pid_and_runtime(self):
        original = process(10)
        for current in (process(10, start=999), process(10, "starter")):
            with self.subTest(current=current), \
                 patch.object(lifecycle.os, "pidfd_open", return_value=100, create=True), \
                 patch.object(lifecycle.signal, "pidfd_send_signal", create=True) as send, \
                 patch.object(lifecycle.os, "close") as close, \
                 patch.object(lifecycle, "process_record", return_value=current):
                lifecycle.signal_payload(original, signal.SIGKILL)
                send.assert_not_called()
                close.assert_called_once_with(100)

    def test_shutdown_allows_runtime_cleanup_and_never_signals_helpers(self):
        self.begin()
        shell = process(10, "bash")
        runtime = process(11, "Apptainer runtim", ppid=10)
        helper = process(12, "squashfuse_ll", ppid=11)
        payload = process(13, "swtpm", ppid=11)
        self.scan([shell, runtime, helper, payload], [shell])
        rt = batch.Runtime(self.root, "apptainer", "attempt", [1])
        rt.inventory = self.inventory
        snapshots = [({r["pid"]: r for r in [shell, runtime, helper, payload]}, set()),
                     ({11: runtime, 12: helper}, set()), ({}, set()), ({}, set())]
        with patch.object(lifecycle, "snapshot", side_effect=snapshots), \
             patch.object(batch, "owned_processes", return_value=[]), \
             patch.object(batch, "signal_payload") as send, \
             patch.object(batch.time, "sleep"):
            rt.stop(1)
        self.assertEqual({call.args[0]["pid"] for call in send.call_args_list}, {10, 13})
        self.assertTrue(all(call.args[1] == signal.SIGTERM for call in send.call_args_list))

    def test_stuck_helper_halts_without_force_killing_runtime(self):
        self.begin()
        helper = process(12, "squashfuse_ll")
        self.scan([helper], [helper])
        rt = batch.Runtime(self.root, "apptainer", "attempt", [1])
        rt.inventory = self.inventory
        with patch.object(lifecycle, "snapshot", return_value=({12: helper}, set())), \
             patch.object(batch, "owned_processes", return_value=[]), \
             patch.object(batch, "signal_payload") as send, \
             patch.object(batch.time, "monotonic", side_effect=range(100)):
            with self.assertRaisesRegex(batch.BatchError, "shutdown unconfirmed"):
                rt.stop(.1)
        send.assert_not_called()
        self.assertTrue(self.path.exists())

    def test_payload_escalation_keeps_runtime_alive_for_cleanup(self):
        self.begin()
        runtime, helper, payload = (process(11, "starter"), process(12, "squashfuse_ll"), process(13))
        self.scan([runtime, helper, payload], [runtime])
        rt = batch.Runtime(self.root, "apptainer", "attempt", [1])
        rt.inventory = self.inventory
        state = {r["pid"]: r for r in [runtime, helper, payload]}
        def signal_child(record, sig):
            if sig == signal.SIGKILL:
                state.clear()  # Runtime completes teardown after its payload exits.
        with patch.object(lifecycle, "snapshot", side_effect=lambda: (state.copy(), set())), \
             patch.object(batch, "owned_processes", return_value=[]), \
             patch.object(batch, "signal_payload", side_effect=signal_child) as send, \
             patch.object(batch.time, "monotonic", side_effect=range(100)):
            rt.stop(.1)
        self.assertEqual([(c.args[0]["pid"], c.args[1]) for c in send.call_args_list],
                         [(13, signal.SIGTERM), (13, signal.SIGKILL)])

    def test_missing_legacy_inventory_cannot_hide_unmarked_helpers(self):
        rt = batch.Runtime(self.root, "apptainer", "attempt", [1])
        helper = process(12, "squashfuse_ll")
        with patch.object(lifecycle, "snapshot", return_value=({12: helper}, set())), \
             patch.object(batch, "owned_processes", return_value=[]), \
             patch.object(batch, "signal_payload") as send:
            with self.assertRaises(batch.BatchError):
                rt.stop(.1)
        send.assert_not_called()

    def test_managed_boot_signal_exits_without_killing_launchers(self):
        source = (batch.ROOT / "scripts/simulation/boot_network_sim_apptainer.sh").read_text()
        handlers = source.split("cleanup() {", 1)[1].split("if [ -L /etc/localtime", 1)[0]
        infra = self.root / "runtime/infra"
        infra.mkdir(parents=True)
        (infra / ".network_ready").touch()
        script = 'PIDS=(424242)\nkill() { echo "$*" > "$PROJECT_ROOT/signals"; }\n'
        script += "cleanup() {" + handlers + '\nbuiltin kill -TERM "$$"\n'
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=5,
                                env={**os.environ, "PROJECT_ROOT": str(self.root), "ZTA_BATCH_MANAGED": "1"})
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertFalse((self.root / "signals").exists())
        self.assertFalse((infra / ".network_ready").exists())

    def test_boot_rejects_failed_provisioning_and_socket_timeout(self):
        from scripts.tests.test_runtime_mounts import RuntimeMountTests
        for failure in ("provision", "socket"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                root = Path(directory)
                env = RuntimeMountTests().prepare_runtime(root)
                env.update(ZTA_BATCH_MANAGED="1", ZTA_TPM_START_TIMEOUT="0")
                stub = root / "bin/apptainer"
                text = stub.read_text()
                if failure == "provision":
                    text += '\nif any("TPMAttestation" in arg for arg in args): sys.exit(7)\n'
                else:
                    text = text.replace("Path(path).touch()", "pass")
                stub.write_text(text)
                source = (batch.ROOT / "scripts/simulation/boot_network_sim_apptainer.sh").read_text()
                source = source.replace('[ ! -S "$EDGE_TPM_DIR/swtpm.sock" ]',
                                        '[ ! -e "$EDGE_TPM_DIR/swtpm.sock" ]')
                # Only the timeout case uses an immediate deadline; let the stub
                # materialize readiness before checking provisioning failure.
                if failure == "provision":
                    env["ZTA_TPM_START_TIMEOUT"] = "5"
                script = root / "scripts/simulation/boot_network_sim_apptainer.sh"
                script.write_text(source)
                result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("FATAL:", result.stdout)
                self.assertFalse((root / "runtime/infra/.network_ready").exists())


if __name__ == "__main__":
    unittest.main()
