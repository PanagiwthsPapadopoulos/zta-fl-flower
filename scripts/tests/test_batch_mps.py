"""Queue control regression tests; subprocess fixtures do not simulate ML correctness."""
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch, Mock

import yaml
from scripts.ops import batch_runs_mps as batch
from scripts.ops.batch_runs import BatchError, plan_queue
from src.shared.utils.simulation_lifecycle import atomic_json


class NativeBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.queue = self.root / 'queue'
        self.session = self.root / 'results/native_batch_archives/session'
        self.session.mkdir(parents=True)
        (self.root/'config').mkdir()
        configs = {'network.yaml':{'num_fogs':1,'uniform_edges_per_fog':2},
                   'training.yaml':{'num_rounds':2,'random_seed':42},'security.yaml':{}}
        for name, value in configs.items():
            (self.root/'config'/name).write_text(yaml.safe_dump(value))
        for index, seed in enumerate([42,123],1):
            folder=self.queue/str(index);folder.mkdir(parents=True)
            (folder/'training.yaml').write_text(f'random_seed: {seed}\n')
        _, runs = plan_queue(self.root,self.queue,fixed_topology=False)
        self.state={'version':1,'runtime':'native-mps','runs':runs,'settings':{
            'workers':1,'tpm_image':'test','run_timeout':10,'shutdown_timeout':3,'smoke_test':False}}

    def test_empty_image_is_rejected_before_creating_batch(self):
        before=set(self.root.rglob("*"))
        for image in ("", "   "):
            with patch.object(batch,"ROOT",self.root), patch.object(batch,"execute") as execute, redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as caught:
                    batch.main(["--queue",str(self.queue),"--tpm-image",image])
            self.assertEqual(caught.exception.code,2)
            self.assertIn("--tpm-image must not be empty",errors.getvalue())
            execute.assert_not_called()
        self.assertEqual(before,set(self.root.rglob("*")))

    def test_early_launcher_failure_reports_exit_code_and_log_path(self):
        folder=self.root/"attempt"
        entry=dict(self.state["runs"][0],run_dir=str(self.root/"never_created"))
        child=Mock(returncode=2);child.poll.return_value=2
        with patch.object(batch.subprocess,"Popen",return_value=child), patch.object(batch,"stop_launcher"):
            with self.assertRaises(BatchError) as caught:
                batch.run_one(self.root,folder,entry,self.state["settings"],lambda:False,Mock())
        self.assertIn("exited 2 without native_status.json",str(caught.exception))
        self.assertIn(str(folder/"launcher.log"),str(caught.exception))

    def test_dry_run_is_read_only_and_needs_no_mps(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(batch,'ROOT',self.root), patch.object(batch,'execute') as execute, redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(['--queue',str(self.queue),'--dry-run']),0)
        execute.assert_not_called()
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_native_queue_allows_distinct_topologies_for_fresh_networks(self):
        (self.queue/'2'/'network.yaml').write_text('num_fogs: 3\n')
        _, runs=plan_queue(self.root,self.queue,fixed_topology=False)
        self.assertEqual([e['topology'] for e in runs],[[2],[2,2,2]])

    def test_failure_stops_queue_and_retry_preserves_history(self):
        def fail(root,folder,entry,settings,cancelled,update):
            atomic_json(Path(entry['run_dir'])/'native_status.json',{'status':'FAILED','cleanup_errors':[]})
            return 'FAILED','fixture'
        with patch.object(batch,'run_one',side_effect=fail) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(batch.execute(self.root,self.session,self.state),1)
        run.assert_called_once()
        self.assertEqual([e['status'] for e in self.state['runs']],['FAILED','PENDING'])
        frozen=self.state['runs'][0]['configs']
        restored=batch.resume_state(self.session,True)
        self.assertEqual([e['status'] for e in restored['runs']],['PENDING','PENDING'])
        self.assertEqual(restored['runs'][0]['configs'],frozen)
        self.assertEqual(restored['runs'][0]['previous_attempts'][0]['status'],'FAILED')

    def test_uncertain_cleanup_blocks_resume_even_without_retry(self):
        self.state['runs'][0].update(status='FAILED',run_dir=str(self.root/'unknown'))
        atomic_json(self.session/'batch.json',self.state)
        with self.assertRaisesRegex(BatchError,'cleanup'):
            batch.resume_state(self.session,False)

    def test_finished_attempt_is_reconciled_without_repeating_it(self):
        run=self.root/'native_done';entry=self.state['runs'][0]
        entry.update(status='RUNNING',run_dir=str(run))
        atomic_json(run/'native_status.json',{'status':'COMPLETED','cleanup_errors':[]})
        evidence=run/'results'/run.name
        atomic_json(evidence/'batch_lifecycle.json',{'attempt':run.name,'status':'COMPLETED'})
        atomic_json(evidence/'batch_progress.json',{'attempt':run.name,'completed_round':2})
        (evidence/'global_model.pt').write_bytes(b'checkpoint')
        atomic_json(self.session/'batch.json',self.state)
        restored=batch.resume_state(self.session,True)
        self.assertEqual(restored['runs'][0]['status'],'SUCCESS')

    def test_same_session_cannot_be_started_twice(self):
        import fcntl
        atomic_json(self.session/'batch.json',self.state)
        with (self.session/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with patch.object(batch,'ROOT',self.root), patch.object(batch,'execute') as execute:
                self.assertEqual(batch.main(['--resume',str(self.session)]),2)
            execute.assert_not_called()

    def test_two_actual_subprocesses_have_isolated_outputs_and_frozen_configs(self):
        script=self.root/'scripts/simulation/run_native_mps.py';script.parent.mkdir(parents=True)
        script.write_text('''import argparse,json,os
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--config-dir');p.add_argument('--run-dir');p.add_argument('--workers');p.add_argument('--tpm-image')
a=p.parse_args();r=Path(a.run_dir);e=r/'results'/r.name;e.mkdir(parents=True)
(e/'batch_lifecycle.json').write_text(json.dumps({'attempt':r.name,'status':'COMPLETED'}))
(e/'batch_progress.json').write_text(json.dumps({'attempt':r.name,'completed_round':2}))
(e/'global_model.pt').write_bytes(b'fixture checkpoint')
(r/'native_status.json').write_text(json.dumps({'status':'COMPLETED','cleanup_errors':[]}))
(r/'seen.txt').write_text(str(os.getpid())+'\\n'+Path(a.config_dir,'training.yaml').read_text())
''')
        before={p:p.read_bytes() for p in (self.root/'config').iterdir()}
        # Queue changes after planning must not leak into the frozen execution.
        (self.queue/'1'/'training.yaml').write_text('random_seed: 999\n')
        with redirect_stdout(io.StringIO()):
            self.assertEqual(batch.execute(self.root,self.session,self.state),0)
        runs=self.state['runs']
        self.assertEqual([e['status'] for e in runs],['SUCCESS','SUCCESS'])
        seen=[(Path(e['run_dir'])/'seen.txt').read_text() for e in runs]
        self.assertNotEqual(seen[0].splitlines()[0],seen[1].splitlines()[0])
        self.assertIn('random_seed: 42',seen[0]);self.assertIn('random_seed: 123',seen[1])
        self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_timeout_and_cancel_request_launcher_cleanup(self):
        for cancelled,expected in [(False,'TIMEOUT'),(True,'CANCELLED')]:
            with self.subTest(expected=expected):
                run=self.root/expected
                entry=dict(self.state['runs'][0],run_dir=str(run))
                atomic_json(run/'native_status.json',{'status':'FAILED','cleanup_errors':[]})
                child=Mock(returncode=1);child.poll.return_value=None
                settings=dict(self.state['settings'],run_timeout=0)
                with patch.object(batch.subprocess,'Popen',return_value=child),patch.object(batch,'stop_launcher') as stop:
                    result=batch.run_one(self.root,self.root/('attempt_'+expected),entry,settings,lambda:cancelled,Mock())
                self.assertEqual(result[0],expected);stop.assert_called_once_with(child,3)

    def test_cleanup_failure_aborts_next_experiment(self):
        with patch.object(batch,'run_one',side_effect=BatchError('Cleanup not confirmed')) as run, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(BatchError,'Cleanup'):
                batch.execute(self.root,self.session,self.state)
        run.assert_called_once();self.assertEqual(self.state['runs'][1]['status'],'PENDING')


if __name__=='__main__':
    unittest.main()
