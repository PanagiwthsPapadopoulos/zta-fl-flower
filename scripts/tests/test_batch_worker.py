"""Persistent queue handover and failure tests; no HPC workload is launched."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
from contextlib import redirect_stdout
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import yaml
from scripts.ops import batch_runs as batch
from scripts.ops import batch_worker as worker
from scripts.tests.test_batch_runs import LIMITS


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = self.root / 'results/batch_archives/session'
        self.session.mkdir(parents=True)
        (self.root / 'config').mkdir()
        self.baseline = {
            'network.yaml': dict(num_fogs=1, uniform_edges_per_fog=100, round_timeout=3600),
            'training.yaml': dict(num_rounds=2, learning_rate=.1, dataset_path='data/raw.csv'),
            'security.yaml': dict(label_flip_ratio=0, tpm_freshness_window=180),
        }
        for name, data in self.baseline.items():
            (self.root / 'config' / name).write_text(yaml.safe_dump(data))
        q = self.root / 'queue'
        for name, filename, data in [('01','training.yaml',{'learning_rate':.2}), ('02','security.yaml',{'label_flip_ratio':.3})]:
            (q / name).mkdir(parents=True)
            (q / name / filename).write_text(yaml.safe_dump(data))
        _, runs = batch.plan_queue(self.root, q)
        self.state = dict(version=2, runtime='apptainer', infrastructure_id='infra_test', active=None,
                          infrastructure_active=False, runs=runs, limits=LIMITS)
        batch.atomic_json(self.session / 'batch.json', self.state)
        self.env = patch.dict(os.environ, {'CONFIG_PATH': str(self.root/'runtime/batch/config')})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_one_coordinator_two_independent_configs_and_unchanged_baseline(self):
        before = {p.name:p.read_bytes() for p in (self.root/'config').iterdir()}
        observed=[]
        def execute(root, folder, entry, limits, flower, update, cancelled):
            config=batch.merged({n:batch.read_yaml(Path(os.environ['CONFIG_PATH'])/n) for n in batch.CONFIG_FILES})
            observed.append((flower, config))
            update(completed_round=2)
            return ('INCOMPLETE','failed') if len(observed)==1 else ('SUCCESS','done')
        with patch.object(worker,'Flower') as cls, patch.object(worker,'run_experiment',side_effect=execute), redirect_stdout(io.StringIO()):
            self.assertEqual(worker.work(self.root,self.session),1)
        cls.assert_called_once()
        cls.return_value.close.assert_called_once()
        self.assertIs(observed[0][0],observed[1][0])
        self.assertEqual([c['learning_rate'] for _,c in observed],[.2,.1])
        self.assertEqual([c['label_flip_ratio'] for _,c in observed],[0,.3])
        self.assertNotEqual(observed[0][1]['batch_attempt'],observed[1][1]['batch_attempt'])
        self.assertEqual(before,{p.name:p.read_bytes() for p in (self.root/'config').iterdir()})
        state=batch.read_json(self.session/'batch.json')
        self.assertEqual([r['status'] for r in state['runs']],['INCOMPLETE','SUCCESS'])
        self.assertIsNone(state['active'])

    def test_stop_timeout_prevents_next_configuration(self):
        with patch.object(worker,'Flower'), patch.object(worker,'run_experiment',side_effect=batch.BatchError('Flower stop timed out')) as execute, redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(batch.BatchError,'Flower stop timed out'):
                worker.work(self.root,self.session)
        execute.assert_called_once()
        state=batch.read_json(self.session/'batch.json')
        self.assertEqual(state['runs'][1]['status'],'PENDING')
        self.assertEqual(batch.read_yaml(Path(os.environ['CONFIG_PATH'])/'training.yaml')['learning_rate'],.2)

    def test_ready_uses_only_flower_api(self):
        flower = worker.Flower.__new__(worker.Flower)
        flower.cli = Mock(return_value={'success': True, 'runs': []})
        flower.ready(5)
        flower.cli.assert_called_once_with('list', 'local', timeout=5)

    def test_stop_waits_for_terminal_status(self):
        flower = worker.Flower.__new__(worker.Flower)
        flower.grace = 10
        flower.cli = Mock(side_effect=[{'runs': [{'run-id': '7', 'status': 'running'}]},
                                      {'runs': [{'run-id': '7', 'status': 'finished:stopped'}]}])
        with patch.object(worker.time, 'sleep'):
            flower.wait_finished('7')
        self.assertEqual(flower.cli.call_count, 2)

    def test_missing_status_ends_on_timeout(self):
        flower = worker.Flower.__new__(worker.Flower)
        flower.grace = 1
        flower.cli = Mock(return_value={'runs': []})
        with patch.object(worker.time, 'monotonic', side_effect=[0, 0, 2]), patch.object(worker.time, 'sleep'):
            with self.assertRaisesRegex(batch.BatchError, 'stop timed out'):
                flower.wait_finished('7')

    def test_cancel_retains_pending_runs(self):
        with patch.object(worker,'Flower') as cls, patch.object(worker,'run_experiment',side_effect=batch.Cancelled('stop')), redirect_stdout(io.StringIO()):
            self.assertEqual(worker.work(self.root,self.session),130)
        cls.return_value.close.assert_called_once()
        self.assertEqual([r['status'] for r in batch.read_json(self.session/'batch.json')['runs']],['CANCELLED','PENDING'])

    def test_flower_terminal_status_controls_completion_without_checkpoint_gate(self):
        (self.root/'pyproject.toml').write_text('[tool.flwr.federations.local.options.backend.client-resources]\nnum-cpus=2\nnum-gpus=0.0\n')
        for row_status, complete, expected in [('finished:failed',True,'INCOMPLETE'),('finished:completed',False,'SUCCESS'),('finished:completed',True,'SUCCESS')]:
            with self.subTest(status=row_status,complete=complete):
                folder=self.session/row_status.replace(':','_')/str(complete)
                folder.mkdir(parents=True)
                flower=Mock()
                flower.cli.side_effect=[{'run-id':'123'}, {'runs':[{'run-id':'123','status':row_status,'status-details':'No heartbeat'}]}, {'success':True}]
                child=Mock(returncode=0)
                child.poll.return_value=0
                entry=dict(self.state['runs'][0],attempt='attempt')
                with patch.object(worker.subprocess,'Popen',return_value=child):
                    status,_=worker.run_experiment(self.root,folder,entry,LIMITS,flower,Mock(),lambda:False)
                self.assertEqual(status,expected)
                flower.wait_finished.assert_not_called()
                self.assertEqual(flower.cli.call_args_list[0].args[-1],'num-supernodes=100 client-resources-num-cpus=2 client-resources-num-gpus=0.0')

    def test_preparation_failure_never_submits_training(self):
        child=Mock(returncode=1);child.poll.return_value=1
        folder=self.session/'attempt';folder.mkdir()
        flower=Mock()
        with patch.object(worker.subprocess,'Popen',return_value=child):
            status,_=worker.run_experiment(self.root,folder,dict(self.state['runs'][0],attempt='test'),LIMITS,flower,Mock(),lambda:False)
        self.assertEqual(status,'CRASHED')
        flower.cli.assert_not_called()
        flower.wait_finished.assert_not_called()

    def test_timed_out_training_requests_stop_and_waits_for_flower(self):
        (self.root/'pyproject.toml').write_text('')
        folder=self.session/'attempt';folder.mkdir()
        flower=Mock()
        flower.cli.side_effect=[{'run-id':'123'},{'runs':[{'run-id':'123','status':'running'}]}, {'success':True}]
        child=Mock(returncode=0);child.poll.return_value=0
        with patch.object(worker.subprocess,'Popen',return_value=child), patch.object(worker.time,'monotonic',side_effect=[0,0,200]):
            status,_=worker.run_experiment(self.root,folder,dict(self.state['runs'][0],attempt='test'),LIMITS,flower,Mock(),lambda:False)
        self.assertEqual(status,'TIMEOUT')
        self.assertEqual(flower.cli.call_args.args,('stop','123','local'))
        flower.wait_finished.assert_called_once_with('123')


    def test_resume_requeues_only_cancelled_and_preserves_prior_attempt(self):
        self.state['runs'][0].update(status='CANCELLED',attempt='old')
        self.state['runs'][1].update(status='SUCCESS',attempt='finished')
        batch.atomic_json(self.session/'batch.json',self.state)
        with patch.object(batch,'ROOT',self.root), patch.object(batch,'supervise',return_value=0) as supervise, redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(['--resume',str(self.session),'--retry-interrupted']),0)
        state=supervise.call_args.args[2]
        self.assertEqual([r['status'] for r in state['runs']],['PENDING','SUCCESS'])
        self.assertEqual(state['runs'][0]['previous_attempts'][0]['attempt'],'old')


    def test_persistent_launchers_use_one_foreground_worker_without_builder_container(self):
        import subprocess
        for backend in ('apptainer','docker'):
            with self.subTest(backend=backend), TemporaryDirectory() as directory:
                root=Path(directory)
                for relative in (f'scripts/simulation/deploy_code_sim_{backend}.sh','scripts/setup/parse_topology.py','pyproject.toml'):
                    target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(batch.ROOT/relative,target)
                shutil.copytree(self.root/'config',root/'config')
                for relative in ('runtime/infra','logs/system','logs/nodes/cloud','data','images','docker','bin'):
                    (root/relative).mkdir(parents=True,exist_ok=True)
                (root/'runtime/infra/.network_ready').touch()
                stub=root/'bin'/backend
                stub.write_text('#!'+sys.executable+'\n'+"""import json,os,sys
args=sys.argv[1:]
with open(os.environ['CALL_RECORD'],'a') as f:f.write(json.dumps(args)+'\\n')
if 'compose' in args and 'ps' in args:print('cloud')
elif '--writable-tmpfs' in args or '/app/scripts/ops/batch_worker.py' in args:sys.exit(7)
""")
                stub.chmod(0o755)
                probe=root/'bin/pgrep';probe.write_text('#!/bin/sh\nexit 0\n');probe.chmod(0o755)
                env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],
                    'CALL_RECORD':str(root/'calls'),'ZTA_BATCH_MANAGED':'1','ZTA_BATCH_ATTEMPT':'infra',
                    'ZTA_BATCH_SESSION':'/app/results/batch_archives/session','ZTA_BATCH_PHASE_FILE':str(root/'runtime/phase'),'USER':'test'}
                result=subprocess.run(['bash',str(root/f'scripts/simulation/deploy_code_sim_{backend}.sh')],env=env,capture_output=True,text=True,timeout=15)
                self.assertEqual(result.returncode,7,result.stdout+result.stderr)
                calls=[json.loads(l) for l in (root/'calls').read_text().splitlines()]
                self.assertEqual(len([c for c in calls if c[0]=='exec']),1)
                self.assertFalse(any('/app/scripts/setup/build_artifacts.py' in c for c in calls))
                launch=next(c for c in calls if c[0]=='exec')
                self.assertNotIn('-d',launch)
                self.assertIn('batch_worker.py',' '.join(launch))
                self.assertIn('/app/results/batch_archives/session',' '.join(launch))


@unittest.skipUnless(os.environ.get('ZTA_FLOWER_SMOKE') == '1', 'Opt-in local Flower/Ray process test')
class RealFlowerSmoke(unittest.TestCase):
    def test_two_real_runs_reuse_coordinator_with_fresh_processes(self):
        import time
        with TemporaryDirectory(prefix='zta-two-runs-') as directory, patch.dict(os.environ):
            root = Path(directory)
            os.environ['FLWR_TELEMETRY_ENABLED'] = '0'
            (root/'pyproject.toml').write_text('''[project]
name = "queue-smoke"
version = "1.0.0"
description = "Persistent coordinator check"
license = "Apache-2.0"
authors = [{name = "Test"}]
dependencies = ["flwr[simulation]==1.30.0"]
[tool.flwr.app]
publisher = "local"
[tool.flwr.app.components]
serverapp = "smoke:server_app"
clientapp = "smoke:client_app"
''')
            (root/'.gitignore').write_text('runtime/\n*.log\nobserved.jsonl\nflower_runtime.json\n')
            (root/'smoke.py').write_text('''import json, os
from pathlib import Path
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
COUNT = 0
class Client(NumPyClient):
    def fit(self, parameters, config):
        return [parameters[0] + 1], 1, {}
    def evaluate(self, parameters, config):
        return 0.0, 1, {}
def client_fn(context):
    return Client().to_client()
client_app = ClientApp(client_fn=client_fn)
def server_fn(context):
    global COUNT
    COUNT += 1
    marker = Path(os.environ['CONFIG_PATH'], 'marker').read_text()
    with Path('observed.jsonl').open('a') as f:
        f.write(json.dumps({'pid':os.getpid(),'count':COUNT,'marker':marker,'run':context.run_id})+'\\n')
    return ServerAppComponents(strategy=FedAvg(min_fit_clients=2,min_available_clients=2,min_evaluate_clients=2,initial_parameters=ndarrays_to_parameters([np.array([0.0])])),config=ServerConfig(num_rounds=1,round_timeout=30))
server_app = ServerApp(server_fn=server_fn)
''')
            flower = worker.Flower(root, root, 3)
            try:
                flower.ready(90)
                coordinator = flower.link.pid
                for marker in ('first', 'second'):
                    Path(os.environ['CONFIG_PATH'],'marker').write_text(marker)
                    submitted = flower.cli('run', str(root), 'local', '--federation-config',
                        'num-supernodes=2 client-resources-num-cpus=1 client-resources-num-gpus=0.0 init-args-num-cpus=2 init-args-num-gpus=0')
                    run_id = submitted['run-id']
                    deadline = time.monotonic()+120
                    while True:
                        row = flower.cli('list','local','--run-id',run_id)['runs'][0]
                        if row['status'].startswith('finished:'):
                            break
                        self.assertLess(time.monotonic(),deadline,'Smoke run timed out')
                        time.sleep(1)
                    self.assertEqual(row['status'],'finished:completed',row)
                    flower.archive_ray(root / marker)
                    self.assertEqual(flower.link.pid,coordinator)
                    self.assertIsNone(flower.link.poll())
                rows=[json.loads(line) for line in (root/'observed.jsonl').read_text().splitlines()]
                self.assertEqual([r['marker'] for r in rows],['first','second'])
                self.assertEqual([r['count'] for r in rows],[1,1])
                self.assertNotEqual(rows[0]['pid'],rows[1]['pid'])
            finally:
                flower.close()
            self.assertFalse(flower.scratch.exists())


if __name__ == "__main__":
    unittest.main()
