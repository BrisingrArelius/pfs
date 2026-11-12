"""Offline tests for the combined reservation study; no cluster operations."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_repetitions import configuration_module, throttled_module


def load():
    common = throttled_module()
    configure = configuration_module(common)
    path = Path(__file__).resolve().parents[1]/"throttled_ssd/run_study.py"
    spec = importlib.util.spec_from_file_location("test_combined_study",path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{"pool_common":common,"configure_pool":configure}), \
         patch.object(sys,"path",list(sys.path)):
        sys.path.insert(0,str(path.parent))
        spec.loader.exec_module(module)
    return module


class StudyTests(unittest.TestCase):
    def test_duration_formats(self):
        study=load()
        for value, expected in (("6hours",21600),("6h",21600),("90m",5400),("2 hours",7200),("30",30)):
            self.assertEqual(study.duration(value),expected)
        for value in ("0","6weeks","garbage"):
            with self.assertRaises(study.argparse.ArgumentTypeError):
                study.duration(value)

    def test_shared_deadline_resume_order_and_restore(self):
        study=load()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            base=Path(directory)
            active=[6]
            history=[]
            completed=[]
            first=[True]
            identity={"dlio_version":"2.0.0","rate":"45/15"}
            stack.enter_context(patch.object(study,"identity",return_value=identity))
            stack.enter_context(patch.object(study,"verify_normal"))
            stack.enter_context(patch.object(study.common,"stop_owned"))
            def normalize(identifier):
                history.append(("normalize",active[0]))
                active[0]=6
            def transition(old,new):
                self.assertEqual(active[0],old)
                history.append(("transition",old,new))
                active[0]=new
            def restore(pool):
                self.assertEqual(active[0],pool)
                history.append(("restore",pool))
                active[0]=6
            stack.enter_context(patch.object(study,"normalize",side_effect=normalize))
            stack.enter_context(patch.object(study,"transition",side_effect=transition))
            stack.enter_context(patch.object(study,"restore",side_effect=restore))
            def run(name,pool,**kwargs):
                self.assertEqual(kwargs["darshan"],Path("/runtime/libdarshan.so"))
                for repetition in (1,2,3):
                    if repetition in kwargs["completed"]:
                        continue
                    if first[0] and name=="unet" and pool==14 and repetition==2:
                        first[0]=False
                        raise study.common.DeadlineReached("test deadline")
                    completed.append((name,pool,repetition))
                    kwargs["on_complete"](repetition)
            stack.enter_context(patch.object(study.common,"run",side_effect=run))
            self.assertEqual(study.sweep(base,Path("/runtime/libdarshan.so"),3600),75)
            journal=next(base.glob("study-*.json"))
            state=json.loads(journal.read_text())
            self.assertEqual(state["status"],"interrupted")
            self.assertEqual(state["workloads"]["unet"]["pools"]["14"]["completed"],[1])
            self.assertEqual(active[0],6)
            self.assertEqual(study.sweep(base,Path("/runtime/libdarshan.so"),3600,resume=True),0)
            self.assertEqual(json.loads(journal.read_text())["status"],"complete")
            self.assertEqual(len(completed),18)
            self.assertEqual(len(set(completed)),18)
            self.assertEqual([item[0] for item in completed], ["cosmoflow"]*9+["unet"]*9)
            self.assertEqual(active[0],6)
            self.assertIn(("restore",18),history)

    def test_identity_mismatch_refuses_resume_before_pool_operations(self):
        study=load()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            base=Path(directory)
            state={"status":"interrupted","identity":{"old":True}}
            (base/"study-old.json").write_text(json.dumps(state))
            stack.enter_context(patch.object(study,"identity",return_value={"dlio_version":"2.0.0","new":True}))
            normalize=stack.enter_context(patch.object(study,"normalize"))
            with self.assertRaisesRegex(RuntimeError,"settings changed"):
                study.sweep(base,Path("/runtime/libdarshan.so"),3600,resume=True)
            normalize.assert_not_called()
            study.common.DEADLINE=None
            study.common.MANAGED=False

    def test_cosmoflow_consumes_shared_budget_before_unet(self):
        study=load()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            base=Path(directory)
            clock=[100.0]
            active=[6]
            calls=[]
            stack.enter_context(patch.object(study.time,"monotonic",side_effect=lambda:clock[0]))
            stack.enter_context(patch.object(study,"identity",return_value={"dlio_version":"2.0.0"}))
            stack.enter_context(patch.object(study,"verify_normal"))
            stack.enter_context(patch.object(study.common,"stop_owned"))
            stack.enter_context(patch.object(study,"normalize",side_effect=lambda identifier:active.__setitem__(0,6)))
            stack.enter_context(patch.object(study,"restore",side_effect=lambda pool:active.__setitem__(0,6)))
            stack.enter_context(patch.object(study,"transition",side_effect=lambda old,new:active.__setitem__(0,new)))
            def run(name,pool,**kwargs):
                calls.append((name,pool,study.common.DEADLINE))
                for repetition in (1,2,3):
                    kwargs["on_complete"](repetition)
                if (name,pool)==("cosmoflow",18):
                    clock[0]=study.common.DEADLINE+1
            stack.enter_context(patch.object(study.common,"run",side_effect=run))
            self.assertEqual(study.sweep(base,Path("/runtime/libdarshan.so"),3600),75)
            self.assertEqual([(name,pool) for name,pool,_ in calls],[("cosmoflow",14),("cosmoflow",16),("cosmoflow",18)])
            self.assertEqual(len({deadline for _,_,deadline in calls}),1)
            state=json.loads(next(base.glob("study-*.json")).read_text())
            self.assertEqual(state["workloads"]["unet"]["pools"],{})
            self.assertEqual(active[0],6)

    def test_unexpected_failure_still_attempts_restoration(self):
        study=load()
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            base=Path(directory)
            stack.enter_context(patch.object(study,"identity",return_value={"dlio_version":"2.0.0"}))
            stack.enter_context(patch.object(study,"verify_normal"))
            normalize=stack.enter_context(patch.object(study,"normalize"))
            stack.enter_context(patch.object(study,"transition",side_effect=RuntimeError("cap failure")))
            stack.enter_context(patch.object(study.common,"stop_owned"))
            with self.assertRaisesRegex(RuntimeError,"cap failure"):
                study.sweep(base,Path("/runtime/libdarshan.so"),3600)
            self.assertEqual(normalize.call_count,2)
            self.assertEqual(json.loads(next(base.glob("study-*.json")).read_text())["status"],"interrupted")

    def test_owned_cleanup_does_not_kill_unrelated_process(self):
        study=load()
        identifier="study-test-"+str(time.time_ns())
        import os
        with subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"],
                              env={**os.environ,"PFS_DLIO_STUDY_ID":identifier}) as owned, \
             subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]) as other:
            try:
                limit=time.monotonic()+3
                while owned.pid not in study.common.owned_processes(identifier) and time.monotonic()<limit:
                    time.sleep(.01)
                self.assertIn(owned.pid,study.common.owned_processes(identifier))
                study.common.stop_owned(identifier)
                owned.wait(timeout=3)
                self.assertIsNone(other.poll())
            finally:
                if other.poll() is None:
                    other.terminate()
                    other.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
