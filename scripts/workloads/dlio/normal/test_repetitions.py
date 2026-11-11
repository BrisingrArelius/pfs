"""Execute repetition dispatch with every cluster operation mocked."""

from contextlib import ExitStack, redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pool_common as normal


def throttled_module():
    path=Path(__file__).resolve().parents[1]/"throttled_ssd/pool_common.py"
    spec=importlib.util.spec_from_file_location("throttled_repetition_test",path)
    module=importlib.util.module_from_spec(spec)
    with patch.object(sys,"path",list(sys.path)):
        spec.loader.exec_module(module)
    return module


class RepetitionTests(unittest.TestCase):
    def test_actual_normal_and_throttled_loops(self):
        throttled=throttled_module()
        profile=SimpleNamespace(folder="cosmoflow",profile="cosmoflow_v100",files=1,
                                batch_size=1,reader_threads=4,model_bytes=100)
        for module in (normal,throttled):
            for count,pilot,expected in ((1,False,1),(3,False,3),(3,True,1)):
                with self.subTest(suite=module.__name__,count=count,pilot=pilot), \
                     tempfile.TemporaryDirectory() as directory,ExitStack() as stack,redirect_stdout(StringIO()):
                    root=Path(directory)
                    data=root/"data"
                    for pool in (1,3,6):
                        (data/f"cosmoflow/pool{pool}/dataset/valid").mkdir(parents=True)
                        (data/f"cosmoflow/pool{pool}/checkpoints").mkdir()
                    (data/"throttled/checkpoints").mkdir(parents=True)
                    commands=[]
                    def execute(command,**kwargs):
                        if command[0]=="mpirun":
                            commands.append(command)
                            raw=Path(next(arg.split("=",1)[1] for arg in command if arg.startswith("hydra.run.dir=")))
                            (raw/"per_epoch_stats.json").write_text(json.dumps({"1":{},"2":{"ckpt1":{}}}))
                    stack.enter_context(patch.object(module.importlib.metadata,"version",return_value="2.0.0"))
                    stack.enter_context(patch.object(module.time,"sleep"))
                    stack.enter_context(patch.object(module,"checkpoint_sizes",return_value={}))
                    stack.enter_context(patch.object(module,"placement",return_value="104,205,305,306"))
                    stack.enter_context(patch.object(module,"metrics",return_value=[]))
                    if module is normal:
                        stack.enter_context(patch.object(module,"DATA_ROOT",data))
                        stack.enter_context(patch.object(module,"RESULT_ROOT",root/"results"))
                        stack.enter_context(patch.object(module,"data_files",return_value=[data/"input"]))
                        stack.enter_context(patch.object(module.subprocess,"run",side_effect=execute))
                        module.run(profile,"checkpoint.Class",100,pilot=pilot,repetitions=count)
                        expected*=3 if not pilot else 1
                    else:
                        stack.enter_context(patch.dict(module.WORKLOADS,{"cosmoflow":profile}))
                        stack.enter_context(patch.object(module,"verify"))
                        stack.enter_context(patch.object(module,"prepare",return_value=data/"throttled"))
                        stack.enter_context(patch.object(module,"execute",side_effect=execute))
                        stack.enter_context(patch.object(module,"physical_counters",return_value={}))
                        stack.enter_context(patch.object(module,"physical_io",return_value=[]))
                        module.run("cosmoflow",14,pilot=pilot,output=root/"results",repetitions=count)
                    self.assertEqual(len(commands),expected)
                    for command in commands:
                        self.assertIn(f"++workload.train.epochs={2 if pilot else 5}",command)
                        self.assertIn("++workload.checkpoint.type=all_ranks",command)

    def test_throttled_selected_pools_forward_one_invocation_and_restore(self):
        module=throttled_module()
        for name in ("cosmoflow","unet"):
            with self.subTest(workload=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                source=root/f"{name}/pool1/dataset/train"
                source.mkdir(parents=True)
                (source/"sample.test").touch()
                profile=SimpleNamespace(folder=name,files=1,extension="test")
                configure=SimpleNamespace(transition=Mock(),restore=Mock())
                with patch.dict(sys.modules,{"configure_pool":configure}), \
                     patch.object(module,"DATA_ROOT",root), \
                     patch.dict(module.WORKLOADS,{name:profile}), \
                     patch.object(module,"run") as run,redirect_stdout(StringIO()):
                    module.run_all(name,repetitions=1,pools=[14,16,18])
                self.assertEqual([call.args[:2] for call in run.call_args_list],[(name,14),(name,16),(name,18)])
                self.assertTrue(all(call.kwargs["repetitions"]==1 for call in run.call_args_list))
                self.assertEqual([call.args for call in configure.transition.call_args_list],[(6,14),(14,16),(16,18)])
                configure.restore.assert_called_once_with(18)


if __name__=="__main__":
    unittest.main()
