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


def configuration_module(common):
    path=Path(__file__).resolve().parents[1]/"throttled_ssd/configure_pool.py"
    spec=importlib.util.spec_from_file_location("configure_pool_test",path)
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{"pool_common":common}):
        spec.loader.exec_module(module)
    return module


class RepetitionTests(unittest.TestCase):
    def test_all_caps_use_physical_devices_and_clear_stale_partition_caps(self):
        configure=configuration_module(throttled_module())
        devices={104:("colva1","/dev/nvme1n1"),205:("colva2","/dev/nvme1n1"),
                 305:("colva3","/dev/nvme1n1"),306:("colva3","/dev/nvme2n1")}
        self.assertEqual(configure.DEVICES,devices)
        for pool,rates in ((14,{104:45,205:45,305:45,306:45}),
                           (16,{104:45,205:45,305:15,306:15}),
                           (18,{104:15,205:15,305:15,306:15})):
            with self.subTest(pool=pool),patch.object(configure,"remote") as remote:
                configure.apply_limits(pool)
                for host in ("colva1","colva2","colva3"):
                    calls=[call.args for call in remote.call_args_list if call.args[0]==host]
                    self.assertEqual(len(calls),2)
                    self.assertEqual(calls[0][-2:],("IOReadBandwidthMax=","IOWriteBandwidthMax="))
                    for target,(node,device) in devices.items():
                        if node==host:
                            for direction in ("IOReadBandwidthMax","IOWriteBandwidthMax"):
                                self.assertIn(f"{direction}={device} {rates[target]*1048576}",calls[1])
                    self.assertFalse(any("n1p1" in arg for arg in calls[1]))

    def test_missing_kernel_cap_is_still_rejected_when_systemd_matches(self):
        common=throttled_module()
        def execute(command,**kwargs):
            host,text=command[1:3]
            if "IOReadBandwidthMax" in text:
                output="IOReadBandwidthMax=/dev/nvme1n1 47185920\nIOWriteBandwidthMax=/dev/nvme1n1 47185920\n"
            elif "/sys/class/block/" in text:
                output="259:0\n" if host=="colva1" else "259:7\n"
            else:
                output="259:0 rbps=47185920 wbps=47185920\n" if host=="colva1" else ""
            return SimpleNamespace(stdout=output)
        with patch.object(common,"beegfs",return_value="14 dlio_ssd_f4_s0 104,205,305,306"), \
             patch.object(common,"execute",side_effect=execute):
            with self.assertRaisesRegex(RuntimeError,"colva2 target 205: io.max 259:7"):
                common.verify(14)

    def test_restore_from_pool14_does_not_require_working_caps(self):
        common=throttled_module()
        configure=configuration_module(common)
        members={14:set(common.TARGETS),6:configure.NORMAL_TARGETS-set(common.TARGETS)}
        def beegfs(*args):
            if args==("--liststoragepools",):
                return "\n".join(f"{pool} {name} {','.join(map(str,sorted(members[pool])))}"
                                 for pool,name in ((6,"ssd"),(14,"dlio_ssd_f4_s0")))
            pool=int(args[1].split("=")[1])
            if args[2].startswith("--rmtargets="):
                members[pool].difference_update(common.TARGETS)
            elif args[2].startswith("--addtargets="):
                members[pool].update(common.TARGETS)
            else:
                raise AssertionError(args)
            return ""
        with patch.object(common,"beegfs",side_effect=beegfs), \
             patch.object(configure,"beegfs",side_effect=beegfs), \
             patch.object(configure,"verify",side_effect=RuntimeError("broken io.max")) as caps, \
             patch.object(configure,"remote",return_value="IOReadBandwidthMax=\nIOWriteBandwidthMax=\n") as remote, \
             redirect_stdout(StringIO()):
            configure.restore(14)
        caps.assert_not_called()
        self.assertEqual(members[6],configure.NORMAL_TARGETS)
        self.assertFalse(members[14])
        cleared=[call.args[0] for call in remote.call_args_list if call.args[1]=="sudo"]
        self.assertEqual(cleared,["colva1","colva2","colva3"])

    def test_actual_normal_and_throttled_loops(self):
        throttled=throttled_module()
        profile=SimpleNamespace(folder="cosmoflow",profile="cosmoflow_v100",files=1,
                                batch_size=1,reader_threads=4,model_bytes=100)
        for module in (normal,throttled):
            cases=[(1,False,1,None),(3,False,3,None),(3,True,1,None)]
            if module is normal:
                cases.extend((1,False,1,pool) for pool in (1,3,6))
            for count,pilot,expected,selected_pool in cases:
                with self.subTest(suite=module.__name__,count=count,pilot=pilot,pool=selected_pool), \
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
                        module.run(profile,"checkpoint.Class",100,pilot=pilot,repetitions=count,pool=selected_pool)
                        expected*=3 if not pilot and selected_pool is None else 1
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
                        if selected_pool is not None:
                            self.assertTrue(any(f"/pool{selected_pool}/dataset" in arg for arg in command))

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
