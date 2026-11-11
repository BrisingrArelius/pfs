"""Exercise the real persistent DataLoader lifecycle without MPI or DLIO."""

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import dlio_darshan as darshan


class PersistentReaderTests(unittest.TestCase):
    def test_pool_is_closed_before_profiler_stop_and_logs_are_complete(self):
        library=Path(os.environ.get("PFS_TEST_DARSHAN_LIBRARY","/usr/local/lib/libdarshan.so.0.0.0"))
        if not library.is_file() or not importlib.util.find_spec("torch"):
            self.skipTest("Installed local Darshan and PyTorch required")
        if not all(shutil.which(tool) for tool in ("darshan-parser","darshan-dxt-parser")):
            self.skipTest("Installed Darshan parsers required")
        code="""
import os, sys, types
from pathlib import Path
import numpy as np
import torch
import dlio_darshan as d
data, checkpoints = map(Path,sys.argv[1:3])
class Profiler:
    def __init__(self):
        self._args=types.SimpleNamespace(framework='pytorch')
        os.environ['DARSHAN_LOGFILE']=str(data/'colliding.darshan')
    def start(self):
        os.environ['DARSHAN_DISABLE']='0'
    def stop(self):
        assert all(not worker.is_alive() for worker in loader._iterator._workers)
        os.environ['DARSHAN_DISABLE']='1'
module=types.ModuleType('dlio_benchmark.profiler.darshan_profiler')
module.DarshanProfiler=Profiler
sys.modules[module.__name__]=module
class Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 21
    def __getitem__(self,index):
        with np.load(data/f'{index}.npz') as sample:
            return sample['sample'].shape[0]
class Framework:
    @staticmethod
    def get_instance(profiling):
        return Framework()
    def get_loader(self):
        return types.SimpleNamespace(_dataset=loader)
module=types.ModuleType('dlio_benchmark.framework.torch_framework')
module.TorchFramework=Framework
sys.modules[module.__name__]=module
def run():
    global loader
    profiler=Profiler()
    profiler.start()
    loader=torch.utils.data.DataLoader(Dataset(),batch_size=4,num_workers=4,
        persistent_workers=True,prefetch_factor=1,multiprocessing_context='fork',
        pin_memory=True,drop_last=True)
    for epoch in range(2):
        for step,batch in enumerate(loader):
            if step==1:
                break
    (checkpoints/'model.pt').write_bytes(b'checkpoint')
    profiler.stop()
class Entry:
    def load(self):
        return run
d.entry_points=lambda **kwargs:[Entry()]
d.main()
"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data,checkpoints=root/"data",root/"checkpoints"
            data.mkdir()
            checkpoints.mkdir()
            import numpy as np
            for index in range(21):
                np.savez(data/f"{index}.npz",sample=np.zeros((16,16,16),dtype=np.float32))
            command=darshan.instrument(["mpirun","-np","1","dlio_benchmark"],root,data,checkpoints,library)
            environment=darshan.clean_environment(os.environ)
            environment.update(dict(value.split("=",1) for value in command[8:command.index(sys.executable)]))
            environment.update(PYTHONPATH=str(Path(darshan.__file__).parent),PMI_RANK="0")
            result=subprocess.run([sys.executable,"-c",code,str(data),str(checkpoints)],
                                  env=environment,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(list((root/"darshan").glob("worker-*.json"))),4)
            self.assertEqual(list((root/"darshan").rglob("*.darshan_partial")),[])
            darshan.check_logs(root,ranks=1)


if __name__=="__main__":
    unittest.main()
