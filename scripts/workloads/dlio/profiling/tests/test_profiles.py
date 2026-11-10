"""Offline tests: no BeeGFS administration, SSH, DLIO or cluster execution."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROFILE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROFILE))
import analyze_profiles as analysis
import plots
import run_profiles as runner
from telemetry import Telemetry


class ParserTests(unittest.TestCase):
    def test_dxt_thread_and_eof_formats(self):
        text = """# DXT_POSIX module data
# DXT, file_id: 1, file_name: /data/a file
 X_POSIX 0 read 0 0 4096 1.0000 1.0010 123
 X_POSIX 0 read 1 4096 0 1.0020 1.0020 N/A
# DXT_MPIIO module data
 X_POSIX 0 write 0 0 999 2.0 3.0
"""
        events = list(analysis.dxt_events(text.splitlines()))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["thread"], "123")
        self.assertEqual(events[0]["file"], "/data/a file")
        self.assertEqual(events[1]["length"], 0)

    def test_clock_anchor_not_job_start_guess(self):
        anchor = {"pid": 1}
        markers = {}
        for kind, value in (("start", 2), ("end", 12)):
            anchor[kind] = {"path": kind, "wall_before_ns": int((1000+value)*1e9),
                            "wall_after_ns": int((1000+value+.001)*1e9)}
            markers[kind] = {"start": value+.0002, "end": value+.0008}
        offset, error = analysis.clock_offset(anchor, markers)
        self.assertAlmostEqual(offset, 1000)
        self.assertLess(error, .001)
        anchor["end"]["wall_before_ns"] += 1000000000
        anchor["end"]["wall_after_ns"] += 1000000000
        with self.assertRaisesRegex(ValueError, "drift"):
            analysis.clock_offset(anchor, markers)

    def test_command_matches_existing_workload_settings(self):
        root = PROFILE.parents[3]
        existing = {"unet": "runs-20261026-164248-962399",
                    "cosmoflow": "runs-20261103-013005-520206"}
        import yaml
        for name, directory in existing.items():
            native = yaml.safe_load((root/"results/workloads/runs"/directory/"rep1-pool1/.hydra/config.yaml").read_text())["workload"]
            command = runner.build_command(name, Path("/data"), Path("/checkpoint"), Path("/raw"))
            self.assertEqual(command[:4], ["mpirun", "-np", "8", "dlio_benchmark"])
            for section, fields in runner.settings(runner.shared.WORKLOADS[name]).items():
                for key,value in fields.items():
                    self.assertEqual(native[section][key], value, f"{name}: {section}.{key}")
                    if key != "num_samples_per_file":
                        self.assertIn(f"++workload.{section}.{key}={value}", command)
        self.assertEqual(len(runner.matrix("both")), 12)
        self.assertEqual([pool for _,suite,pool in runner.matrix("unet") if suite == "throttled"], [14,16,18])

    def test_interval_union_not_sum_of_concurrent_ranks(self):
        self.assertEqual(analysis.union([(0,2),(1,3),(3,4),(6,7)]), [(0,4),(6,7)])
        self.assertEqual(plots.intersections([(0,4)], [(2,3),(6,7)]), [(2,3)])

    def test_directory_pattern_is_not_an_actual_file_target_list(self):
        raw="Chunksize: 512K\nStorage Pool: 14\nNumber of storage targets: desired: 4"
        self.assertEqual(runner.validate_layout(raw,14,{104,205,305,306},directory=True),[])
        with self.assertRaisesRegex(RuntimeError,"four distinct"):
            runner.validate_layout(raw,14,{104,205,305,306})
        file="Chunksize: 512K\n"+"\n".join(f"+ {target} @ host" for target in (104,205,305,306))
        self.assertEqual(runner.validate_layout(file,14,{104,205,305,306}),[104,205,305,306])

    def test_epoch_partition_nests_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            (case/"raw/.hydra").mkdir(parents=True)
            origin = 1700000000
            def stamp(seconds):
                return datetime.fromtimestamp(origin+seconds, timezone.utc).replace(tzinfo=None).isoformat()
            native = {}
            for epoch in range(1,6):
                left = epoch*10
                native[str(epoch)] = {"start": stamp(left), "end": stamp(left+9)}
                if epoch in (2,4):
                    native[str(epoch)]["ckpt1"] = {"start":stamp(left+7),"end":stamp(left+8)}
            (case/"raw/per_epoch_stats.json").write_text(json.dumps(native))
            (case/"raw/summary.json").write_text(json.dumps({"start":stamp(1),"end":stamp(60),"num_accelerators":8}))
            profile = runner.shared.WORKLOADS["unet"]
            config = runner.settings(profile)
            import yaml
            (case/"raw/.hydra/config.yaml").write_text(yaml.safe_dump({"workload":config}))
            for rank in range(8):
                (case/f"raw/{rank}_output.json").write_text(json.dumps({str(epoch):{"load":{"block1":[.1]*5}} for epoch in range(1,6)}))
            anchors = [{"role":"rank","start":{"wall_before_ns":origin*10**9},"end":{"wall_after_ns":(origin+65)*10**9}}]
            _, phases, checkpoints, metrics = analysis.reconstruct(case,{"timezone_offset_s":0,"settings":config},anchors)
            self.assertAlmostEqual(sum(p["end_s"]-p["start_s"] for p in phases),65)
            self.assertEqual(len(checkpoints),2)
            self.assertEqual(metrics["total_processed_samples"],800)
            self.assertAlmostEqual(metrics["end_to_end_samples_s"],800/59)
            self.assertEqual(len([p for p in phases if p["phase"].startswith("C")]),0)


class CoverageIntegration(unittest.TestCase):
    """Use real local Darshan against tiny Python/fork files, not MPI/DLIO.

    The local installed runtime may itself link MPI; no MPI call is made in
    this test. The separate production preflight checks the active MPI stack
    and real rank/fork coverage before running any DLIO workload.
    """
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("PFS_TEST_DARSHAN_LIBRARY", "/usr/local/lib/libdarshan.so.0.0.0")
        if not Path(library).exists() or not shutil.which("darshan-dxt-parser"):
            raise unittest.SkipTest("Local Darshan runtime/utilities unavailable")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.case = Path(cls.temporary.name)/"case"
        cls.case.mkdir()
        data = cls.case/"smoke-data"
        data.mkdir()
        environment = runner.instrumentation(cls.case,data,data,Path(library),8,16)
        analysis.dump(cls.case/"case.json",{"smoke":True,"dataset":str(data),"checkpoints":str(data)})
        code = """import os,sys
from pathlib import Path
path=Path(sys.argv[1])/('data-'+os.environ['PMI_RANK'])
path.write_bytes(b'x'*4096)
assert path.read_bytes()==b'x'*4096
pid=os.fork()
if pid==0:
    assert path.read_bytes()==b'x'*4096
    os._exit(0)
_,status=os.waitpid(pid,0)
assert status==0
"""
        for rank in range(8):
            result=subprocess.run([sys.executable,"-c",code,str(data)],env={**os.environ,**environment,"PMI_RANK":str(rank)},capture_output=True,text=True,timeout=30)
            if result.returncode:
                raise AssertionError(result.stderr)
        cls.report=analysis.analyze_case(cls.case)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,"temporary"):
            cls.temporary.cleanup()

    def test_real_fork_and_clock_coverage(self):
        self.assertTrue(self.report["valid"])
        self.assertEqual(self.report["processes"],16)
        self.assertEqual(self.report["worker_processes"],8)
        self.assertLess(max(x["uncertainty_s"] for x in self.report["clock_alignment"].values()),.1)

    def test_dxt_shortfall_is_not_published_as_complete(self):
        def shortened(command, **kwargs):
            result = subprocess.run(command, **kwargs)
            if command[0] == "darshan-dxt-parser":
                kwargs["stdout"].flush()
                path = Path(kwargs["stdout"].name)
                lines = path.read_text().splitlines(keepends=True)
                data_file = False
                for index, line in enumerate(lines):
                    if "# DXT, file_id:" in line:
                        data_file = "/smoke-data/" in line
                    if data_file and line.lstrip().startswith("X_POSIX") and "read" in line:
                        del lines[index]
                        path.write_text("".join(lines))
                        break
            return result
        try:
            with self.assertRaisesRegex(ValueError,"count or byte mismatch"):
                analysis.analyze_case(self.case,execute=shortened)
            self.assertFalse(json.loads((self.case/"analysis/coverage.json").read_text())["valid"])
        finally:
            self.report = analysis.analyze_case(self.case)

    def test_missing_process_log_fails_closed(self):
        log=next((self.case/"darshan").glob("*.darshan"))
        renamed=log.with_suffix(".saved")
        log.rename(renamed)
        try:
            with self.assertRaisesRegex(ValueError,"Missing process logs"):
                analysis.analyze_case(self.case)
            self.assertFalse(json.loads((self.case/"analysis/coverage.json").read_text())["valid"])
        finally:
            renamed.rename(log)
            self.report=analysis.analyze_case(self.case)

    def test_complete_figures_and_exact_detail(self):
        # Give the toy capture native phase labels to test the chart pipeline.
        manifest=json.loads((self.case/"case.json").read_text())
        manifest.update(workload="unet",suite="test-fixture",pool=14)
        analysis.dump(self.case/"case.json",manifest)
        report=json.loads((self.case/"analysis/coverage.json").read_text())
        span=report["phases"][0]["end_s"]
        phases=[{"phase":f"E{epoch}","start_s":(epoch-1)*span/5,"end_s":epoch*span/5} for epoch in range(1,6)]
        checkpoints=[{"phase":f"C{epoch//2}","epoch":epoch,"start_s":(epoch-.2)*span/5,"end_s":(epoch-.1)*span/5} for epoch in (2,4)]
        report.update(chart_duration_s=span,end_to_end_samples_s=1,checkpoints=checkpoints,phases=phases)
        with sqlite3.connect(self.case/"analysis/events.sqlite") as db:
            for phase in phases:
                db.execute("UPDATE events SET phase=? WHERE end>=? AND end<=?",(phase["phase"],phase["start_s"],phase["end_s"]))
            analysis.phase_tables(db,phases,self.case/"analysis")
        analysis.dump(self.case/"analysis/coverage.json",report)
        plots.render([self.case],5)
        plots.comparison([self.case],self.case/"comparison")
        plots.detail(self.case,rank=0)
        for name in ("phase_access.png","phase_access.svg","overview.html","detail.html","size-bins.csv"):
            self.assertTrue((self.case/"analysis"/name).exists())


class OrchestrationTests(unittest.TestCase):
    def test_owned_process_cleanup_does_not_touch_another_study(self):
        with tempfile.TemporaryDirectory() as directory:
            owned=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],
                                   env={**os.environ,"PFS_DXT_CASE":directory+"/case"})
            other=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],
                                   env={**os.environ,"PFS_DXT_CASE":directory+"-other/case"})
            try:
                runner.terminate_owned(Path(directory))
                owned.wait(timeout=3)
                self.assertIsNone(other.poll())
            finally:
                if owned.poll() is None:
                    owned.kill()
                other.kill()
                owned.wait(); other.wait()

    def test_coverage_failure_restores_and_does_not_launch_next_case(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            args=argparse.Namespace(deadline=60,darshan_library=Path("/fake"),resume=None,workload="cosmoflow",dxt_mib=64,module_mib=16,preflight=False)
            with patch.object(runner,"source_identity",return_value={}),patch.object(runner,"preflight"), \
                    patch.object(runner,"verify_normal"),patch.object(runner,"discover",return_value={}), \
                    patch.object(runner,"normalize") as reset, \
                    patch.object(runner,"case_run",side_effect=ValueError("DXT truncation")) as launched:
                with self.assertRaisesRegex(ValueError,"truncation"):
                    runner.sweep(output,args)
                self.assertEqual(launched.call_count,1)
                reset.assert_called_once()
                state=json.loads((output/"state.json").read_text())
                self.assertEqual(len(state["cases"]),1)
                self.assertNotEqual(next(iter(state["cases"].values()))["status"],"complete")

    def test_existing_mpi_build_allowed_when_active_library_matches(self):
        result=subprocess.CompletedProcess([],0,json.dumps({"version":"MPICH existing", "libraries":["/existing/libmpi.so.12"]}))
        with patch.object(runner.shared,"execute",return_value=result):
            actual=runner.mpi_compatibility("libmpi.so.12 => /existing/libmpi.so.12",runner.shared.execute)
            self.assertEqual(actual["darshan_mpi_libraries"],["/existing/libmpi.so.12"])
            with self.assertRaisesRegex(RuntimeError,"different MPI"):
                runner.mpi_compatibility("libmpi.so.40 => /other/libmpi.so.40",runner.shared.execute)

    def test_previous_darshan_runtime_environment_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            library=Path(directory)/"libdarshan.so"
            library.touch()
            with patch.dict(os.environ,{"PFS_DARSHAN_LIBRARY":"", "DARSHAN_RUNTIME_LIB":str(library)}):
                self.assertEqual(runner.existing_runtime(),library.resolve())

    def test_preflight_timeout_is_not_reported_as_a_five_hour_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            library=root/"libdarshan.so"
            library.touch()
            args=argparse.Namespace(darshan_library=library,dxt_mib=8,module_mib=16)
            def execute(command,**kwargs):
                if command[0]=="mpirun":
                    raise runner.shared.DeadlineReached("probe budget")
                return subprocess.CompletedProcess(command,0,"ext4" if command[0]=="findmnt" else "libc.so.6")
            with patch.object(shutil,"which",return_value="tool"),patch.object(runner.shared,"execute",side_effect=execute), \
                    patch.object(runner.shared,"DEADLINE",None),patch.object(runner,"mpi_compatibility",return_value={}), \
                    patch.object(runner,"terminate_owned"):
                with self.assertRaisesRegex(RuntimeError,"preflight timed out"):
                    runner.preflight(root,args)
                self.assertIsNone(runner.shared.DEADLINE)

    def test_unsupported_version_flag_does_not_fail_successful_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            library=root/"libdarshan.so"
            library.touch()
            for tool in ("mpirun","darshan-parser","darshan-dxt-parser"):
                (root/tool).write_text("existing installed tool")
            args=argparse.Namespace(darshan_library=library,dxt_mib=8,module_mib=16)
            def execute(command,**kwargs):
                if command[:2]==["darshan-parser","--version"]:
                    if kwargs.get("check"):
                        raise subprocess.CalledProcessError(1,command)
                    return subprocess.CompletedProcess(command,1,"","unrecognized option --version")
                return subprocess.CompletedProcess(command,0,"ext4" if command[0]=="findmnt" else "existing version","")
            with patch.object(shutil,"which",side_effect=lambda tool:str(root/tool)), \
                    patch.object(runner.shared,"execute",side_effect=execute), \
                    patch.object(runner.shared,"DEADLINE",None),patch.object(runner,"mpi_compatibility",return_value={}), \
                    patch.object(runner,"terminate_owned"),patch.object(runner,"analyze_case") as captured:
                runner.preflight(root,args)
                captured.assert_called_once()
            recorded=json.loads(next(root.glob("preflight-*/provenance.json")).read_text())["darshan-parser"]
            self.assertEqual(recorded["returncode"],1)
            self.assertIn("unrecognized",recorded["stderr"])
            self.assertEqual(recorded["sha256"],runner.sha(root/"darshan-parser"))

    def test_resume_skips_complete_cases_and_restores_before_next_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            matrix=runner.matrix("both")
            key=lambda item:f"{item[0]}-{item[1]}-pool{item[2]}"
            state={"identity":{},"trace_memory":[64,16],"matrix":matrix,
                   "cases":{key(item):{"status":"complete","attempt":"previous"} for item in matrix[:4]},"status":"interrupted"}
            analysis.dump(output/"state.json",state)
            args=argparse.Namespace(deadline=60,darshan_library=Path("/fake"),resume=output,workload="both",dxt_mib=64,module_mib=16,preflight=False)
            with patch.object(runner,"source_identity",return_value={}),patch.object(runner,"preflight"), \
                    patch.object(runner,"verify_normal"),patch.object(runner,"discover",return_value={}), \
                    patch.object(runner,"normalize") as reset,patch.object(runner,"transition") as moved, \
                    patch.object(runner,"restore") as restored,patch.object(runner,"case_run") as launched:
                runner.sweep(output,args)
                self.assertEqual(launched.call_count,8)
                self.assertEqual([(item.args[2],item.args[3]) for item in launched.call_args_list[:2]],[("throttled",16),("throttled",18)])
                self.assertEqual(moved.call_args_list[0].args,(6,16))
                restored.assert_called_once_with(18)
                self.assertEqual(reset.call_count,2)
                self.assertEqual(json.loads((output/"state.json").read_text())["status"],"complete")


class TelemetryTests(unittest.TestCase):
    def test_local_sampler_lifecycle_without_io_load(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory)
            (case/"anchors").mkdir()
            sampler=Telemetry(case,{"client":{"targets":{},"interfaces":["lo"]}},set(),subprocess.run,.02)
            sampler.start()
            self.assertEqual(sampler.stop(),[])
            with (case/"telemetry/client.jsonl").open() as file:
                lines=[json.loads(line) for line in file]
            self.assertGreaterEqual(len(lines),2)
            self.assertIn("lo",lines[0]["interfaces"])

    def test_boundary_intervals_are_not_invented(self):
        with tempfile.TemporaryDirectory() as directory:
            case=Path(directory)
            (case/"telemetry").mkdir(); (case/"analysis").mkdir()
            anchor={"offset_s":0,"uncertainty_s":0}
            analysis.dump(case/"telemetry/clocks.json",{"client":{"before":anchor,"after":anchor}})
            with (case/"telemetry/client.jsonl").open("w") as file:
                for second in range(4):
                    file.write(json.dumps({"wall_s":1000+second,"monotonic_s":second,
                        "devices":{"test":{"read_bytes":100*second,"write_bytes":0}},"interfaces":{}})+"\n")
            coverage={"clock_origin_wall_s":1000,"phases":[{"phase":"E1","start_s":.25,"end_s":2.75}]}
            rows=plots.telemetry_phases(case,coverage)
            self.assertEqual(rows[0]["read_or_rx_bytes"],100)
            self.assertEqual(rows[0]["observed_interval_s"],1)
            self.assertAlmostEqual(rows[0]["phase_coverage_fraction"],.4)


if __name__ == "__main__":
    unittest.main()
