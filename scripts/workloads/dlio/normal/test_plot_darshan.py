"""Offline pilot-plot checks using native parser-format and DLIO fixtures."""

import importlib.util
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if importlib.util.find_spec("matplotlib") and importlib.util.find_spec("yaml"):
    import plot_darshan as plot
else:
    plot=None


@unittest.skipIf(plot is None,"Matplotlib and PyYAML required for plotting")
class PilotPlotTests(unittest.TestCase):
    def fixture(self, root):
        (root/".hydra").mkdir()
        (root/"darshan").mkdir()
        (root/".hydra/config.yaml").write_text(json.dumps({"workload":{
            "model":"unet3d","dataset":{"data_folder":"/data"},
            "checkpoint":{"checkpoint_folder":"/ckpt"}}}))
        (root/"summary.json").write_text(json.dumps({"num_accelerators":1}))
        (root/"per_epoch_stats.json").write_text(json.dumps({
            "1":{"start":"2026-11-11T00:00:00","end":"2026-11-11T00:00:05"},
            "2":{"start":"2026-11-11T00:00:05","end":"2026-11-11T00:00:11",
                 "ckpt1":{"start":"2026-11-11T00:00:09","end":"2026-11-11T00:00:10"}}}))
        for name,pid in (("rank-0",100),("worker-200",200)):
            (root/"darshan"/f"{name}.json").write_text(json.dumps({"rank":0,"pid":pid}))
        for pid in (100,200,300):
            (root/"darshan"/f"test_id100-{pid}_1.darshan").touch()

    @staticmethod
    def parsed(tool, path):
        if "-100_" in path.name:
            text=("POSIX 0 1 POSIX_BYTES_WRITTEN 512 /ckpt/model.pt / fs\n"
                  "STDIO 0 2 STDIO_BYTES_WRITTEN 32 /ckpt/index / fs\n") if tool=="darshan-parser" else (
                  "# DXT, file_id: 1, file_name: /ckpt/model.pt\n"
                  "X_POSIX 0 write 0 0 512 10.0 10.1\n")
        elif "-200_" in path.name:
            text="POSIX 0 3 POSIX_BYTES_READ 8192 /data/sample.npz / fs\n" if tool=="darshan-parser" else (
                "# DXT, file_id: 3, file_name: /data/sample.npz\n"
                "X_POSIX 0 read 0 0 8192 1.0 1.1 42\n"
                "X_POSIX 0 read 1 8192 0 1.1 1.1 N/A\n")
        else:
            text=""
        if tool=="darshan-parser":
            start=datetime(2026,11,11,tzinfo=timezone.utc).timestamp()
            text=f"# start_time: {int(start)-(1 if '-100_' in path.name else 0)}\n"+text
        return iter(text.splitlines())

    def test_readers_stdio_and_nested_checkpoint_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            with patch.object(plot.dlio_darshan,"check_logs"),patch.object(plot,"parser_lines",side_effect=self.parsed):
                data=plot.collect(root,native_utc_offset=0)
            self.assertEqual(data["rank_api_bytes"][0],{
                "POSIX_BYTES_WRITTEN":512,"STDIO_BYTES_WRITTEN":32,"POSIX_BYTES_READ":8192})
            self.assertEqual(data["dxt_request_sizes"]["read"],{0:1,8192:1})
            self.assertEqual(data["dxt_request_sizes"]["write"],{512:1})
            self.assertEqual(data["phases"][1]["end_s"],11)
            self.assertEqual(data["phases"][1]["rank0_checkpoints"],[[9,10]])
            self.assertEqual(next(event for event in data["events"] if event["op"]=="write")["start_s"],9)
            self.assertEqual(next(event for event in data["events"] if event["op"]=="read")["thread"],"42")
            output=root/"figures"
            plot.render(data,output)
            image=plot.plt.imread(output/"pilot-darshan.png")
            self.assertGreater(image.shape[0],100)
            self.assertIn("<svg",(output/"pilot-darshan.svg").read_text())
            saved=json.loads((output/"pilot-darshan.json").read_text())
            self.assertEqual(saved["rank_api_bytes"],data["rank_api_bytes"])
            self.assertEqual(sum(item["count"] for item in saved["request_time_bins"]["counts"]),3)
            for name in ("A-phase-activity","B-rank-activity","C-request-sizes"):
                self.assertTrue((output/f"{name}.png").is_file())
                self.assertIn("COARSE",(output/f"{name}.svg").read_text())

    def test_unknown_process_with_measured_io_is_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.fixture(root)
            def parsed(tool,path):
                if "-300_" in path.name and tool=="darshan-parser":
                    return iter(["POSIX 0 4 POSIX_BYTES_READ 100 /data/unknown.npz / fs"])
                return self.parsed(tool,path)
            with patch.object(plot.dlio_darshan,"check_logs"),patch.object(plot,"parser_lines",side_effect=parsed):
                with self.assertRaisesRegex(ValueError,"unmapped process"):
                    plot.collect(root)


if __name__=="__main__":
    unittest.main()
