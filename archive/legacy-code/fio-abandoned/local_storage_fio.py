#!/usr/bin/env python3
"""FIO job generation and execution for one inventoried local target filesystem."""

import json
import os
from pathlib import Path


def write_json_atomic(path, data):
    """Replace a JSON artifact only after its complete content reaches disk."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as output:
        json.dump(data, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def write_text_atomic(path, data):
    """Replace a text artifact only after its complete content reaches disk."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def cleanup_data_files(work_dir, job_name):
    """Remove only data files created for one named FIO configuration."""
    for path in Path(work_dir).glob(f"{job_name}.*"):
        if path.is_file():
            path.unlink()


def prepare_read_files(
    job_name, num_files, size, block_size, io_depth, num_jobs,
    work_dir, artifact_dir, run_command
):
    """Write the complete read dataset once and retain its native FIO evidence."""
    fio_text = f"""
[global]
ioengine=libaio
direct=1
fallocate=none
group_reporting=1
nrfiles={num_files}
size={size}
iodepth={io_depth}
numjobs={num_jobs}
directory={work_dir}
filename_format={job_name}.$jobnum.$filenum

[{job_name}_prepare]
rw=write
bs={block_size}
end_fsync=1
"""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    job_file = artifact_dir / "prepare.fio"
    write_text_atomic(job_file, fio_text)
    cleanup_data_files(work_dir, job_name)

    try:
        result = run_command(["fio", str(job_file), "--output-format=json"])
        write_json_atomic(artifact_dir / "prepare.json", json.loads(result.stdout))
    except Exception as error:
        if getattr(error, "stdout", None):
            write_text_atomic(artifact_dir / "prepare.stdout.txt", error.stdout)
        if getattr(error, "stderr", None):
            write_text_atomic(artifact_dir / "prepare.stderr.txt", error.stderr)
        cleanup_data_files(work_dir, job_name)
        raise


def run_fio(
    job_name, run_index, num_files, size, mode, block_size, io_depth,
    num_jobs, time_based, runtime, ramp_time, work_dir, artifact_dir,
    run_command, cleanup_files
):
    """Run one measured FIO repetition and retain its job file and native JSON."""
    timing = ""
    if time_based:
        timing = f"time_based=1\nruntime={runtime}\nramp_time={ramp_time}"
    fio_text = f"""
[global]
ioengine=libaio
direct=1
fallocate=none
group_reporting=1
{timing}
nrfiles={num_files}
size={size}
iodepth={io_depth}
numjobs={num_jobs}
directory={work_dir}
filename_format={job_name}.$jobnum.$filenum

[{job_name}]
rw={mode}
bs={block_size}
"""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    job_file = artifact_dir / f"run_{run_index}.fio"
    write_text_atomic(job_file, fio_text)

    try:
        result = run_command(["fio", str(job_file), "--output-format=json"])
    except Exception as error:
        if getattr(error, "stdout", None):
            write_text_atomic(artifact_dir / f"run_{run_index}.stdout.txt", error.stdout)
        if getattr(error, "stderr", None):
            write_text_atomic(artifact_dir / f"run_{run_index}.stderr.txt", error.stderr)
        raise

    data = json.loads(result.stdout)
    write_json_atomic(artifact_dir / f"run_{run_index}.json", data)
    if cleanup_files:
        cleanup_data_files(work_dir, job_name)
    return data["jobs"][0]
