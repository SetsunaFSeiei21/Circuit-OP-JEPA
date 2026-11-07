#!/usr/bin/env python3
"""Import one raw circuit and run its final_exam pin-level comparison inside one allocation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import traceback
import uuid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = PROJECT_ROOT / "final_exam" / "single_circuit_pin"
sys.path.insert(0, str(PROJECT_ROOT))


def directory_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("Expected a directory name, without path separators")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    """Update this launcher's metadata atomically after reserving its new directory."""
    temporary = path.with_name(path.stem + "_tmp_pin" + path.suffix)
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def new_run_paths(run_id):
    paths = {name: (TASK_ROOT / (name + "_pin") / run_id).resolve()
             for name in ("results", "logs", "hpc_logs")}
    for name, path in paths.items():
        if not path.is_relative_to(TASK_ROOT.resolve()):
            raise ValueError(f"{name} directory must remain under {TASK_ROOT}")
    # Slurm's output directory is created before submission. Training and logs
    # are claimed here and must be fresh even when an old directory is empty.
    for name in ("results", "logs"):
        if paths[name].exists():
            raise ValueError(f"Refusing an existing {name} directory: {paths[name]}")
    return paths


def run_logged(command, log, lock):
    """Tee both streams live, preserving stderr for Slurm's separate error log."""
    print("+ " + shlex.join(list(map(str, command))), flush=True)
    log.write("+ " + shlex.join(list(map(str, command))) + "\n")
    log.flush()
    process = subprocess.Popen(
        list(map(str, command)), cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    def copy_stream(source, destination):
        try:
            for line in source:
                with lock:
                    destination.write(line)
                    destination.flush()
                    log.write(line)
                    log.flush()
        finally:
            source.close()

    threads = [
        threading.Thread(target=copy_stream, args=(process.stdout, sys.stdout)),
        threading.Thread(target=copy_stream, args=(process.stderr, sys.stderr)),
    ]
    for thread in threads:
        thread.start()
    try:
        return_code = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        for thread in threads:
            thread.join()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--circuit", required=True, type=directory_name)
    parser.add_argument("--raw-root", default=str(PROJECT_ROOT / "runs/raw/full_data"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "final_exam/configs/default_pin.json"))
    parser.add_argument("--run-id", type=directory_name)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-table-rows", default="none")
    parser.add_argument("--smoke", action="store_true",
                        help="Reduce training budgets; complete raw source is still imported")
    args = parser.parse_args(argv)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{args.circuit}_{stamp}_{uuid.uuid4().hex[:8]}_pin"
    if not run_id.endswith("_pin"):
        parser.error("Pin experiment run IDs must end in _pin")
    try:
        paths = new_run_paths(run_id)
        config_source = Path(args.config).expanduser().resolve()
        config = json.loads(config_source.read_text())
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
        config.setdefault("dataset", {}).update(train_circuits=[args.circuit], ood_circuits=[])
        from device_jepa.cli import _max_table_rows
        from scripts.run.run_raw_comparison import source_identity
        from final_exam.run_pin import validate_request_pin

        limit = _max_table_rows(args.max_table_rows)
        raw_root = Path(args.raw_root).expanduser().resolve()
        source = (raw_root / args.circuit).resolve()
        if not source.is_relative_to(raw_root):
            raise ValueError("Circuit directory escapes raw root")
        circuit, count = source_identity(source)
        if circuit != args.circuit:
            raise ValueError(f"SQLite circuit {circuit!r} differs from selected {args.circuit!r}")
        config, models = validate_request_pin(config, args.models, args.seeds, args.device, smoke=args.smoke)
    except (ValueError, OSError, TypeError, KeyError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))

    output = paths["results"]
    log_directory = paths["logs"]
    output.mkdir(parents=True, exist_ok=False)
    try:
        log_directory.mkdir(parents=True, exist_ok=False)
    except BaseException:
        output.rmdir()
        raise
    config_path = output / "config_pin.json"
    data_path = output / "data_pin.json"
    training_output = output / "training_pin"
    record_path = output / "run_pin.json"
    record = dict(
        run_id=run_id, task="single_circuit_pin", circuit=circuit, status="initializing",
        started_at=utc_now(), source_samples=count, raw_source=str(source),
        configuration_source=str(config_source), effective_config=str(config_path),
        models=models, seeds=args.seeds, device=args.device, smoke=args.smoke,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        slurm_node_list=os.environ.get("SLURM_JOB_NODELIST"),
        hpc_logs=str(paths["hpc_logs"]), logs=str(log_directory),
        launch_log=str(log_directory / "launch_pin.log"),
        experiment_log=str(log_directory / "experiment_pin.log"),
        results=str(output), training_output=str(training_output),
        summary_xlsx=str(training_output / "summary_pin.xlsx"),
    )
    try:
        save_json(config_path, config)
        save_json(record_path, record)
        with (log_directory / "launch_pin.log").open("x", buffering=1) as log:
            lock = threading.Lock()
            log.write(f"Run {run_id} | job {record['slurm_job_id']} | circuit {circuit}\n")
            print(f"Run {run_id}\nHPC logs: {paths['hpc_logs']}\n"
                  f"Experiment logs: {log_directory}\nResults: {output}", flush=True)
            record["status"] = "importing"
            save_json(record_path, record)
            run_logged([
                sys.executable, "-u", "-m", "device_jepa", "import_analog_agent",
                "--input", source, "--config", config_path, "--output", data_path,
                "--max-table-rows", "none" if limit is None else str(limit),
            ], log, lock)
            record["status"] = "training"
            save_json(record_path, record)
            command = [
                sys.executable, "-u", PROJECT_ROOT / "final_exam/run_pin.py",
                "--data", data_path, "--config", config_path, "--output", training_output,
                "--log-dir", log_directory, "--models", *models,
                "--seeds", *map(str, args.seeds), "--device", args.device,
            ]
            if args.smoke:
                command.append("--smoke")
            run_logged(command, log, lock)
            completion = json.loads((training_output / "run_pin.json").read_text())
            if completion.get("status") != "complete" or not (training_output / "summary_pin.xlsx").is_file():
                raise ValueError("Runner did not produce a complete experiment and summary_pin.xlsx")
            record.update(status="complete", finished_at=utc_now(), exit_code=0)
            save_json(record_path, record)
            print(f"Completed. Excel: {training_output / 'summary_pin.xlsx'}", flush=True)
    except BaseException as error:
        code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
        record.update(status="failed", finished_at=utc_now(), exit_code=code,
                      failure_type=type(error).__name__, failure_message=str(error))
        save_json(record_path, record)
        with (log_directory / "launch_pin.log").open("a") as log:
            traceback.print_exc(file=log)
        raise
    return output


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        traceback.print_exc()
        sys.exit(error.returncode if error.returncode > 0 else 128 - error.returncode)
