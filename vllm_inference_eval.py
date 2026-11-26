#!/usr/bin/env python3
"""Run one continuous Qwen load on every GPU that stays idle for 30 minutes.

The normal launcher is a persistent local watchdog, so this script can run
from a shell, tmux, or another Python supervisor without requiring systemd.
The watchdog restarts its monitor worker if it exits.  The monitor thread polls
NVIDIA SMI while each inference process owns one CUDA device.
``--launch-mode systemd`` remains available for installations that still
prefer the template units.
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


DEFAULT_IDLE_THRESHOLD_SECONDS = 30 * 60
DEFAULT_SCAN_SECONDS = 30
DEFAULT_START_RETRY_SECONDS = 5 * 60
DEFAULT_MODEL_PATH = Path("/tmp/wyb/models/Qwen3-30B-A3B")
DEFAULT_INFERENCE_SCRIPT = Path(__file__).with_name("qwen_inference.py")
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_NEW_TOKENS = 64
DEFAULT_WORKLOAD_MODE = "prefill"
DEFAULT_INPUT_TOKENS = 1024
DEFAULT_SERVICE_TEMPLATE = "qwen-inference@{index}.service"
DEFAULT_SYSTEMCTL = "systemctl"
DEFAULT_LAUNCH_MODE = "process"
DEFAULT_LOCK_FILE = Path("/tmp/vllm_inference_eval.lock")
DEFAULT_WATCHDOG_RESTART_SECONDS = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gpu-idle-monitor")
stopping = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--idle-threshold",
        type=float,
        default=DEFAULT_IDLE_THRESHOLD_SECONDS,
        help=f"continuous idle duration before auto-start (default: {DEFAULT_IDLE_THRESHOLD_SECONDS}s)",
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=DEFAULT_SCAN_SECONDS,
        help=f"nvidia-smi polling interval (default: {DEFAULT_SCAN_SECONDS}s)",
    )
    parser.add_argument(
        "--start-retry-interval",
        type=float,
        default=DEFAULT_START_RETRY_SECONDS,
        help=(
            "delay before retrying a failed or exited inference process "
            f"(default: {DEFAULT_START_RETRY_SECONDS}s)"
        ),
    )
    parser.add_argument(
        "--launch-mode",
        choices=("process", "systemd"),
        default=DEFAULT_LAUNCH_MODE,
        help="launch inference directly or through systemd (default: process)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"local model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--inference-script",
        type=Path,
        default=DEFAULT_INFERENCE_SCRIPT,
        help=f"inference entry point (default: {DEFAULT_INFERENCE_SCRIPT})",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help=f"Python executable for child processes (default: {sys.executable})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"inference batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"tokens generated per prompt (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--workload-mode",
        choices=("prefill", "decode"),
        default=DEFAULT_WORKLOAD_MODE,
        help=f"inference mode for children (default: {DEFAULT_WORKLOAD_MODE})",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=DEFAULT_INPUT_TOKENS,
        help=f"fixed prefill length (default: {DEFAULT_INPUT_TOKENS})",
    )
    parser.add_argument(
        "--service-template",
        default=DEFAULT_SERVICE_TEMPLATE,
        help=(
            "systemd unit template used only with --launch-mode systemd; "
            f"use {{index}} for the GPU index (default: {DEFAULT_SERVICE_TEMPLATE})"
        ),
    )
    parser.add_argument(
        "--systemctl",
        default=DEFAULT_SYSTEMCTL,
        help=f"systemctl executable (default: {DEFAULT_SYSTEMCTL})",
    )
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=DEFAULT_LOCK_FILE,
        help=f"single-instance lock path (default: {DEFAULT_LOCK_FILE})",
    )
    parser.add_argument(
        "--watchdog-restart",
        type=float,
        default=DEFAULT_WATCHDOG_RESTART_SECONDS,
        help=(
            "seconds before restarting a monitor worker that exits "
            f"(default: {DEFAULT_WATCHDOG_RESTART_SECONDS}s)"
        ),
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.idle_threshold <= 0:
        parser.error("--idle-threshold must be greater than zero")
    if args.scan_interval <= 0:
        parser.error("--scan-interval must be greater than zero")
    if args.start_retry_interval <= 0:
        parser.error("--start-retry-interval must be greater than zero")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    if args.input_tokens < 1:
        parser.error("--input-tokens must be at least 1")
    if not args.service_template:
        parser.error("--service-template must not be empty")
    if args.watchdog_restart <= 0:
        parser.error("--watchdog-restart must be greater than zero")
    return args


def smi(args: list[str]) -> str:
    return subprocess.run(
        ["nvidia-smi", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout


def read_gpus() -> tuple[dict[str, tuple[int, int]], dict[str, set[int]]]:
    gpus: dict[str, tuple[int, int]] = {}
    rows = smi(
        [
            "--query-gpu=index,uuid,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    for row in rows.splitlines():
        parts = [part.strip() for part in row.split(",", 2)]
        if len(parts) != 3:
            continue
        index, uuid, util = parts
        gpus[uuid] = (int(index), int(util))

    pids: dict[str, set[int]] = {}
    apps = smi(
        ["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    )
    for row in apps.splitlines():
        parts = [part.strip() for part in row.split(",", 1)]
        if len(parts) == 2 and parts[1].isdigit():
            pids.setdefault(parts[0], set()).add(int(parts[1]))
    return gpus, pids


def unit_name(template: str, index: int) -> str:
    """Expand either the documented ``{index}`` or systemd-style ``%i``."""

    if "{index}" in template:
        return template.format(index=index)
    if "%i" in template:
        return template.replace("%i", str(index))
    return template


def start_inference(
    index: int,
    *,
    python_executable: str = sys.executable,
    inference_script: Path = DEFAULT_INFERENCE_SCRIPT,
    model_path: Path = DEFAULT_MODEL_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    workload_mode: str = DEFAULT_WORKLOAD_MODE,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
) -> subprocess.Popen[bytes] | None:
    """Start one continuous inference child bound to ``index``."""

    command = [
        python_executable,
        str(inference_script),
        "--model-path",
        str(model_path),
        "--device",
        str(index),
        "--duration",
        "0",
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
        "--workload-mode",
        workload_mode,
        "--input-tokens",
        str(input_tokens),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(inference_script.parent),
            env=environment,
            start_new_session=True,
        )
    except OSError as exc:
        log.error("Failed to start inference process on GPU %s: %s", index, exc)
        return None
    log.info("Started inference process on GPU %s (pid=%s)", index, process.pid)
    return process


def start_inference_service(
    index: int,
    *,
    service_template: str = DEFAULT_SERVICE_TEMPLATE,
    systemctl: str = DEFAULT_SYSTEMCTL,
) -> bool:
    """Ask systemd to start the per-GPU inference service."""

    unit = unit_name(service_template, index)
    try:
        subprocess.run(
            [systemctl, "start", "--no-block", unit],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        log.error("Failed to start inference service %s: %s", unit, exc)
        return False
    log.info("Started inference service %s for GPU %s", unit, index)
    return True


def stop_inference(process: subprocess.Popen[bytes], index: int) -> None:
    """Stop a child process and its process group during monitor shutdown."""

    if process.poll() is not None:
        return
    log.info("Stopping inference process for GPU %s (pid=%s)", index, process.pid)
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            process.kill()
        process.wait()


def monitor_loop(args: argparse.Namespace, stop_event: threading.Event) -> None:
    """Poll GPUs and launch/track one inference child per GPU."""

    idle_since: dict[str, float] = {}
    processes: dict[str, tuple[int, subprocess.Popen[bytes]]] = {}
    started_services: set[str] = set()
    last_start_attempt: dict[str, float] = {}
    log.info(
        "Started GPU idle monitor thread: threshold=%.0fs scan_interval=%.0fs launch_mode=%s",
        args.idle_threshold,
        args.scan_interval,
        args.launch_mode,
    )

    while not stop_event.is_set() and not stopping:
        try:
            gpus, pids = read_gpus()
            now = time.monotonic()
            known = set(gpus)

            for uuid, (index, process) in list(processes.items()):
                return_code = process.poll()
                if return_code is None:
                    continue
                log.warning(
                    "Inference process for GPU %s exited with status %s; it can be retried",
                    index,
                    return_code,
                )
                processes.pop(uuid, None)
                last_start_attempt[uuid] = now

            for uuid in set(processes) - known:
                index, process = processes.pop(uuid)
                stop_inference(process, index)
            for uuid in set(idle_since) - known:
                idle_since.pop(uuid, None)
                started_services.discard(uuid)
                last_start_attempt.pop(uuid, None)

            for uuid, (index, util) in gpus.items():
                process_running = uuid in processes
                service_running = uuid in started_services
                busy = util > 0 or bool(pids.get(uuid))
                if busy:
                    if uuid in idle_since:
                        log.info("GPU %s is no longer idle", index)
                    idle_since.pop(uuid, None)
                    continue

                start = idle_since.setdefault(uuid, now)
                idle_for = now - start
                if (
                    idle_for < args.idle_threshold
                    or process_running
                    or service_running
                ):
                    continue

                last_attempt = last_start_attempt.get(uuid)
                if (
                    last_attempt is not None
                    and now - last_attempt < args.start_retry_interval
                ):
                    continue
                last_start_attempt[uuid] = now

                if args.launch_mode == "systemd":
                    if start_inference_service(
                        index,
                        service_template=args.service_template,
                        systemctl=args.systemctl,
                    ):
                        started_services.add(uuid)
                    continue

                process = start_inference(
                    index,
                    python_executable=args.python_executable,
                    inference_script=args.inference_script,
                    model_path=args.model_path,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    workload_mode=args.workload_mode,
                    input_tokens=args.input_tokens,
                )
                if process is not None:
                    processes[uuid] = (index, process)
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            log.warning("GPU query failed; retrying: %s", exc)

        sleep_until = time.monotonic() + args.scan_interval
        while not stop_event.is_set() and not stopping:
            remaining = sleep_until - time.monotonic()
            if remaining <= 0:
                break
            stop_event.wait(timeout=min(1, remaining))

    for index, process in processes.values():
        stop_inference(process, index)
    log.info("Stopped GPU idle monitor thread")


def run_monitor(args: argparse.Namespace) -> int:
    """Run one monitor worker until it receives a termination signal."""

    global stopping
    stopping = False
    stop_event = threading.Event()

    def request_stop(_sig, _frame):
        global stopping
        stopping = True
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    monitor = threading.Thread(
        target=monitor_loop,
        args=(args, stop_event),
        name="gpu-idle-monitor",
    )
    monitor.start()
    try:
        while monitor.is_alive():
            monitor.join(timeout=1)
    except KeyboardInterrupt:
        request_stop(signal.SIGINT, None)
    finally:
        stop_event.set()
        monitor.join()
    return 0


def acquire_lock(path: Path):
    """Hold an advisory lock for the watchdog lifetime."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        log.error("Another vllm_inference_eval watchdog already owns %s", path)
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def worker_command() -> list[str]:
    """Recreate the current CLI for a worker child, excluding supervisor flags."""

    arguments = [arg for arg in sys.argv[1:] if arg != "--worker"]
    arguments.append("--worker")
    return [sys.executable, str(Path(__file__).resolve()), *arguments]


def run_watchdog(args: argparse.Namespace) -> int:
    """Keep a monitor worker alive and restart it after an unexpected exit."""

    lock_handle = acquire_lock(args.lock_file)
    if lock_handle is None:
        return 2

    stop_event = threading.Event()
    worker: subprocess.Popen[bytes] | None = None

    def request_stop(_sig, _frame):
        stop_event.set()
        if worker is not None and worker.poll() is None:
            worker.terminate()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log.info(
        "Started persistent watchdog (pid=%s); worker restart delay=%.0fs",
        os.getpid(),
        args.watchdog_restart,
    )
    try:
        while not stop_event.is_set():
            try:
                worker = subprocess.Popen(
                    worker_command(),
                    cwd=str(Path(__file__).resolve().parent),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
            except OSError as exc:
                log.error("Failed to start monitor worker: %s", exc)
                stop_event.wait(args.watchdog_restart)
                continue

            log.info("Started monitor worker (pid=%s)", worker.pid)
            while worker.poll() is None and not stop_event.wait(1):
                pass

            if stop_event.is_set():
                if worker.poll() is None:
                    worker.terminate()
                try:
                    worker.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except OSError:
                        worker.kill()
                    worker.wait()
                break

            return_code = worker.wait()
            log.error(
                "Monitor worker exited with status %s; restarting in %.0fs",
                return_code,
                args.watchdog_restart,
            )
            stop_event.wait(args.watchdog_restart)
    finally:
        lock_handle.close()
        log.info("Stopped persistent watchdog")
    return 0


def main() -> int:
    args = parse_args()
    if args.worker:
        return run_monitor(args)
    return run_watchdog(args)


if __name__ == "__main__":
    raise SystemExit(main())
