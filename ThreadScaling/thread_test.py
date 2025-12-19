#!/usr/bin/env python3
import os
import re
import sys
import time
import shutil
import threading
import subprocess
from statistics import mean

NUM_ENGINES = 384
ENGINE_PATH = "./monty"
MOVETIME_MS = 240_000  # 240 seconds
SETTLE_SECONDS = 20    # wait after setoption
CPU_IDS = list(range(NUM_ENGINES))

INFO_NPS_RE = re.compile(r'^info\b.*\bnps\s+(\d+)\b')

def ensure_environment():
    if not os.path.exists(ENGINE_PATH) or not os.access(ENGINE_PATH, os.X_OK):
        sys.stderr.write(f"Error: {ENGINE_PATH} not found or not executable.\n")
        sys.exit(1)

    try:
        cpu_count = os.cpu_count() or 0
    except Exception:
        cpu_count = 0
    if cpu_count < NUM_ENGINES:
        sys.stderr.write(f"Error: Need {NUM_ENGINES} logical CPUs but system reports {cpu_count}.\n")
        sys.exit(1)

def make_popen_bound(cpu_id: int):
    """Start a monty process bound to a specific logical CPU."""
    use_taskset = shutil.which("taskset") is not None

    if use_taskset:
        cmd = ["taskset", "-c", str(cpu_id), ENGINE_PATH]
        return subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1
        )
    else:
        # Fallback: set CPU affinity in the child with preexec_fn.
        def _preexec():
            try:
                os.sched_setaffinity(0, {cpu_id})
            except AttributeError:
                # Not available on this platform
                pass
        return subprocess.Popen(
            [ENGINE_PATH], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, preexec_fn=_preexec
        )

def send_line(p: subprocess.Popen, line: str):
    try:
        p.stdin.write(line + "\n")
        p.stdin.flush()
    except BrokenPipeError:
        pass

def reader_thread(proc_idx: int, p: subprocess.Popen, results: list, done_evt: threading.Event):
    """Read until bestmove; record the last seen NPS value."""
    last_nps = None
    try:
        for line in p.stdout:
            line = line.rstrip("\n")
            m = INFO_NPS_RE.match(line)
            if m:
                last_nps = int(m.group(1))
            if line.startswith("bestmove"):
                results[proc_idx] = last_nps
                break
    except Exception:
        # On any read error, leave results[proc_idx] as None
        pass
    finally:
        done_evt.set()

def main():
    ensure_environment()

    procs = []
    for i in CPU_IDS:
        p = make_popen_bound(i)
        procs.append(p)

    # Send required options to each engine
    for p in procs:
        send_line(p, "setoption name Threads value 1")
        send_line(p, "setoption name Hash value 1024")

    # Wait the requested 20 seconds
    time.sleep(SETTLE_SECONDS)

    # Prepare readers
    results = [None] * NUM_ENGINES
    done_events = [threading.Event() for _ in range(NUM_ENGINES)]
    readers = []
    for idx, p in enumerate(procs):
        t = threading.Thread(target=reader_thread, args=(idx, p, results, done_events[idx]), daemon=True)
        t.start()
        readers.append(t)

    # Fire 'go movetime 240000' to all engines "at the same time"
    # Use threads to minimize skew.
    barrier = threading.Barrier(NUM_ENGINES + 1)

    def _go_sender(p: subprocess.Popen):
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return
        send_line(p, f"go movetime {MOVETIME_MS}")

    senders = []
    for p in procs:
        t = threading.Thread(target=_go_sender, args=(p,), daemon=True)
        t.start()
        senders.append(t)

    # Release all senders simultaneously
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass

    # Wait for all engines to finish (expect ~MOVETIME_MS + buffer)
    timeout_sec = MOVETIME_MS / 1000 + 60  # 60s buffer
    deadline = time.time() + timeout_sec
    for evt in done_events:
        remaining = deadline - time.time()
        if remaining > 0:
            evt.wait(remaining)

    # Ask engines to quit (don't block if they already exited)
    for p in procs:
        send_line(p, "quit")

    # Ensure processes are reaped
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass

    # Validate and compute average
    missing = [i for i, v in enumerate(results) if v is None]
    if missing:
        sys.stderr.write(f"Error: Missing NPS from {len(missing)} engines: {missing[:10]}{'...' if len(missing)>10 else ''}\n")
        sys.exit(2)

    avg_nps = mean(results)
    print(f"{avg_nps:.2f}")

if __name__ == "__main__":
    main()
