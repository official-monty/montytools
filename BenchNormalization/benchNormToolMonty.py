import multiprocessing
import os
import sys
import time
import subprocess

IS_WINDOWS = os.name == "nt"

def worker(engine, queue):
    """
    Start Monty once, then repeatedly send 'bench' and read its result.
    Each time a 'Bench:' line is seen, push (sig, nps) into queue.
    """
    p = subprocess.Popen(
        [engine],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        close_fds=not IS_WINDOWS,
    )

    assert p.stdin and p.stdout

    try:
        while True:
            # send a bench command
            p.stdin.write("bench\n")
            p.stdin.flush()

            sig, nps = None, None
            for line in iter(p.stdout.readline, ""):
                if "Bench:" in line and "nps" in line.lower():
                    parts = line.strip().split()
                    # e.g. Bench: 1119942 nodes 584628 nps
                    if len(parts) >= 5:
                        sig = int(parts[1])
                        nps = float(parts[3])
                    break  # stop after one bench result

            if sig is not None and nps is not None:
                queue.put((sig, nps))
            else:
                # engine misbehaved? bail
                queue.put((None, None))
                break
    finally:
        with subprocess.Popen:
            try:
                p.terminate()
            except Exception:
                pass


def verify_signature(engine, active_cores, duration=60):
    queue = multiprocessing.Queue()

    workers = [
        multiprocessing.Process(target=worker, args=(engine, queue))
        for _ in range(active_cores)
    ]

    for w in workers:
        w.start()

    start = time.time()
    results = []

    while time.time() - start < duration:
        try:
            sig, nps = queue.get(timeout=5)
            if nps is not None:
                results.append(nps)
        except Exception:
            pass

    # cleanup
    for w in workers:
        if w.is_alive():
            w.terminate()
        w.join()

    if not results:
        raise RuntimeError("No benchmark results parsed from engine output.")

    return sum(results) / len(results)


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <engine_path> <active_cores>")
        sys.exit(1)

    engine_path = sys.argv[1]
    active_cores = int(sys.argv[2])

    print(f"Running benchmark for 60s with {active_cores} Monty workers...")

    avg_nps = verify_signature(engine_path, active_cores, duration=60)
    print(f"Final Average Benchmark NPS: {avg_nps:.2f}")


if __name__ == "__main__":
    main()
