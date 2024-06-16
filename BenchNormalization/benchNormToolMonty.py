import subprocess
import multiprocessing
import os
import sys
import time

IS_WINDOWS = os.name == 'nt'

def run_single_bench_monty(engine, queue):
    bench_sig = None
    bench_nps = None

    p = subprocess.Popen(
        [engine, "bench"],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
        close_fds=not IS_WINDOWS,
    )

    for line in iter(p.stdout.readline, ""):
        if "Bench: " in line:
            spl = line.split(' ')
            bench_sig = int(spl[1].strip())
            bench_nps = float(spl[3].strip())

    queue.put((bench_sig, bench_nps))

def verify_signature(engine, active_cores):
    queue = multiprocessing.Queue()

    processes = [
        multiprocessing.Process(
            target=run_single_bench_monty,
            args=(engine, queue),
        ) for _ in range(active_cores)
    ]

    for p in processes:
        p.start()

    results = [queue.get() for _ in range(active_cores)]
    bench_nps = 0.0

    for sig, nps in results:
        bench_nps += nps

    bench_nps /= active_cores

    return bench_nps

def main():
    print("Running benchmark. This will take a minute")
    if len(sys.argv) != 3:
        print("Usage: python script.py <engine_path> <active_cores>")
        sys.exit(1)

    engine_path = sys.argv[1]
    active_cores = int(sys.argv[2])

    start_time = time.time()
    total_nps = 0.0
    count = 0

    while time.time() - start_time < 60:  # 5 minutes = 300 seconds
        bench_nps = verify_signature(engine_path, active_cores)
        total_nps += bench_nps
        count += 1

    average_nps = total_nps / count
    print(f"Final Average Benchmark NPS over a minute: {average_nps}")

if __name__ == "__main__":
    main()
