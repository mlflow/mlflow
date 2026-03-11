"""
Resource monitor for the gateway benchmark.

Polls CPU%, RSS, VMS, and thread count for a process and its children every 1 second.
Writes results to a CSV file.

Usage:
    python collect_resources.py <pid> <output_csv>
"""

import csv
import sys
import time

import psutil


def collect(pid, output_path, interval=1.0):
    fieldnames = ["timestamp", "cpu_percent", "rss_mb", "vms_mb", "threads"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            print(f"Process {pid} not found", file=sys.stderr)
            return

        # Prime CPU measurement
        proc.cpu_percent()
        for child in proc.children(recursive=True):
            try:
                child.cpu_percent()
            except psutil.NoSuchProcess:
                pass

        while True:
            try:
                if not proc.is_running():
                    break

                children = proc.children(recursive=True)
                all_procs = [proc] + children

                total_cpu = 0.0
                total_rss = 0
                total_vms = 0
                total_threads = 0

                for p in all_procs:
                    try:
                        total_cpu += p.cpu_percent()
                        mem = p.memory_info()
                        total_rss += mem.rss
                        total_vms += mem.vms
                        total_threads += p.num_threads()
                    except psutil.NoSuchProcess:
                        continue

                writer.writerow(
                    {
                        "timestamp": time.time(),
                        "cpu_percent": round(total_cpu, 1),
                        "rss_mb": round(total_rss / (1024 * 1024), 1),
                        "vms_mb": round(total_vms / (1024 * 1024), 1),
                        "threads": total_threads,
                    }
                )
                f.flush()
                time.sleep(interval)

            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pid> <output_csv>", file=sys.stderr)
        sys.exit(1)

    collect(int(sys.argv[1]), sys.argv[2])
