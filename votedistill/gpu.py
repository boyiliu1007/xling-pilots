"""nvidia-smi helpers: wait for a shared GPU to be free, and track peak memory."""
import logging
import os
import subprocess
import threading
import time

log = logging.getLogger(__name__)
GPU_LIMIT_MIB = 4096   # other processes using more than this count as "busy"
POLL_S = 300


def _smi(query):
    out = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=30).stdout.strip()
    return [l for l in out.splitlines() if l.strip()]


def gpu_used_mib():
    """memory.used of the first visible GPU in MiB, or -1 if nvidia-smi fails."""
    try:
        return int(_smi("--query-gpu=memory.used")[0])
    except Exception:
        return -1


def other_gpu_mib():
    me = os.getpid()
    procs = [(int(a.split(",")[0]), int(a.split(",")[1])) for a in _smi("--query-compute-apps=pid,used_memory")]
    used = int(_smi("--query-gpu=memory.used")[0])
    return sum(m for p, m in procs if p != me), used, procs


def wait_for_gpu():
    """Block until no other process holds more than GPU_LIMIT_MIB on the GPU."""
    while True:
        others, used, procs = other_gpu_mib()
        busy = others > GPU_LIMIT_MIB or (not procs and used > GPU_LIMIT_MIB)
        log.info(f"GPU check: other processes {others} MiB, total used {used} MiB, procs {procs} -> "
                 f"{'BUSY, waiting ' + str(POLL_S) + 's' if busy else 'free'}")
        if not busy:
            return
        time.sleep(POLL_S)


class GpuPeak(threading.Thread):
    """Background poll of peak memory.used (MiB); includes vLLM's KV-cache reservation."""

    def __init__(self, period=2.0):
        super().__init__(daemon=True)
        self.peak, self.period, self.stop_ev = 0, period, threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            self.peak = max(self.peak, gpu_used_mib())
            self.stop_ev.wait(self.period)
