import os
import time
import threading
import psutil


class ResourceMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.proc = psutil.Process(os.getpid())
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            self.samples.append(
                {
                    "ts": time.time(),
                    "cpu_pct": self.proc.cpu_percent(interval=None),
                    "rss_mb": self.proc.memory_info().rss / (1024 * 1024),
                }
            )
            time.sleep(self.interval)

    def start(self):
        self.proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {
                "cpu_pct_avg": None,
                "cpu_pct_max": None,
                "rss_mb_avg": None,
                "rss_mb_max": None,
            }
        cpu_vals = [x["cpu_pct"] for x in self.samples]
        mem_vals = [x["rss_mb"] for x in self.samples]
        return {
            "cpu_pct_avg": round(sum(cpu_vals) / len(cpu_vals), 4),
            "cpu_pct_max": round(max(cpu_vals), 4),
            "rss_mb_avg": round(sum(mem_vals) / len(mem_vals), 4),
            "rss_mb_max": round(max(mem_vals), 4),
        }
