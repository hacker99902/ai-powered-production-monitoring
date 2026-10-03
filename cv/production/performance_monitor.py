
import time
import subprocess
from collections import deque

import psutil
import torch


class PerformanceMonitor:
    def __init__(self, camera_count=7, window_seconds=5):
        self.camera_count = camera_count
        self.window_seconds = window_seconds
        self.frame_times = [deque() for _ in range(camera_count)]
        self.total_processed = [0] * camera_count
        self.start_time = time.monotonic()
        self.last_report = 0

        psutil.cpu_percent(interval=None)

    def record_processed(self, camera_id):
        now = time.monotonic()
        self.frame_times[camera_id].append(now)
        self.total_processed[camera_id] += 1

    def get_gpu_stats(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=True
            )

            values = result.stdout.strip().split(",")
            return (
                int(values[0].strip()),
                int(values[1].strip()),
                int(values[2].strip())
            )
        except (FileNotFoundError, subprocess.SubprocessError,
                ValueError, IndexError):
            return None, None, None

    def report(self, queues, read_counts):
        now = time.monotonic()

        if now - self.last_report < 1:
            return

        self.last_report = now

        print("\n" + "=" * 65)
        print("       7-CAMERA SMART FACTORY PERFORMANCE")
        print("=" * 65)

        total_fps = 0

        for i in range(self.camera_count):
            times = self.frame_times[i]

            while times and now - times[0] > self.window_seconds:
                times.popleft()

            fps = len(times) / self.window_seconds
            total_fps += fps

            print(
                f"Camera {i + 1}: "
                f"FPS={fps:.2f} | "
                f"Queue={queues[i].qsize()} | "
                f"Read={read_counts[i]} | "
                f"Processed={self.total_processed[i]}"
            )

        print("-" * 65)
        print(f"Total processing FPS: {total_fps:.2f}")
        print(f"Total processed: {sum(self.total_processed)}")
        print(f"CPU usage: {psutil.cpu_percent():.1f}%")

        ram = psutil.virtual_memory()
        print(
            f"RAM: {ram.used / (1024**3):.2f} / "
            f"{ram.total / (1024**3):.2f} GB "
            f"({ram.percent:.1f}%)"
        )

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(
                f"PyTorch GPU memory: "
                f"{allocated:.2f} GB allocated | "
                f"{reserved:.2f} GB reserved"
            )

        # gpu_util, vram_used, vram_total = self.get_gpu_stats()

        # if gpu_util is not None:
        #     print(f"GPU utilization: {gpu_util}%")
        #     print(f"GPU VRAM: {vram_used} / {vram_total} MB")

        print(f"Elapsed: {now - self.start_time:.1f} seconds")
        print("=" * 65)