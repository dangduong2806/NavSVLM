"""Optional synchronized stage diagnostics; disabled in normal inference.

Use a separate diagnostic pass: synchronizing at every stage changes execution
timing, so these measurements must not replace uninstrumented end-to-end timing.
"""
from contextlib import contextmanager, nullcontext
import time

import torch


def stage_scope(recorder, name):
    return nullcontext() if recorder is None else recorder.measure(name)


class StageTimings:
    def __init__(self, device):
        self.device = torch.device(device)
        self.events = []

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.events.append({"stage": name, "seconds": time.perf_counter() - start})
