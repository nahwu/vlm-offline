import threading
from dataclasses import dataclass, field


@dataclass
class RuntimeMetrics:
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    last_gpu_allocated_mb: float = 0.0
    last_gpu_reserved_mb: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, latency_ms: float, status_code: int, gpu_allocated_mb: float, gpu_reserved_mb: float) -> None:
        with self._lock:
            self.total_requests += 1
            if status_code >= 400:
                self.total_errors += 1
            self.total_latency_ms += latency_ms
            self.last_latency_ms = latency_ms
            self.last_gpu_allocated_mb = gpu_allocated_mb
            self.last_gpu_reserved_mb = gpu_reserved_mb

    def snapshot(self) -> dict:
        with self._lock:
            avg = self.total_latency_ms / self.total_requests if self.total_requests else 0.0
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "average_latency_ms": round(avg, 2),
                "last_latency_ms": round(self.last_latency_ms, 2),
                "last_gpu_allocated_mb": round(self.last_gpu_allocated_mb, 2),
                "last_gpu_reserved_mb": round(self.last_gpu_reserved_mb, 2),
            }
