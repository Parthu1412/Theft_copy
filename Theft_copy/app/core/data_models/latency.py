from dataclasses import dataclass


@dataclass
class LatencyStats:
    total: int = 0
    inference_count: int = 0
    sum_latency: float = 0.0
    min_latency: float = float("inf")
    max_latency: float = 0.0

    def add_latency(self, latency: float):
        self.inference_count += 1
        self.sum_latency += latency

        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)

    def average_latency(self) -> float:
        if self.inference_count == 0:
            return 0.0
        return self.sum_latency / self.inference_count

    def reset_latency_stats(self):
        self.total = 0
        self.inference_count = 0
        self.sum_latency = 0.0
        self.min_latency = float("inf")
        self.max_latency = 0.0

    def __str__(self) -> str:
        return (
            f"Inferenced: {self.inference_count} of Total: {self.total}, "
            f"Avg Latency: {self.average_latency():.4f}s, "
            f"Min Latency: {self.min_latency:.4f}s, Max Latency: {self.max_latency:.4f}s"
        )
