from __future__ import annotations

from dataclasses import dataclass
from queue import Queue


@dataclass
class ManualReviewRouter:
    confidence_threshold: float = 0.95

    def __post_init__(self) -> None:
        self.queue: Queue[dict] = Queue()

    def route(self, appraisal_payload: dict, confidence: float) -> bool:
        reject = confidence < self.confidence_threshold
        if reject:
            self.queue.put(appraisal_payload)
        return reject

