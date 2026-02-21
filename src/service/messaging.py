from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Any, Awaitable, Callable

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
except ModuleNotFoundError:  # pragma: no cover
    AIOKafkaConsumer = None
    AIOKafkaProducer = None


APPRAISAL_REQUESTS_TOPIC = "appraisal_requests"
APPRAISAL_RESULTS_TOPIC = "appraisal_results"
RETRAINING_TRIGGERS_TOPIC = "retraining_triggers"
MANUAL_REVIEW_TOPIC = "manual_review_queue"


class KafkaBus:
    def __init__(self, bootstrap_servers: str, client_id: str) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self._producer: AIOKafkaProducer | None = None
        self._queues: dict[str, asyncio.Queue[dict[str, Any]]] = defaultdict(asyncio.Queue)

    async def connect(self) -> None:
        if AIOKafkaProducer is None:
            return
        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            client_id=self.client_id,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        try:
            await asyncio.wait_for(producer.start(), timeout=1.0)
            self._producer = producer
        except Exception:
            self._producer = None

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()

    async def ping(self) -> bool:
        if self._producer is None:
            return False
        try:
            partitions = await self._producer.partitions_for(APPRAISAL_REQUESTS_TOPIC)
            return partitions is not None
        except Exception:
            return False

    async def publish(self, topic: str, value: dict[str, Any], key: str | None = None) -> None:
        if self._producer is not None:
            try:
                encoded_key = None if key is None else key.encode("utf-8")
                await self._producer.send_and_wait(topic, value=value, key=encoded_key)
                return
            except Exception:
                pass
        await self._queues[topic].put(value)

    async def consume_retraining_forever(
        self,
        handler: Callable[[dict[str, Any]], Awaitable[None]],
        stop_event: asyncio.Event,
    ) -> None:
        if AIOKafkaConsumer is not None:
            consumer = AIOKafkaConsumer(
                RETRAINING_TRIGGERS_TOPIC,
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"{self.client_id}-retrainer",
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            )
            try:
                await asyncio.wait_for(consumer.start(), timeout=1.0)
                while not stop_event.is_set():
                    msg = await consumer.getone()
                    await handler(msg.value)
            except Exception:
                pass
            finally:
                try:
                    await consumer.stop()
                except Exception:
                    pass

        queue = self._queues[RETRAINING_TRIGGERS_TOPIC]
        while not stop_event.is_set():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            await handler(event)
