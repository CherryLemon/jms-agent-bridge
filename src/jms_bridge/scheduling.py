"""Bound active commands independently of authentication and idle connections."""

import asyncio
import contextlib
import time
from collections import deque


class QueueTimeout(Exception):
    def __init__(self, stage):
        self.stage = stage
        super().__init__(f"{stage} timed out; command was not sent")


class Timings:
    def __init__(self):
        self.values = dict(
            queue_seconds=0.0, connection_seconds=0.0, channel_seconds=0.0, execution_seconds=0.0
        )
        self.connection_details = {}

    @contextlib.contextmanager
    def connection_phase(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.connection_details[name] = (
                self.connection_details.get(name, 0) + time.monotonic() - started
            )

    def connection_snapshot(self):
        return {key: round(value, 6) for key, value in self.connection_details.items()}

    @contextlib.contextmanager
    def phase(self, name):
        started = time.monotonic()
        try:
            yield
        finally:
            self.values[name] += time.monotonic() - started

    def snapshot(self):
        return {k: round(v, 3) for k, v in self.values.items()}


class ExecutionSlots:
    """FIFO among eligible requests; inspection may use otherwise idle slots.

    General work cannot consume the inspection reserve. Inspection may use all
    slots when general work is absent. Authentication never holds a slot.
    """

    def __init__(self, limit, inspection_reserved=2):
        if type(limit) is not int or limit < 1:
            raise ValueError("execution limit must be a positive integer")
        if type(inspection_reserved) is not int or not 0 <= inspection_reserved < limit:
            raise ValueError("inspection_reserved must be in 0..limit-1")
        self.limit = limit
        self.inspection_reserved = inspection_reserved
        self.active = dict(general=0, inspection=0)
        self.waiters = deque()
        self.peak_active = 0

    def _eligible(self, lane):
        return sum(self.active.values()) < self.limit and (
            lane == "inspection" or self.active["general"] < self.limit - self.inspection_reserved
        )

    def _wake(self):
        for waiter in list(self.waiters):
            if waiter["future"].cancelled():
                self.waiters.remove(waiter)
            elif self._eligible(waiter["lane"]):
                self.waiters.remove(waiter)
                self.active[waiter["lane"]] += 1
                waiter["granted"] = True
                self.peak_active = max(self.peak_active, sum(self.active.values()))
                waiter["future"].set_result(None)

    def _release(self, lane):
        self.active[lane] -= 1
        self._wake()

    @contextlib.asynccontextmanager
    async def slot(self, lane, timeout, timings):
        if lane not in self.active:
            raise ValueError("unknown execution lane")
        if timeout <= 0:
            raise ValueError("queue timeout must be positive")
        waiter = dict(lane=lane, future=asyncio.get_running_loop().create_future(), granted=False)
        self.waiters.append(waiter)
        self._wake()
        try:
            with timings.phase("queue_seconds"):
                try:
                    await asyncio.wait_for(asyncio.shield(waiter["future"]), timeout)
                except asyncio.TimeoutError:
                    raise QueueTimeout("execution_queue") from None
            yield
        finally:
            if waiter["granted"]:
                self._release(lane)
            else:
                waiter["future"].cancel()
                self.waiters.remove(waiter)
                self._wake()

    def status(self):
        return {
            "limit": self.limit,
            "inspection_reserved": self.inspection_reserved,
            "general_limit": self.limit - self.inspection_reserved,
            "active": dict(self.active),
            "queued": {lane: sum(w["lane"] == lane for w in self.waiters) for lane in self.active},
            "peak_active": self.peak_active,
        }
