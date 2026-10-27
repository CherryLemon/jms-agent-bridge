"""Off-loop durable I/O; cancellation waits for the outstanding disk operation."""

import asyncio
import contextvars
import time

io_metrics = contextvars.ContextVar("jms_io_metrics", default=None)


async def offload(function, *args, on_cancel=None, **kwargs):
    started = time.monotonic()
    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    cancelled = None
    try:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancelled = exc
        value = task.result()
        if cancelled is not None:
            if on_cancel is not None:
                await offload(on_cancel, value)
            raise cancelled
        return value
    finally:
        metrics = io_metrics.get()
        if metrics is not None:
            metrics["storage_wait_sum_seconds"] = (
                metrics.get("storage_wait_sum_seconds", 0) + time.monotonic() - started
            )
            metrics["storage_operations"] = metrics.get("storage_operations", 0) + 1
