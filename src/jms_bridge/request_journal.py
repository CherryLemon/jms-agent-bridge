"""Durable, caller-addressable receipts; a claimed request is never replayed.

This journal deliberately does not resume work after a service restart. A receipt
without a live owner is unknown and must be reconciled using its job/result IDs.
Only one resident service should own a journal directory.
"""

import asyncio
import contextvars
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from .async_io import offload
from .results import atomic_json, locked


class ReceiptList(list):
    def __init__(self, observer):
        super().__init__()
        self.observer = observer

    def append(self, receipt):
        self.observer(receipt)
        super().append(receipt)


class RequestJournal:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        self.tasks = {}
        self._lock = threading.RLock()
        self._claim_lock = asyncio.Lock()
        self.current = contextvars.ContextVar("request_receipt", default=None)

    def directory(self, request_id):
        if not isinstance(request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{1,128}", request_id
        ):
            raise ValueError(
                "request_id must contain 1..128 letters, digits, dots, colons, "
                "underscores, or hyphens"
            )
        return self.root / hashlib.sha256(request_id.encode()).hexdigest()

    @staticmethod
    def sync_directory(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @locked
    def save(self, receipt):
        directory = self.directory(receipt["request_id"])
        atomic_json(directory / "state.json", receipt)
        self.sync_directory(directory)

    @locked
    def load(self, request_id):
        directory = self.directory(request_id)
        try:
            return json.loads((directory / "state.json").read_text())
        except FileNotFoundError:
            if directory.exists():
                # A crash between the atomic directory claim and first write must
                # never make a second invocation eligible to execute.
                return dict(
                    request_id=request_id,
                    request_state="interrupted",
                    completion_unknown=True,
                    error="receipt initialization interrupted",
                )
            raise ValueError("unknown request_id") from None

    @locked
    def status(self, request_id, *, duplicate=False):
        receipt = self.load(request_id)
        if receipt["request_state"] == "running" and request_id not in self.tasks:
            receipt.update(
                request_state="interrupted",
                completion_unknown=True,
                error="no live request owner; reconcile saved job/result IDs before new work",
            )
            self.save(receipt)
        response = dict(receipt.get("response") or {})
        response.update({k: v for k, v in receipt.items() if k not in ("fingerprint", "response")})
        response.setdefault("ok", receipt["request_state"] == "running")
        response["duplicate"] = duplicate
        response["execution_replayed"] = False
        return response

    @locked
    def progress(self, field, value):
        receipt = self.current.get()
        if receipt is None:
            return
        rows = receipt.setdefault(field, [])
        identity = "result_id" if field == "execution_results" else "job_id"
        for i, row in enumerate(rows):
            if row.get(identity) == value.get(identity) and row.get("host") == value.get("host"):
                rows[i] = dict(value)
                break
        else:
            rows.append(dict(value))
        self.save(receipt)

    def receipts(self, field):
        return ReceiptList(lambda value: self.progress(field, value))

    async def astatus(self, request_id, *, duplicate=False):
        async with self._claim_lock:
            return await offload(self.status, request_id, duplicate=duplicate)

    async def aprogress(self, field, value):
        await offload(self.progress, field, value)

    @locked
    def claim(self, request_id, operation, arguments):
        directory = self.directory(request_id)
        fingerprint = hashlib.sha256(
            json.dumps(
                [operation, arguments],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            receipt = self.load(request_id)
            if receipt.get("fingerprint") != fingerprint:
                raise ValueError(
                    "request_id already claimed; use request_status to reconcile it "
                    "(arguments cannot change)"
                ) from None
            return False, self.status(request_id, duplicate=True)
        self.sync_directory(self.root)
        receipt = dict(
            request_id=request_id,
            operation=operation,
            fingerprint=fingerprint,
            request_state="running",
            accepted_at=time.time(),
            completion_unknown=False,
        )
        self.save(receipt)
        return True, receipt

    async def execute(self, request_id, operation, arguments, perform):
        async def work(receipt):
            token = self.current.set(receipt)
            try:
                receipt["response"] = await perform()
                response = receipt["response"]
                unknown = response.get("completion_unknown", False) or any(
                    row.get("completion_unknown", False)
                    for field in ("results", "submission_receipts")
                    for row in response.get(field, receipt.get(field, []))
                )
                receipt.update(
                    request_state="complete",
                    completion_unknown=bool(unknown),
                    finished_at=time.time(),
                )
            except asyncio.CancelledError:
                receipt.update(
                    request_state="interrupted",
                    completion_unknown=True,
                    error="request worker cancelled; reconcile before new work",
                    finished_at=time.time(),
                )
                raise
            except Exception as exc:
                # A raised exception alone cannot prove that no side effect occurred.
                receipt.update(
                    request_state="failed",
                    completion_unknown=True,
                    error=f"request failed ({type(exc).__name__}); inspect saved receipts",
                    finished_at=time.time(),
                )
            finally:
                self.current.reset(token)
                try:
                    await offload(self.save, receipt)
                finally:
                    self.tasks.pop(request_id, None)

        # Serialize claim + task registration so a concurrent lookup cannot label
        # an in-progress durable claim as an orphaned request.
        async with self._claim_lock:
            created, receipt = await offload(self.claim, request_id, operation, arguments)
            if not created:
                return receipt
            task = asyncio.create_task(
                work(receipt), name="jms-request-" + self.directory(request_id).name
            )
            self.tasks[request_id] = task
        # Keep worker ownership when an MCP response/client disappears. Its final
        # result can be fetched with request_status using the caller's original ID.
        await asyncio.shield(task)
        return await self.astatus(request_id)

    async def drain(self):
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks.values()), return_exceptions=True)
