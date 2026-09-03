"""Non-blocking background processing for durable-memory checkpoints."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from .engine import MemoryEngine

logger = logging.getLogger(__name__)


class MemoryWorker:
    """Process coalesced session checkpoints on one daemon thread.

    Checkpoints are durable before they are submitted here.  Therefore shutdown
    can stop accepting work without waiting for an in-flight network request;
    anything unfinished remains in ``.pending`` for a later run.
    """

    def __init__(
        self,
        engine: MemoryEngine,
        *,
        recovery_limit: int = 10,
        recovery_min_age: float = 300.0,
    ):
        self.engine = engine
        self.recovery_limit = recovery_limit
        self.recovery_min_age = recovery_min_age
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        self._thread: threading.Thread | None = None
        self._active = False
        self._closed = False
        self._recover_existing = False

    def start(self, *, recover_existing: bool = False) -> None:
        """Start immediately while doing all disk/network work in the thread."""
        with self._condition:
            if self._closed:
                return
            self._recover_existing = self._recover_existing or recover_existing
            if self._thread is not None:
                self._condition.notify_all()
                return
            self._thread = threading.Thread(
                target=self._run,
                name="corecoder-memory",
                daemon=True,
            )
            self._thread.start()

    def submit(self, source_session: str) -> bool:
        """Prioritize the latest durable checkpoint for ``source_session``."""
        if not source_session:
            return False
        self.start()
        with self._condition:
            if self._closed:
                return False
            if source_session in self._queued:
                self._queue.remove(source_session)
            else:
                self._queued.add(source_session)
            # The right side is high priority; startup recovery stays on the left.
            self._queue.append(source_session)
            self._condition.notify_all()
        return True

    def close(self, *, wait: bool = False, timeout: float | None = None) -> None:
        """Stop accepting work; by default never wait for model requests."""
        with self._condition:
            self._closed = True
            self._queue.clear()
            self._queued.clear()
            thread = self._thread
            self._condition.notify_all()
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait for tests or explicit administration, never for normal shutdown."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._active or self._queue or self._recover_existing:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
            return True

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._recover_existing and not self._closed:
                        self._condition.wait()
                    if self._closed:
                        return
                    if self._recover_existing:
                        self._recover_existing = False
                        self._active = True
                        source_session = None
                    else:
                        source_session = self._queue.pop()
                        self._queued.discard(source_session)
                        self._active = True
                if source_session is None:
                    try:
                        self._enqueue_existing()
                    finally:
                        with self._condition:
                            self._active = False
                            self._condition.notify_all()
                    continue
                try:
                    self.engine.recover_session(source_session, force=True)
                except Exception:
                    logger.warning(
                        "Background memory extraction failed for %s",
                        source_session,
                        exc_info=True,
                    )
                finally:
                    with self._condition:
                        self._active = False
                        self._condition.notify_all()
        finally:
            extractor = getattr(self.engine, "extractor", None)
            llm = getattr(extractor, "llm", None)
            close = getattr(llm, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    logger.debug("Could not close background memory client", exc_info=True)

    def _enqueue_existing(self) -> None:
        try:
            sessions = self.engine.pending_session_ids(
                limit=self.recovery_limit,
                min_age_seconds=self.recovery_min_age,
            )
        except Exception:
            logger.warning("Could not scan pending memory checkpoints", exc_info=True)
            return
        with self._condition:
            if self._closed:
                return
            for source_session in reversed(sessions):
                if source_session in self._queued:
                    continue
                self._queue.appendleft(source_session)
                self._queued.add(source_session)
            self._condition.notify_all()
