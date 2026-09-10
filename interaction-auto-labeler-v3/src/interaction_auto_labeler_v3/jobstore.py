from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Job:
    job_id: str
    dataset_id: str
    dataset_version: str
    episode_id: str
    stage: str
    payload: dict[str, Any]
    priority: int = 0
    max_attempts: int = 3

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Job:
        required = ("dataset_id", "dataset_version", "episode_id", "stage")
        missing = [name for name in required if not str(row.get(name, "")).strip()]
        if missing:
            raise ValueError(f"job is missing required fields: {missing}")
        identity = ":".join(str(row[name]) for name in required)
        job_id = str(row.get("job_id") or hashlib.sha256(identity.encode()).hexdigest())
        return cls(
            job_id=job_id,
            dataset_id=str(row["dataset_id"]),
            dataset_version=str(row["dataset_version"]),
            episode_id=str(row["episode_id"]),
            stage=str(row["stage"]),
            payload=dict(row.get("payload", {})),
            priority=int(row.get("priority", 0)),
            max_attempts=int(row.get("max_attempts", 3)),
        )


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    available_at REAL NOT NULL,
                    lease_owner TEXT,
                    lease_until REAL,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_ready
                    ON jobs(status, available_at, priority DESC, created_at);
                CREATE INDEX IF NOT EXISTS jobs_episode
                    ON jobs(dataset_id, dataset_version, episode_id, stage);
                """
            )

    def enqueue(self, jobs: Iterable[Job]) -> dict[str, int]:
        inserted = existing = 0
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for job in jobs:
                    if job.max_attempts < 1:
                        raise ValueError("max_attempts must be positive")
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO jobs (
                            job_id, dataset_id, dataset_version, episode_id, stage,
                            payload_json, priority, max_attempts, available_at,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job.job_id,
                            job.dataset_id,
                            job.dataset_version,
                            job.episode_id,
                            job.stage,
                            json.dumps(job.payload, ensure_ascii=False, separators=(",", ":")),
                            job.priority,
                            job.max_attempts,
                            now,
                            now,
                            now,
                        ),
                    )
                    if cursor.rowcount:
                        inserted += 1
                    else:
                        existing += 1
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return {"inserted": inserted, "existing": existing}

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        output = dict(row)
        output["payload"] = json.loads(output.pop("payload_json"))
        if output.get("result_json"):
            output["result"] = json.loads(output.pop("result_json"))
        else:
            output.pop("result_json", None)
        return output

    def lease(
        self, worker: str, count: int = 1, lease_seconds: float = 900
    ) -> list[dict[str, Any]]:
        if not worker or count < 1 or lease_seconds <= 0:
            raise ValueError("worker, count, and positive lease_seconds are required")
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE jobs SET status='pending', lease_owner=NULL, lease_until=NULL,
                        available_at=?, updated_at=?
                    WHERE status='leased' AND lease_until < ? AND attempt < max_attempts
                    """,
                    (now, now, now),
                )
                connection.execute(
                    """
                    UPDATE jobs SET status='dead', lease_owner=NULL, lease_until=NULL,
                        error=COALESCE(error, 'lease expired after final attempt'), updated_at=?
                    WHERE status='leased' AND lease_until < ? AND attempt >= max_attempts
                    """,
                    (now, now),
                )
                ids = [
                    row["job_id"]
                    for row in connection.execute(
                        """
                        SELECT job_id FROM jobs
                        WHERE status='pending' AND available_at <= ?
                        ORDER BY priority DESC, created_at, job_id LIMIT ?
                        """,
                        (now, count),
                    )
                ]
                for job_id in ids:
                    connection.execute(
                        """
                        UPDATE jobs SET status='leased', lease_owner=?, lease_until=?,
                            attempt=attempt+1, updated_at=?
                        WHERE job_id=? AND status='pending'
                        """,
                        (worker, now + lease_seconds, now, job_id),
                    )
                rows = []
                if ids:
                    placeholders = ",".join("?" for _ in ids)
                    rows = list(
                        connection.execute(
                            f"SELECT * FROM jobs WHERE job_id IN ({placeholders})", ids
                        )
                    )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return [self._row(row) for row in rows]

    def heartbeat(self, job_id: str, worker: str, lease_seconds: float = 900) -> bool:
        now = time.time()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET lease_until=?, updated_at=?
                WHERE job_id=? AND status='leased' AND lease_owner=?
                """,
                (now + lease_seconds, now, job_id, worker),
            )
        return bool(cursor.rowcount)

    def complete(self, job_id: str, worker: str, result: dict[str, Any]) -> bool:
        now = time.time()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs SET status='succeeded', result_json=?, error=NULL,
                    lease_owner=NULL, lease_until=NULL, updated_at=?
                WHERE job_id=? AND status='leased' AND lease_owner=?
                """,
                (json.dumps(result, ensure_ascii=False), now, job_id, worker),
            )
        return bool(cursor.rowcount)

    def fail(
        self,
        job_id: str,
        worker: str,
        error: str,
        retry_base_seconds: float = 30,
    ) -> str:
        now = time.time()
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT attempt, max_attempts FROM jobs WHERE job_id=? AND status='leased' AND lease_owner=?",
                (job_id, worker),
            ).fetchone()
            if row is None:
                connection.rollback()
                return "not_owned"
            dead = int(row["attempt"]) >= int(row["max_attempts"])
            status = "dead" if dead else "pending"
            available = now if dead else now + retry_base_seconds * (2 ** (int(row["attempt"]) - 1))
            connection.execute(
                """
                UPDATE jobs SET status=?, available_at=?, error=?, lease_owner=NULL,
                    lease_until=NULL, updated_at=? WHERE job_id=?
                """,
                (status, available, error[-8000:], now, job_id),
            )
            connection.commit()
        return status

    def summary(self) -> dict[str, Any]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM jobs GROUP BY status"
            ).fetchall()
            stages = connection.execute(
                "SELECT stage, status, COUNT(*) AS count FROM jobs GROUP BY stage, status"
            ).fetchall()
        return {
            "schema": "embodied_job_queue_summary_v1",
            "database": str(self.path),
            "by_status": {row["status"]: int(row["count"]) for row in rows},
            "by_stage": [dict(row) for row in stages],
        }
