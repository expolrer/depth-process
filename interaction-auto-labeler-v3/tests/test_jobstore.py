from interaction_auto_labeler_v3.jobstore import Job, JobStore


def test_jobstore_is_idempotent_and_leases_are_exclusive(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    jobs = [
        Job.from_dict(
            {
                "dataset_id": "d",
                "dataset_version": "v1",
                "episode_id": str(index),
                "stage": "label",
                "payload": {"index": index},
            }
        )
        for index in range(3)
    ]
    assert store.enqueue(jobs) == {"inserted": 3, "existing": 0}
    assert store.enqueue(jobs) == {"inserted": 0, "existing": 3}
    first = store.lease("worker-a", 2)
    second = store.lease("worker-b", 2)
    assert len(first) == 2 and len(second) == 1
    assert {row["job_id"] for row in first}.isdisjoint({row["job_id"] for row in second})
    assert store.complete(first[0]["job_id"], "worker-a", {"ok": True})


def test_jobstore_retries_then_moves_to_dead_letter(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = Job.from_dict(
        {
            "dataset_id": "d",
            "dataset_version": "v1",
            "episode_id": "1",
            "stage": "label",
            "max_attempts": 2,
        }
    )
    store.enqueue([job])
    leased = store.lease("w")[0]
    assert store.fail(leased["job_id"], "w", "first", retry_base_seconds=0) == "pending"
    leased = store.lease("w")[0]
    assert store.fail(leased["job_id"], "w", "second", retry_base_seconds=0) == "dead"
    assert store.summary()["by_status"] == {"dead": 1}
