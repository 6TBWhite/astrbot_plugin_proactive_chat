from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from core.task_scheduler import SchedulerMixin
from core.thought import ThoughtMixin


class _FakeScheduler:
    def __init__(self) -> None:
        self.jobs: dict[str, SimpleNamespace] = {}

    def add_job(self, function, _trigger, **kwargs) -> None:
        self.jobs[kwargs["id"]] = SimpleNamespace(
            id=kwargs["id"], function=function, kwargs=kwargs
        )

    def get_job(self, job_id: str):
        return self.jobs.get(job_id)

    def get_jobs(self) -> list:
        return list(self.jobs.values())

    def remove_job(self, job_id: str) -> None:
        self.jobs.pop(str(job_id), None)


class _SchedulerHarness(SchedulerMixin, ThoughtMixin):
    def __init__(self) -> None:
        self.timezone = ZoneInfo("Asia/Shanghai")
        self.scheduler = _FakeScheduler()
        self.data_lock = asyncio.Lock()
        self.session_data = {}
        self.last_message_times = {}
        self.config = {}
        self.saved = 0
        self.session_config = {
            "enable": True,
            "schedule_settings": {
                "min_interval_minutes": 40,
                "max_interval_minutes": 200,
                "mean_position_ratio": 0.375,
                "weibull_shape": 1.7,
            },
        }

    @staticmethod
    def _parse_session_id(session_id: str):
        parts = str(session_id).split(":", 2)
        return tuple(parts) if len(parts) == 3 else None

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        return session_id

    def _get_session_config(self, _session_id: str):
        return self.session_config

    @staticmethod
    def _get_session_log_str(session_id: str, _config=None) -> str:
        return session_id

    @staticmethod
    def _cleanup_invalid_session_data() -> int:
        return 0

    async def _save_data_internal(self) -> None:
        self.saved += 1


class SchedulingTests(unittest.IsolatedAsyncioTestCase):
    def test_private_initial_sampling_uses_weibull_bounds(self) -> None:
        harness = _SchedulerHarness()
        timing = harness._build_schedule_timing(
            "default:FriendMessage:1",
            harness.session_config,
            scheduled_at=1_000,
        )
        self.assertTrue(timing["is_private"])
        self.assertEqual(timing["phase"], "initial")
        self.assertGreaterEqual(timing["sampled_seconds"], 40 * 60)
        self.assertLessEqual(timing["sampled_seconds"], 200 * 60)

    def test_group_sampling_remains_uniform(self) -> None:
        harness = _SchedulerHarness()
        config = {
            "schedule_settings": {
                "min_interval_minutes": 90,
                "max_interval_minutes": 360,
            }
        }
        with patch("core.task_scheduler.random.randint", return_value=123 * 60):
            timing = harness._build_schedule_timing(
                "default:GroupMessage:1", config, scheduled_at=1_000
            )
        self.assertFalse(timing["is_private"])
        self.assertEqual(timing["phase"], "uniform")
        self.assertEqual(timing["sampled_seconds"], 123 * 60)

    def test_after_silence_starts_fresh_cycle(self) -> None:
        harness = _SchedulerHarness()
        scheduled_at = 1_000 + 125 * 60
        timing = harness._build_schedule_timing(
            "default:FriendMessage:1",
            harness.session_config,
            scheduled_at=scheduled_at,
            cycle_started_at=1_000,
            after_silence=True,
            fresh_after_silence=True,
        )
        self.assertEqual(timing["cycle_started_at"], scheduled_at)
        self.assertEqual(timing["phase"], "after_silence")
        self.assertGreaterEqual(timing["sampled_seconds"], 40 * 60)
        self.assertLessEqual(timing["sampled_seconds"], 200 * 60)
        self.assertGreater(timing["next_trigger_time"], scheduled_at)
        self.assertLessEqual(timing["next_trigger_time"], scheduled_at + 200 * 60)

    async def test_silence_reschedule_increments_count_and_starts_fresh_cycle(self) -> None:
        harness = _SchedulerHarness()
        session_id = "default:FriendMessage:1"
        now = time.time()
        harness.session_data[session_id] = {
            "thought_cycle_started_at": now - 100 * 60,
            "last_scheduled_at": now - 100 * 60,
            "thought_silence_count": 0,
        }
        result = await harness._reschedule_after_thought_silence(
            session_id, expected_last_message_time=0
        )
        self.assertEqual(result, "scheduled")
        self.assertEqual(harness.session_data[session_id]["thought_silence_count"], 1)
        self.assertEqual(
            harness.session_data[session_id]["last_schedule_phase"],
            "after_silence",
        )
        self.assertIn(session_id, harness.scheduler.jobs)

    async def test_restart_restores_used_thought_silence_count(self) -> None:
        harness = _SchedulerHarness()
        session_id = "default:FriendMessage:1"
        now = time.time()
        harness.session_data[session_id] = {
            "next_trigger_time": now + 600,
            "last_scheduled_at": now - 6_000,
            "thought_cycle_started_at": now - 6_000,
            "thought_silence_count": 1,
            "last_schedule_phase": "after_silence",
        }
        await harness._init_jobs_from_data()
        job = harness.scheduler.get_job(session_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.function, harness.thoughtful_check_and_chat)
        self.assertEqual(harness.session_data[session_id]["thought_silence_count"], 1)


if __name__ == "__main__":
    unittest.main()
