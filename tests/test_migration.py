from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.data_storage import StorageMixin
from core.session_config import ConfigMixin
from core.session_override_manager import SessionOverrideManager


class _SavingConfig(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.saved = 0

    def save_config(self) -> None:
        self.saved += 1


class _ConfigHarness(ConfigMixin):
    def __init__(self, config: _SavingConfig) -> None:
        self.config = config


class _StorageHarness(StorageMixin):
    def __init__(self, session_data: dict) -> None:
        self.session_data = session_data


class MigrationTests(unittest.IsolatedAsyncioTestCase):
    def test_global_gate_config_is_removed_once(self) -> None:
        config = _SavingConfig(
            {
                "friend_settings": {
                    "gate_settings": {
                        "enable_gate": True,
                        "gate_model": "cheap-provider",
                    },
                    "schedule_settings": {"mean_position_ratio": 0.375},
                }
            }
        )
        harness = _ConfigHarness(config)
        self.assertTrue(harness._migrate_legacy_gate_config())
        self.assertNotIn("gate_settings", config["friend_settings"])
        self.assertEqual(
            config["friend_settings"]["schedule_settings"]["mean_position_ratio"],
            0.375,
        )
        self.assertEqual(
            config["friend_settings"]["schedule_settings"]["thought_model"],
            "cheap-provider",
        )
        self.assertEqual(config.saved, 1)
        self.assertFalse(harness._migrate_legacy_gate_config())

    def test_legacy_runtime_fields_are_removed(self) -> None:
        harness = _StorageHarness(
            {
                "default:FriendMessage:1": {
                    "gate_pending": {"trigger_time": 123},
                    "gate_date": "2026-08-10",
                    "gate_calls_today": 2,
                    "thought_silence_used": True,
                }
            }
        )
        self.assertEqual(harness._migrate_legacy_gate_state(), 3)
        payload = harness.session_data["default:FriendMessage:1"]
        self.assertNotIn("gate_pending", payload)
        self.assertNotIn("gate_date", payload)
        self.assertNotIn("gate_calls_today", payload)
        self.assertTrue(payload["thought_silence_used"])

    def test_old_thought_silence_boolean_migrates_to_count(self) -> None:
        harness = _StorageHarness(
            {
                "default:FriendMessage:1": {"thought_silence_used": True},
                "default:FriendMessage:2": {"thought_silence_used": False},
            }
        )
        self.assertEqual(harness._migrate_thought_silence_state(), 2)
        self.assertEqual(
            harness.session_data["default:FriendMessage:1"]["thought_silence_count"],
            1,
        )
        self.assertEqual(
            harness.session_data["default:FriendMessage:2"]["thought_silence_count"],
            0,
        )
        self.assertNotIn(
            "thought_silence_used", harness.session_data["default:FriendMessage:1"]
        )
        self.assertNotIn(
            "thought_silence_used", harness.session_data["default:FriendMessage:2"]
        )

    async def test_session_override_gate_config_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / SessionOverrideManager.OVERRIDES_FILE
            path.write_text(
                json.dumps(
                    {
                        "default:FriendMessage:1": {
                            "gate_settings": {
                                "stickiness": 3,
                                "gate_model": "session-cheap-provider",
                            },
                            "schedule_settings": {
                                "mean_position_ratio": 0.375,
                                "weibull_shape": 1.7,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            manager = SessionOverrideManager(Path(temp_dir))
            self.assertEqual(await manager.remove_legacy_gate_settings(), 1)
            override = manager.get_override("default:FriendMessage:1")
            self.assertNotIn("gate_settings", override)
            self.assertEqual(
                override["schedule_settings"],
                {
                    "mean_position_ratio": 0.375,
                    "weibull_shape": 1.7,
                    "thought_model": "session-cheap-provider",
                },
            )


if __name__ == "__main__":
    unittest.main()
