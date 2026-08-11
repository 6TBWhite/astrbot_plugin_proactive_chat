from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads((ROOT / "_conf_schema.json").read_text("utf-8"))

    def test_upstream_private_defaults_are_preserved(self) -> None:
        friend = self.schema["friend_settings"]["items"]
        schedule = friend["schedule_settings"]["items"]
        self.assertEqual(schedule["min_interval_minutes"]["default"], 30)
        self.assertEqual(schedule["max_interval_minutes"]["default"], 600)
        self.assertEqual(schedule["min_interval_minutes"]["slider"]["step"], 1)
        self.assertEqual(schedule["max_interval_minutes"]["slider"]["step"], 1)
        self.assertEqual(schedule["quiet_hours"]["default"], "1-7")
        self.assertEqual(schedule["max_unanswered_times"]["default"], 4)

    def test_advanced_sliders_are_safe_and_clearly_marked(self) -> None:
        schedule = self.schema["friend_settings"]["items"]["schedule_settings"]["items"]
        ratio = schedule["mean_position_ratio"]
        shape = schedule["weibull_shape"]
        thought_model = schedule["thought_model"]
        thought_persona = schedule["thought_persona_prompt"]
        self.assertEqual(thought_model["default"], "")
        self.assertEqual(thought_model["_special"], "select_provider")
        self.assertIn("最终主动消息仍由当前会话主模型生成", thought_model["hint"])
        self.assertEqual(thought_persona["type"], "text")
        self.assertEqual(thought_persona["default"], "")
        self.assertIn("留空则自动沿用当前对话", thought_persona["hint"])
        self.assertEqual(ratio["default"], 0.375)
        self.assertEqual(ratio["slider"], {"min": 0.2, "max": 0.575, "step": 0.025})
        self.assertEqual(shape["default"], 1.7)
        self.assertEqual(shape["slider"], {"min": 1.5, "max": 2.0, "step": 0.1})
        self.assertIn("特殊配置", ratio["description"])
        self.assertIn("一般不要改", ratio["hint"])
        self.assertIn("特殊配置", shape["description"])
        self.assertIn("一般不要改", shape["hint"])

    def test_group_schedule_schema_is_unchanged(self) -> None:
        group_schedule = self.schema["group_settings"]["items"]["schedule_settings"][
            "items"
        ]
        self.assertNotIn("mean_position_ratio", group_schedule)
        self.assertNotIn("weibull_shape", group_schedule)

    def test_thought_settings_are_private_only(self) -> None:
        group_schedule = self.schema["group_settings"]["items"]["schedule_settings"][
            "items"
        ]
        self.assertNotIn("thought_model", group_schedule)
        self.assertNotIn("thought_persona_prompt", group_schedule)


if __name__ == "__main__":
    unittest.main()
