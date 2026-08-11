from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from core.thought import ThoughtMixin


class _ThoughtContext:
    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.last_generate_kwargs = None

    async def get_current_chat_provider_id(self, _session_id: str) -> str:
        return "current-provider"

    async def llm_generate(self, **_kwargs):
        self.last_generate_kwargs = _kwargs
        if self.error:
            raise self.error
        if isinstance(self.result, tuple) and self.result[0] == "sleep":
            await asyncio.sleep(self.result[1])
            return SimpleNamespace(completion_text="{}")
        return SimpleNamespace(completion_text=self.result)


class _ThoughtHarness(ThoughtMixin):
    def __init__(self, context: _ThoughtContext) -> None:
        self.context = context
        self.timezone = ZoneInfo("Asia/Shanghai")
        self.telemetry = None

    @staticmethod
    def _sanitize_history_content(history: list) -> list:
        return history


class ThoughtParserTests(unittest.TestCase):
    def test_fixed_top_level_shape_is_parsed(self) -> None:
        thought = ThoughtMixin._parse_thought_json(
            'prefix {"decision":"speak","reason":"想问问",'
            '"topic":"研究进展","anchor":"上次在看论文","angle":"轻轻问一句"} suffix'
        )
        self.assertEqual(
            thought,
            {
                "decision": "speak",
                "reason": "想问问",
                "topic": "研究进展",
                "anchor": "上次在看论文",
                "angle": "轻轻问一句",
            },
        )

    def test_invalid_json_fails_open(self) -> None:
        self.assertIsNone(ThoughtMixin._parse_thought_json("not json"))
        self.assertIsNone(ThoughtMixin._parse_thought_json('{"decision":"wait"}'))

    def test_noncanonical_decision_is_rejected(self) -> None:
        self.assertIsNone(
            ThoughtMixin._parse_thought_json('{"decision":"skip","topic":"研究进展"}')
        )


class ThoughtGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_failure_returns_none(self) -> None:
        harness = _ThoughtHarness(_ThoughtContext(error=RuntimeError("boom")))
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1", [], "persona", 0, 40, can_stay_silent=True
        )
        self.assertIsNone(result)

    async def test_invalid_model_json_returns_none(self) -> None:
        harness = _ThoughtHarness(_ThoughtContext(result="invalid"))
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1", [], "persona", 0, 40, can_stay_silent=True
        )
        self.assertIsNone(result)

    async def test_dedicated_model_overrides_current_provider(self) -> None:
        context = _ThoughtContext(
            result=(
                '{"decision":"speak","reason":"想聊",'
                '"topic":"话题","anchor":"锚点","angle":"切入"}'
            )
        )
        harness = _ThoughtHarness(context)
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1",
            [],
            "persona",
            0,
            40,
            can_stay_silent=True,
            thought_provider_id="cheap-provider",
        )
        self.assertEqual(result["decision"], "speak")
        self.assertEqual(
            context.last_generate_kwargs["chat_provider_id"],
            "cheap-provider",
        )

    async def test_persona_prompt_is_used_as_system_prompt(self) -> None:
        context = _ThoughtContext(
            result=(
                '{"decision":"speak","reason":"想聊",'
                '"topic":"话题","anchor":"锚点","angle":"切入"}'
            )
        )
        harness = _ThoughtHarness(context)
        await harness._generate_inner_thought(
            "default:FriendMessage:1",
            [],
            "精炼心念人格",
            0,
            40,
            can_stay_silent=True,
        )
        system_prompt = context.last_generate_kwargs["system_prompt"]
        self.assertTrue(system_prompt.startswith("精炼心念人格"))
        self.assertIn("内部心念任务", system_prompt)

    async def test_timeout_returns_none(self) -> None:
        harness = _ThoughtHarness(_ThoughtContext(result=("sleep", 0.05)))
        harness.THOUGHT_TIMEOUT_SECONDS = 0.001
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1", [], "persona", 0, 40, can_stay_silent=True
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
