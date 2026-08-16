from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from core.llm_adapter import LlmMixin
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


class _ThoughtHarness(ThoughtMixin, LlmMixin):
    def __init__(self, context: _ThoughtContext) -> None:
        self.context = context
        self.timezone = ZoneInfo("Asia/Shanghai")
        self.telemetry = None


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


class ThoughtContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.harness = _ThoughtHarness(_ThoughtContext())

    def test_closed_food_thread_falls_outside_24_message_window(self) -> None:
        closed_food_thread = [
            {"role": "user", "content": "我都吃完了"},
            {"role": "assistant", "content": "南瓜的事到底还没说呢"},
            {"role": "user", "content": "切块丢进去炖"},
            {"role": "assistant", "content": "那没完全化，留着块"},
        ]
        latest_exchange = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": f"小猪表情包对话 {index}",
            }
            for index in range(23)
        ]
        latest_exchange.append(
            {"role": "assistant", "content": "你今天挺闲的，书翻了吗"}
        )

        contexts = self.harness._prepare_thought_contexts(
            closed_food_thread + latest_exchange
        )

        self.assertEqual(len(contexts), 24)
        self.assertNotIn("南瓜", "\n".join(item["content"] for item in contexts))
        self.assertEqual(contexts[-1]["content"], "你今天挺闲的，书翻了吗")

    def test_archived_internal_prompt_is_not_reused_as_user_history(self) -> None:
        contexts = self.harness._prepare_thought_contexts(
            [
                {
                    "role": "user",
                    "content": "你可以主动发消息\n[本次主动消息心念]\n- 话题：旧候选",
                },
                {"role": "assistant", "content": "上一条主动消息"},
                {"role": "user", "content": "这才是真实用户消息"},
            ]
        )

        self.assertEqual(
            contexts,
            [
                {"role": "assistant", "content": "上一条主动消息"},
                {"role": "user", "content": "这才是真实用户消息"},
            ],
        )

    def test_thought_prompt_orders_impulse_before_topic_selection(self) -> None:
        prompt = ThoughtMixin.THOUGHT_PROMPT
        self.assertIn("先判断此刻是否真的有一个具体、自然的开口冲动", prompt)
        self.assertIn("没有任何具体、自然想说的东西", prompt)
        self.assertIn("已经回答、吃完、做完", prompt)
        self.assertIn("不得再当成待办或进展追问", prompt)
        self.assertIn("重复催问", prompt)
        self.assertIn("只在决定开口之后考虑候选话题", prompt)
        self.assertLess(prompt.index("先判断此刻"), prompt.index("选题必须遵守"))

    def test_final_generator_may_veto_stale_candidate(self) -> None:
        text = ThoughtMixin._format_thought_text(
            {
                "decision": "speak",
                "reason": "想追问",
                "topic": "备菜进展",
                "anchor": "一小时前聊过做饭",
                "angle": "问切好了没",
            }
        )

        self.assertIn("这只是候选，不是必须执行的指令", text)
        self.assertIn("后来已经回答、完成或结束", text)
        self.assertIn("丢弃它并自行换一个自然话头", text)


class ThoughtGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_failure_returns_none(self) -> None:
        harness = _ThoughtHarness(_ThoughtContext(error=RuntimeError("boom")))
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1", [], "persona", 0, 40, silence_count=0
        )
        self.assertIsNone(result)

    async def test_invalid_model_json_returns_none(self) -> None:
        harness = _ThoughtHarness(_ThoughtContext(result="invalid"))
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1", [], "persona", 0, 40, silence_count=0
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
            silence_count=0,
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
            silence_count=0,
        )
        system_prompt = context.last_generate_kwargs["system_prompt"]
        self.assertTrue(system_prompt.startswith("精炼心念人格"))
        self.assertIn("内部心念任务", system_prompt)

    async def test_first_thought_may_stay_silent_without_natural_impulse(self) -> None:
        context = _ThoughtContext(
            result=(
                '{"decision":"silent","reason":"此刻没有自然想说的",'
                '"topic":"","anchor":"","angle":""}'
            )
        )
        harness = _ThoughtHarness(context)
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1",
            [],
            "persona",
            0,
            40,
            silence_count=0,
        )
        prompt = context.last_generate_kwargs["prompt"]
        self.assertEqual(result["decision"], "silent")
        self.assertIn("没有任何具体、自然想说的东西", prompt)
        self.assertIn("不要为了完成主动功能制造话题", prompt)
        self.assertNotIn("应换一个轻量话题", prompt)

    async def test_second_silence_prompt_has_last_chance(self) -> None:
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
            "persona",
            0,
            40,
            silence_count=1,
        )
        prompt = context.last_generate_kwargs["prompt"]
        self.assertIn("仍有最后一次选择暂不开口的机会", prompt)

    async def test_third_silence_prompt_forces_speak(self) -> None:
        context = _ThoughtContext(
            result=(
                '{"decision":"silent","reason":"还是想忍",'
                '"topic":"","anchor":"","angle":""}'
            )
        )
        harness = _ThoughtHarness(context)
        await harness._generate_inner_thought(
            "default:FriendMessage:1",
            [],
            "persona",
            0,
            40,
            silence_count=2,
        )
        prompt = context.last_generate_kwargs["prompt"]
        self.assertIn("已经克制过两次，这次必须选择 speak", prompt)

    async def test_timeout_returns_none(self) -> None:
        harness = _ThoughtHarness(_ThoughtContext(result=("sleep", 0.05)))
        harness.THOUGHT_TIMEOUT_SECONDS = 0.001
        result = await harness._generate_inner_thought(
            "default:FriendMessage:1", [], "persona", 0, 40, silence_count=0
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
