from __future__ import annotations

import asyncio
import sys
import time
import types
import unittest
from pathlib import Path

# 仓库目录名带有 ``-main``，不能直接作为 Python 包名导入。测试中建立一个
# 只读包别名，让 chat_flow 的 ``..utils`` 相对导入与 AstrBot 实机一致。
_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_NAME = "proactive_chat_test_package"
if _PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(_PACKAGE_NAME)
    package.__path__ = [str(_ROOT)]
    sys.modules[_PACKAGE_NAME] = package

from proactive_chat_test_package.core.chat_flow import ProactiveCoreMixin
from proactive_chat_test_package.core.thought import ThoughtMixin


class _ChatFlowHarness(ProactiveCoreMixin, ThoughtMixin):
    SESSION_ID = "default:FriendMessage:1"

    def __init__(self) -> None:
        self.data_lock = asyncio.Lock()
        self.session_data = {
            self.SESSION_ID: {
                "unanswered_count": 0,
                "thought_cycle_started_at": time.time() - 60 * 60,
                "thought_silence_used": False,
            }
        }
        self.last_message_times = {self.SESSION_ID: 0}
        self.telemetry = None
        self.manual_trigger_sessions = set()
        self.web_admin_server = None
        self.thought_result = {
            "decision": "speak",
            "reason": "想接着聊",
            "topic": "上次的话题",
            "anchor": "最近对话",
            "angle": "自然问问进展",
        }
        self.generated_thought_text = None
        self.sent = []
        self.finalized = 0
        self.reschedule_result = "scheduled"
        self.reschedule_calls = 0
        self.thought_calls = 0
        self.scheduled_new_cycles = 0
        self.update_message_during_thought = False
        self.thought_model = ""
        self.thought_persona_prompt = ""
        self.received_thought_provider = None
        self.received_thought_persona_prompt = None

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        return session_id

    @staticmethod
    def _parse_session_id(session_id: str):
        parts = str(session_id).split(":", 2)
        return tuple(parts) if len(parts) == 3 else None

    def _is_private_session_id(self, session_id: str) -> bool:
        parsed = self._parse_session_id(session_id)
        return bool(parsed and "Friend" in parsed[1])

    async def _is_chat_allowed(self, _session_id: str):
        return True, "allowed"

    def _get_session_config(self, _session_id: str):
        return {
            "enable": True,
            "schedule_settings": {
                "max_unanswered_times": 4,
                "thought_model": self.thought_model,
                "thought_persona_prompt": self.thought_persona_prompt,
            },
        }

    @staticmethod
    def _get_session_log_str(session_id: str, _config=None) -> str:
        return session_id

    async def _prepare_llm_request(self, session_id: str):
        return {
            "conv_id": "conversation",
            "history": [],
            "system_prompt": "persona",
            "session_id": session_id,
        }

    async def _generate_inner_thought(self, *_args, **_kwargs):
        self.thought_calls += 1
        self.received_thought_persona_prompt = _args[2]
        self.received_thought_provider = _kwargs.get("thought_provider_id")
        if self.update_message_during_thought:
            self.last_message_times[self.SESSION_ID] = 1
        return self.thought_result

    @staticmethod
    def _format_thought_text(thought: dict) -> str:
        return f"[本次主动消息心念]\n- 话题：{thought.get('topic', '')}"

    @staticmethod
    def _track_thought_decision(*_args, **_kwargs) -> None:
        return None

    async def _reschedule_after_thought_silence(self, *_args, **_kwargs):
        self.reschedule_calls += 1
        return self.reschedule_result

    async def _generate_llm_response(
        self, _session_id, _config, _history, _system, _unanswered, *, thought_text=""
    ):
        self.generated_thought_text = thought_text
        return "主动消息", "生成提示"

    async def _send_proactive_message(self, _session_id: str, text: str) -> None:
        self.sent.append(text)

    async def _finalize_and_reschedule(self, *_args, **_kwargs) -> None:
        self.finalized += 1

    async def _schedule_next_chat_and_save(self, _session_id: str) -> None:
        self.scheduled_new_cycles += 1

    @staticmethod
    def _clear_session_schedule_state(_session_id: str) -> bool:
        return False

    async def _save_data_internal(self) -> None:
        return None


class ChatFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_speak_injects_topic_into_final_generator(self) -> None:
        harness = _ChatFlowHarness()
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(harness.thought_calls, 1)
        self.assertIn("上次的话题", harness.generated_thought_text)
        self.assertEqual(harness.sent, ["主动消息"])
        self.assertEqual(harness.finalized, 1)

    async def test_first_silence_reschedules_without_sending(self) -> None:
        harness = _ChatFlowHarness()
        harness.thought_result = {
            "decision": "silent",
            "reason": "对方可能正忙",
            "topic": "稍后问候",
            "anchor": "最近对话",
            "angle": "轻轻问候",
        }
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(harness.reschedule_calls, 1)
        self.assertIsNone(harness.generated_thought_text)
        self.assertEqual(harness.sent, [])

    async def test_second_silence_cannot_delay_and_uses_latest_topic(self) -> None:
        harness = _ChatFlowHarness()
        harness.session_data[harness.SESSION_ID]["thought_silence_used"] = True
        harness.thought_result = {
            "decision": "silent",
            "reason": "仍有点犹豫",
            "topic": "第二次重新选择的话题",
            "anchor": "最新对话",
            "angle": "简短开口",
        }
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(harness.reschedule_calls, 0)
        self.assertIn("第二次重新选择的话题", harness.generated_thought_text)
        self.assertEqual(harness.sent, ["主动消息"])

    async def test_invalid_thought_fails_open(self) -> None:
        harness = _ChatFlowHarness()
        harness.thought_result = None
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(harness.generated_thought_text, "")
        self.assertEqual(harness.sent, ["主动消息"])

    async def test_user_message_during_thought_discards_result(self) -> None:
        harness = _ChatFlowHarness()
        harness.update_message_during_thought = True
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertIsNone(harness.generated_thought_text)
        self.assertEqual(harness.sent, [])
        self.assertEqual(harness.reschedule_calls, 0)

    async def test_manual_trigger_bypasses_thought(self) -> None:
        harness = _ChatFlowHarness()
        await harness.check_and_chat(harness.SESSION_ID)
        self.assertEqual(harness.thought_calls, 0)
        self.assertEqual(harness.generated_thought_text, "")
        self.assertEqual(harness.sent, ["主动消息"])

    async def test_dedicated_thought_model_is_forwarded(self) -> None:
        harness = _ChatFlowHarness()
        harness.thought_model = "cheap-provider"
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(harness.received_thought_provider, "cheap-provider")

    async def test_compact_thought_persona_overrides_conversation_persona(self) -> None:
        harness = _ChatFlowHarness()
        harness.thought_persona_prompt = "精炼心念人格"
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(
            harness.received_thought_persona_prompt,
            "精炼心念人格",
        )

    async def test_blank_thought_persona_reuses_conversation_persona(self) -> None:
        harness = _ChatFlowHarness()
        await harness.check_and_chat(harness.SESSION_ID, use_thought=True)
        self.assertEqual(harness.received_thought_persona_prompt, "persona")


if __name__ == "__main__":
    unittest.main()
