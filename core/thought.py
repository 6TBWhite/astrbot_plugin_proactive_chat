"""主动消息心念层。

心念只负责在开口前选择话头，并拥有一次克制机会；时间由调度分布决定，
心念本身不能指定额外等待分钟数。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime

from astrbot.api import logger


@dataclass(slots=True)
class ThoughtCycleContext:
    """一次定时心念判断所需的稳定快照。"""

    session_id: str
    history_messages: list
    persona_system_prompt: str
    unanswered_count: int
    cycle_started_at: float
    silence_used: bool
    last_message_time: float
    thought_provider_id: str = ""
    thought_persona_prompt: str = ""


class ThoughtMixin:
    """为私聊定时任务提供一次内置的开口前思考。"""

    THOUGHT_TIMEOUT_SECONDS = 45
    THOUGHT_PROMPT = """[系统任务：主动消息心念]
现在是 {{current_time}}。距离本轮沉默周期开始已经过去 {{elapsed_minutes}} 分钟。
我此前连续主动开口但尚未收到回复的次数是 {{unanswered_count}}。
这一次{{silence_instruction}}

请结合最近对话和我的人格，先形成一个简短、具体的心念：我现在想聊什么、为什么会想到它、从哪里切入最自然。
如果允许克制，只有在此刻开口明显会打扰、冒犯或伤害对方时才能选择 silent；仅仅“没有特别好的话题”不构成沉默理由，应换一个轻量话题。

只输出一个 JSON 对象：
{
  "decision": "speak | silent",
  "reason": "一句简短中文理由",
  "topic": "想说的话题",
  "anchor": "触发这个念头的最近对话或背景",
  "angle": "自然切入角度"
}
不要输出 JSON 之外的内容。"""

    def _get_check_entry(self):
        """调度任务统一经过心念入口；群聊在入口内直接放行。"""
        return self.thoughtful_check_and_chat

    async def thoughtful_check_and_chat(self, session_id: str) -> None:
        normalized_session_id = self._normalize_session_id(session_id)
        await self.check_and_chat(
            normalized_session_id,
            use_thought=self._is_private_session_id(normalized_session_id),
        )

    async def _generate_inner_thought(
        self,
        session_id: str,
        history_messages: list,
        persona_system_prompt: str,
        unanswered_count: int,
        elapsed_minutes: float,
        *,
        can_stay_silent: bool,
        thought_provider_id: str = "",
    ) -> dict | None:
        """调用当前会话模型生成心念；失败时返回 None 供主流程放行。"""
        silence_instruction = (
            "仍有一次选择暂不开口的机会。"
            if can_stay_silent
            else "已经克制过一次，这次必须选择 speak，并给出新的自然话头。"
        )
        prompt = (
            self.THOUGHT_PROMPT.replace(
                "{{current_time}}",
                datetime.now(self.timezone).strftime("%Y年%m月%d日 %H:%M"),
            )
            .replace("{{elapsed_minutes}}", f"{elapsed_minutes:.1f}")
            .replace("{{unanswered_count}}", str(unanswered_count))
            .replace("{{silence_instruction}}", silence_instruction)
        )
        strict_output_rule = (
            "[内部心念任务]\n"
            "以上人格仍然有效。当前任务只需返回规定的 JSON 对象，不要直接对用户说话。"
        )
        system_prompt = str(persona_system_prompt or "").strip()
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{strict_output_rule}"
        else:
            system_prompt = strict_output_rule

        try:
            provider_id = str(thought_provider_id or "").strip()
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    session_id
                )
            contexts = self._sanitize_history_content(list(history_messages or []))
            response = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                    contexts=contexts,
                    system_prompt=system_prompt,
                ),
                timeout=self.THOUGHT_TIMEOUT_SECONDS,
            )
            if not response or not response.completion_text:
                return None
            return self._parse_thought_json(response.completion_text)
        except Exception as exc:
            logger.warning(f"[主动消息] 心念生成失败，回退原主动生成流程喵: {exc}")
            return None

    def _is_thought_cycle_stale(self, context: ThoughtCycleContext) -> bool:
        """判断心念期间是否已有用户消息开启了新周期。"""
        return (
            self.last_message_times.get(context.session_id, 0)
            > context.last_message_time
            or self.session_data.get(context.session_id, {}).get("unanswered_count", 0)
            < context.unanswered_count
        )

    async def _run_scheduled_thought(
        self, context: ThoughtCycleContext
    ) -> tuple[bool, str]:
        """执行定时心念。

        返回 ``(是否继续生成, 注入文本)``。首次克制或用户插话时停止本轮；
        心念失败、到达最晚时间或已经克制过时继续生成。
        """
        can_stay_silent = not context.silence_used
        elapsed_minutes = max(0.0, (time.time() - context.cycle_started_at) / 60.0)
        thought_persona_prompt = str(context.thought_persona_prompt or "").strip()
        thought = await self._generate_inner_thought(
            context.session_id,
            context.history_messages,
            thought_persona_prompt or context.persona_system_prompt,
            context.unanswered_count,
            elapsed_minutes,
            can_stay_silent=can_stay_silent,
            thought_provider_id=context.thought_provider_id,
        )

        if self._is_thought_cycle_stale(context):
            logger.info("[主动消息] 心念期间检测到用户新消息，本次心念已作废喵。")
            return False, ""

        if not thought:
            logger.info("[主动消息] 心念无有效结果，已放行原主动生成流程喵。")
            return True, ""

        decision = thought["decision"]
        reason = thought.get("reason", "")
        self._track_thought_decision(
            context.session_id,
            decision,
            can_stay_silent=can_stay_silent,
        )

        if decision == "speak":
            logger.info(
                f"[主动消息] 心念：现在想说话喵。话题：{thought.get('topic') or '由生成器自然选择'}；"
                f"理由：{reason or '想自然地继续聊下去'}"
            )
            return True, self._format_thought_text(thought)

        if not can_stay_silent:
            logger.info(
                f"[主动消息] 心念再次想克制，但本轮机会已用完，继续生成主动消息喵。"
                f"理由：{reason or '未提供'}"
            )
            return True, self._format_thought_text(thought)

        logger.info(
            f"[主动消息] 心念：这次先不说喵。理由：{reason or '此刻开口不够合适'}"
        )
        result = await self._reschedule_after_thought_silence(
            context.session_id,
            expected_last_message_time=context.last_message_time,
        )
        if result in {"scheduled", "stale"}:
            return False, ""

        logger.info("[主动消息] 心念已到本轮最晚时间，不再推迟，继续生成主动消息喵。")
        return True, self._format_thought_text(thought)

    @staticmethod
    def _parse_thought_json(raw: str) -> dict | None:
        """容错解析模型输出，并限制注入主生成器的文本长度。"""
        text = str(raw or "").strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None

        decision = str(data.get("decision") or "").strip().lower()
        if decision not in {"speak", "silent"}:
            return None

        def _trim(value: object, limit: int) -> str:
            return str(value or "").strip()[:limit]

        return {
            "decision": decision,
            "reason": _trim(data.get("reason"), 240),
            "topic": _trim(data.get("topic"), 160),
            "anchor": _trim(data.get("anchor"), 400),
            "angle": _trim(data.get("angle"), 240),
        }

    @staticmethod
    def _format_thought_text(thought: dict) -> str:
        """把心念压缩成最终生成器可参考的话头。"""
        lines = ["[本次主动消息心念]"]
        if thought.get("reason"):
            lines.append(f"- 心念判断：{thought['reason']}")
        if thought.get("topic"):
            lines.append(f"- 话题：{thought['topic']}")
        if thought.get("anchor"):
            lines.append(f"- 锚点：{thought['anchor']}")
        if thought.get("angle"):
            lines.append(f"- 切入角度：{thought['angle']}")
        lines.append("请结合当前人格和最新上下文，自然地把这个念头说出来。")
        return "\n".join(lines)

    def _track_thought_decision(
        self, session_id: str, decision: str, *, can_stay_silent: bool
    ) -> None:
        """仅上报决策类型，不上传理由和对话正文。"""
        if not (self.telemetry and self.telemetry.enabled):
            return
        try:
            self._track_task(
                asyncio.create_task(
                    self.telemetry.track_feature(
                        "thought_decision",
                        {
                            "decision": decision,
                            "session_type": "friend",
                            "can_stay_silent": can_stay_silent,
                        },
                    )
                )
            )
        except Exception as exc:
            logger.debug(f"[主动消息] 心念遥测上报失败喵: {exc}")
