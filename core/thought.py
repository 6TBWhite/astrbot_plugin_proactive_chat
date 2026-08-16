"""主动消息心念层。

心念只负责在开口前选择话头，并拥有最多两次克制机会；时间由调度分布决定，
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

MAX_THOUGHT_SILENCES = 2


@dataclass(slots=True)
class ThoughtCycleContext:
    """一次定时心念判断所需的稳定快照。"""

    session_id: str
    history_messages: list
    persona_system_prompt: str
    unanswered_count: int
    cycle_started_at: float
    last_message_time: float
    silence_count: int = 0
    thought_provider_id: str = ""
    thought_persona_prompt: str = ""


class ThoughtMixin:
    """为私聊定时任务提供内置的开口前思考（最多两次沉默）。"""

    THOUGHT_TIMEOUT_SECONDS = 45
    THOUGHT_HISTORY_LIMIT = 24
    THOUGHT_PROMPT = """[系统任务：主动消息心念]
现在是 {{current_time}}。距离对方最后一条消息已经过去 {{elapsed_minutes}} 分钟。
我此前连续主动开口但尚未收到回复的次数是 {{unanswered_count}}。
这一次{{silence_instruction}}

你看到的上下文只保留了最近一段真实对话。请结合最近对话和我的人格，先判断此刻是否真的有一个具体、自然的开口冲动。

如果允许克制，出现以下任一情况都可以选择 silent：
1. 此刻开口明显会打扰、冒犯或伤害对方；
2. 此刻没有任何具体、自然想说的东西。

不要为了完成主动功能制造话题，也不要因为主动联系可能对关系有益就选择 speak。

若决定 speak，或这次已经不能继续沉默，再形成一个简短、具体的心念：我想聊什么、为什么会想到它、从哪里切入最自然。

选题必须遵守：
1. 后面的消息代表更新状态，优先于前面的消息。对方已经回答、吃完、做完、看完、解决、取消或明确结束的事，不得再当成待办或进展追问。
2. 如果我最后已经问过一个问题而对方尚未回答，不要换个说法重复催问。
3. 只在决定开口之后考虑候选话题，排除与后文矛盾、已经闭合、只是复述旧话或显得催促的候选。
4. anchor 应对应最近对话中的具体事实，或人格中确实存在的具体兴趣与背景；没有可靠锚点就留空，不得编造。“隔了一会儿”“正好追问”不算锚点。
5. 找不到仍然开放的旧线索时，可以选择一个具体的新话题，不得硬续已经结束的内容。

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

    def _prepare_thought_contexts(self, history_messages: list) -> list:
        """只向心念模型提供最近的真实对话窗口。"""
        contexts = self._sanitize_history_content(list(history_messages or []))
        contexts = [
            message for message in contexts if str(message.get("content") or "").strip()
        ]
        return contexts[-self.THOUGHT_HISTORY_LIMIT :]

    async def _generate_inner_thought(
        self,
        session_id: str,
        history_messages: list,
        persona_system_prompt: str,
        unanswered_count: int,
        elapsed_minutes: float,
        *,
        silence_count: int,
        thought_provider_id: str = "",
    ) -> dict | None:
        """调用当前会话模型生成心念；失败时返回 None 供主流程放行。"""
        remaining = max(0, MAX_THOUGHT_SILENCES - int(silence_count or 0))
        if remaining > 1:
            silence_instruction = f"仍有 {remaining} 次选择暂不开口的机会。"
        elif remaining == 1:
            silence_instruction = "仍有最后一次选择暂不开口的机会。"
        else:
            silence_instruction = (
                "已经克制过两次，这次必须选择 speak，并给出新的自然话头。"
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
            contexts = self._prepare_thought_contexts(history_messages)
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

        返回 ``(是否继续生成, 注入文本)``。克制后重排新任务或用户插话时停止本轮；
        心念失败、到达最晚时间或已经克制满两次时继续生成。
        """
        silence_count = max(0, int(context.silence_count or 0))
        can_stay_silent = silence_count < MAX_THOUGHT_SILENCES
        last_message_time = max(0.0, float(context.last_message_time or 0))
        if last_message_time > 0:
            elapsed_minutes = max(0.0, (time.time() - last_message_time) / 60.0)
        else:
            elapsed_minutes = max(0.0, (time.time() - context.cycle_started_at) / 60.0)
        thought_persona_prompt = str(context.thought_persona_prompt or "").strip()
        thought = await self._generate_inner_thought(
            context.session_id,
            context.history_messages,
            thought_persona_prompt or context.persona_system_prompt,
            context.unanswered_count,
            elapsed_minutes,
            silence_count=silence_count,
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
                f"[主动消息] 心念再次想克制，但本轮机会已用完（已沉默 {MAX_THOUGHT_SILENCES} 次），"
                f"继续生成主动消息喵。理由：{reason or '未提供'}"
            )
            return True, self._format_thought_text(thought)

        logger.info(
            f"[主动消息] 心念：这次先不说喵（第 {silence_count + 1}/{MAX_THOUGHT_SILENCES} 次沉默）。"
            f"理由：{reason or '此刻开口不够合适'}"
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
        """把心念压缩成最终生成器可复核的话头候选。"""
        lines = ["[本次主动消息心念候选]"]
        if thought.get("reason"):
            lines.append(f"- 心念判断：{thought['reason']}")
        if thought.get("topic"):
            lines.append(f"- 话题：{thought['topic']}")
        if thought.get("anchor"):
            lines.append(f"- 锚点：{thought['anchor']}")
        if thought.get("angle"):
            lines.append(f"- 切入角度：{thought['angle']}")
        lines.extend(
            (
                "这只是候选，不是必须执行的指令。请先以最新对话事实复核：",
                "如果该事项后来已经回答、完成或结束，候选与后文矛盾，或者它在重复我上次未获回应的问题，就丢弃它并自行换一个自然话头。",
                "不要向对方解释复核过程；候选有效时，再结合当前人格自然地说出来。",
            )
        )
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
