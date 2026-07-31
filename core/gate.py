"""心动门模块：防冷场增强的 LLM 决策门禁。

调度任务到点后先经此门评估"该不该说、说什么、怎么说"，
通过后才进入原有主动消息流程；被否决或等待时静默重调度，
不增加未回复计数、不打扰用户。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime
from typing import Any

from astrbot.api import logger


class GateMixin:
    """心动门混入类：作为调度任务的代理入口。"""

    GATE_HISTORY_MAX_CHARS = 200
    DEFAULT_WAIT_MINUTES = 30.0

    DESIRE_LEVELS = ("low", "medium", "high")

    _STICKINESS_TABLE: dict[int, list[str]] = {
        1: ["low", "low", "low", "low", "medium", "medium", "high"],
        2: ["low", "low", "low", "medium", "medium", "medium", "high"],
        3: ["low", "low", "medium", "medium", "medium", "high", "high"],
        4: ["low", "medium", "medium", "medium", "high", "high", "high"],
        5: ["medium", "medium", "high", "high", "high", "high", "high"],
    }

    DEFAULT_GATE_PROMPT = """[系统任务：主动消息门禁评估]
你是一个消息门禁评估器。请评估此刻是否应该在私聊中发起一次主动消息。你必须严格输出 JSON，不要输出任何其他内容。

[评估要求]
1. 时间是最重要的信号：冷场越久，越值得开口，语气也可以更有分量；刚聊完不久应避免打扰。
2. 结合最近对话判断：是否有值得延续的话题？上次的话题是否被对方冷落（没接话）？对方是否正忙（对话似乎正在进行中）？有明确值得说的内容时，倾向 send。
3. 渴望度高时优先考虑 send 或较短的 wait；渴望度低时倾向 skip。
4. 未回复次数高时，除非有极强的理由，否则不要 send。

[输出格式]
{
  "decide": "send | wait | skip",
  "reason": "一句话中文理由",
  "impulse": {"topic": "话题名", "anchor": "对话原文或背景，用于生成开场时的锚点"},
  "angle": "切入角度，即这次开口从哪个角度说",
  "wait_minutes": 45
}
说明：
- send：现在就该说，必须提供 impulse 和 angle。
- wait：值得说但此刻时机不合适，请给出 wait_minutes（多少分钟后发送），impulse 和 angle 必须保留，到点后将直接发送。
- skip：不值得说，impulse / angle / wait_minutes 可以留空。
请只输出上述 JSON。

[情景信息]
- 当前时间：{{current_time}}
- 距离上次和对方互动已经过去：{{hours_since_last_chat}} 小时
- 系统评估的当前"开口渴望度"：{{desire_level}}（low=低 / medium=中 / high=高）。渴望度由系统根据冷场时长与粘人度计算：冷场越久、粘人度越高，渴望越高。它代表此刻该有的开口意愿，请把它作为重要的决策信号。
- 我之前已经主动找过对方但没有收到回复的次数：{{unanswered_count}} 次。这个次数越高，开口越要克制，除非有非常充分的理由。

[最近对话参考]
{{history_context}}"""

    def _get_gate_settings(self, session_config: dict | None) -> dict:
        """读取会话生效配置中的门禁设置，缺失时返回空字典。"""
        if not isinstance(session_config, dict):
            return {}
        gate = session_config.get("gate_settings") or {}
        return gate if isinstance(gate, dict) else {}

    def _is_private_session(self, session_id: str) -> bool:
        """判断会话是否为私聊（Friend/Private）。"""
        parsed = self._parse_session_id(session_id)
        return bool(parsed and ("Friend" in parsed[1] or "Private" in parsed[1]))

    def _get_check_entry(self):
        """返回调度任务实际执行的入口（统一代理，内部放行未启用门禁的会话）。"""
        return self.gated_check_and_chat

    async def gated_check_and_chat(self, session_id: str) -> None:
        """门禁代理入口：先评估，再决定执行、等待或静默重调度。"""
        normalized_session_id = self._normalize_session_id(session_id)
        try:
            if not self._is_private_session(normalized_session_id):
                await self.check_and_chat(normalized_session_id)
                return

            session_config = self._get_session_config(normalized_session_id)
            gate_settings = self._get_gate_settings(session_config)
            if not gate_settings.get("enable_gate", False):
                await self.check_and_chat(normalized_session_id)
                return

            is_allowed, block_reason = await self._is_chat_allowed(
                normalized_session_id
            )
            if not is_allowed:
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"不满足基础触发条件（{block_reason}），静默重调度喵。"
                )
                await self._schedule_next_chat_and_save(normalized_session_id)
                return

            async with self.data_lock:
                session_info = self.session_data.get(normalized_session_id, {})
                unanswered = (
                    int(session_info.get("unanswered_count", 0) or 0)
                    if isinstance(session_info, dict)
                    else 0
                )
                max_unanswered = int(
                    (session_config.get("schedule_settings", {}) or {}).get(
                        "max_unanswered_times", 3
                    )
                    or 0
                )
            if max_unanswered > 0 and unanswered >= max_unanswered:
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"未回复次数（{unanswered}）已达上限（{max_unanswered}），跳过门禁评估喵。"
                )
                return

            if not await self._gate_quota_check_and_tick(
                normalized_session_id, gate_settings
            ):
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"今日门禁调用已达上限，静默重调度喵。"
                )
                await self._schedule_next_chat_and_save(normalized_session_id)
                return

            history_text, history_count = await self._load_gate_history(
                normalized_session_id, gate_settings
            )
            if not history_text and self._gate_history_limit(gate_settings) > 0:
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"无可用对话历史（{history_count} 条），静默重调度喵。"
                )
                await self._schedule_next_chat_and_save(normalized_session_id)
                return

            hours_since = self._hours_since_last_chat(normalized_session_id)
            try:
                stickiness = int(gate_settings.get("stickiness", 3) or 3)
            except Exception:
                stickiness = 3
            desire_level = self._compute_desire_level(
                hours_since, unanswered, stickiness
            )
            gate_prompt = self._build_gate_prompt(
                gate_settings,
                history_text,
                hours_since,
                desire_level,
                unanswered,
            )

            raw = await self._call_gate_llm(
                gate_settings, gate_prompt, normalized_session_id
            )
            result = self._parse_gate_json(raw) if raw else None

            if not result:
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"门禁评估失败或无有效结果，按 skip 静默重调度喵。"
                )
                await self._schedule_next_chat_and_save(normalized_session_id)
                return

            decide = result["decide"]
            reason = result.get("reason", "")
            self._track_gate_decision(normalized_session_id, decide)

            if decide == "send":
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"评估通过（send），执行主动消息喵。理由：{reason}"
                )
                await self.check_and_chat(normalized_session_id)
            elif decide == "wait":
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"评估为 wait，{result['wait_minutes']} 分钟后直接发送喵。理由：{reason}"
                )
                await self._reschedule_wait(
                    normalized_session_id,
                    result["wait_minutes"],
                    result.get("topic", ""),
                    result.get("anchor", ""),
                    result.get("angle", ""),
                    reason,
                )
            else:
                logger.info(
                    f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id, session_config)} "
                    f"评估为 skip，静默重调度喵。理由：{reason}"
                )
                await self._schedule_next_chat_and_save(normalized_session_id)

        except Exception as e:
            logger.error(f"[主动消息] 心动门处理异常喵: {e}", exc_info=True)
            try:
                await self._schedule_next_chat_and_save(normalized_session_id)
            except Exception as se:
                logger.error(f"[主动消息] 心动门异常后的重调度也失败喵: {se}")

    async def _send_pending_proactive(self, session_id: str) -> None:
        """wait 到点执行：直接发送已暂存的主动消息（不再过门禁）。"""
        normalized_session_id = self._normalize_session_id(session_id)
        try:
            async with self.data_lock:
                session_info = self.session_data.get(normalized_session_id, {})
                pending = session_info.get("gate_pending")
                if not isinstance(pending, dict) or not pending.get("trigger_time"):
                    logger.debug(
                        f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id)} "
                        f"没有待发送的 wait 任务，跳过喵。"
                    )
                    return
                try:
                    trigger_time = float(pending.get("trigger_time") or 0)
                except Exception:
                    trigger_time = 0.0
                if time.time() - trigger_time > 600:
                    logger.info(
                        f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id)} "
                        f"wait 任务已过期，作废喵。"
                    )
                    self.session_data.setdefault(normalized_session_id, {}).pop(
                        "gate_pending", None
                    )
                    await self._save_data_internal()
                    return
                self.session_data.setdefault(normalized_session_id, {}).pop(
                    "gate_pending", None
                )
                await self._save_data_internal()

            impulse = pending.get("impulse") or {}
            gate_impulse = {
                "topic": str(impulse.get("topic") or "")
                if isinstance(impulse, dict)
                else "",
                "anchor": str(impulse.get("anchor") or "")
                if isinstance(impulse, dict)
                else "",
                "angle": str(pending.get("angle") or ""),
                "reason": str(pending.get("reason") or ""),
            }
            logger.info(
                f"[主动消息] 心动门：{self._get_session_log_str(normalized_session_id)} "
                f"wait 任务到点，直接执行主动消息喵。"
            )
            await self.check_and_chat(normalized_session_id, gate_impulse=gate_impulse)
        except Exception as e:
            logger.error(f"[主动消息] 心动门 wait 任务执行失败喵: {e}", exc_info=True)
            try:
                await self._schedule_next_chat_and_save(normalized_session_id)
            except Exception as se:
                logger.error(f"[主动消息] wait 任务失败后的重调度也失败喵: {se}")

    async def _gate_quota_check_and_tick(
        self, session_id: str, gate_settings: dict
    ) -> bool:
        """检查并记录当日门禁调用配额（跨天自动重置）。"""
        try:
            limit = int(gate_settings.get("daily_gate_calls_limit", 3) or 0)
        except Exception:
            limit = 3
        today = datetime.now(self.timezone).strftime("%Y-%m-%d")

        async with self.data_lock:
            payload = self.session_data.setdefault(session_id, {})
            if payload.get("gate_date") != today:
                payload["gate_date"] = today
                payload["gate_calls_today"] = 0
            calls = int(payload.get("gate_calls_today", 0) or 0)
            if limit > 0 and calls >= limit:
                return False
            payload["gate_calls_today"] = calls + 1
            await self._save_data_internal()
            return True

    async def _clear_gate_state(self, session_id: str) -> None:
        """清除会话的门禁状态（用户回复后调用，恢复完整配额）。"""
        async with self.data_lock:
            payload = self.session_data.get(session_id)
            if not isinstance(payload, dict):
                return
            changed = False
            for key in ("gate_date", "gate_calls_today", "gate_pending"):
                if key in payload:
                    del payload[key]
                    changed = True
            if changed:
                await self._save_data_internal()

    def _hours_since_last_chat(self, session_id: str) -> float:
        """计算距上次互动的小时数（无记录时从插件启动时间起算）。"""
        last_ts = 0.0
        session_info = self.session_data.get(session_id, {})
        if isinstance(session_info, dict):
            try:
                last_ts = float(session_info.get("last_message_time", 0) or 0)
            except Exception:
                last_ts = 0.0
        if not last_ts:
            last_ts = float(self.last_message_times.get(session_id, 0) or 0)
        if not last_ts:
            last_ts = float(getattr(self, "plugin_start_time", time.time()))
        return max(0.0, (time.time() - last_ts) / 3600.0)

    def _compute_desire_level(
        self, hours_since: float, unanswered_count: int, stickiness: int
    ) -> str:
        """按时长分档与粘人度计算渴望度档位（未回复次数高时降档）。"""
        if hours_since < 1:
            hour_index = 0
        elif hours_since < 2:
            hour_index = 1
        elif hours_since < 4:
            hour_index = 2
        elif hours_since < 8:
            hour_index = 3
        elif hours_since < 12:
            hour_index = 4
        elif hours_since < 24:
            hour_index = 5
        else:
            hour_index = 6

        table = self._STICKINESS_TABLE.get(stickiness, self._STICKINESS_TABLE[3])
        level = table[hour_index]
        level_index = self.DESIRE_LEVELS.index(level)
        if unanswered_count >= 2:
            level_index = max(0, level_index - 1)
        if unanswered_count >= 4:
            level_index = max(0, level_index - 1)
        return self.DESIRE_LEVELS[level_index]

    def _build_gate_prompt(
        self,
        gate_settings: dict,
        history_text: str,
        hours_since: float,
        desire_level: str,
        unanswered_count: int,
    ) -> str:
        """填充门禁提示词模板。"""
        template = str(gate_settings.get("gate_prompt") or "").strip()
        if not template:
            template = self.DEFAULT_GATE_PROMPT
        now_str = datetime.now(self.timezone).strftime("%Y年%m月%d日 %H:%M")
        hours_str = f"{hours_since:.1f}"
        return (
            template.replace("{{current_time}}", now_str)
            .replace("{{hours_since_last_chat}}", hours_str)
            .replace("{{desire_level}}", desire_level)
            .replace("{{unanswered_count}}", str(unanswered_count))
            .replace("{{history_context}}", history_text or "(未启用对话历史参考)")
        )

    def _gate_history_limit(self, gate_settings: dict) -> int:
        """门评估可见的对话条数（独立配置）。设为 0 表示不参考对话历史。"""
        try:
            count = int(gate_settings.get("gate_history_count", 20) or 0)
        except Exception:
            count = 20
        if count <= 0:
            return 0
        return max(1, min(count, 50))

    async def _load_gate_history(
        self, session_id: str, gate_settings: dict
    ) -> tuple[str, int]:
        """读取最近对话历史用于门禁评估（不创建新会话）。

        配置为 0 时跳过对话读取，门评估仅凭时间与渴望度判断。
        """
        history_limit = self._gate_history_limit(gate_settings)
        if history_limit <= 0:
            return "", 0
        try:
            candidates = [session_id]
            try:
                normalized = self._normalize_session_id(session_id)
                if normalized and normalized not in candidates:
                    candidates.append(normalized)
            except Exception:
                pass

            conv_id = None
            for candidate in candidates:
                conv_id = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        candidate
                    )
                )
                if conv_id:
                    break
            if not conv_id:
                return "", 0

            conversation = await self.context.conversation_manager.get_conversation(
                session_id, conv_id
            )
            history = []
            if conversation and conversation.history:
                try:
                    if isinstance(conversation.history, str):
                        history = await asyncio.to_thread(
                            json.loads, conversation.history
                        )
                    else:
                        history = conversation.history
                except (json.JSONDecodeError, TypeError):
                    history = []

            if not isinstance(history, list) or not history:
                return "", len(history) if isinstance(history, list) else 0

            lines: list[str] = []
            for msg in history[-history_limit:]:
                role, text = self._textify_gate_message(msg)
                if not text:
                    continue
                if len(text) > self.GATE_HISTORY_MAX_CHARS:
                    text = text[: self.GATE_HISTORY_MAX_CHARS] + "..."
                lines.append(f"{role}: {text}")

            return "\n".join(lines), len(history)
        except Exception as e:
            logger.warning(f"[主动消息] 心动门读取对话历史失败喵: {e}")
            return "", 0

    def _textify_gate_message(self, msg: Any) -> tuple[str, str]:
        """将历史消息转为 (role, text) 纯文本对。"""
        if hasattr(msg, "to_dict"):
            try:
                msg = msg.to_dict()
            except Exception:
                msg = {}
        if isinstance(msg, dict):
            role = str(msg.get("role") or "user")
            content = msg.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for segment in content:
                    if isinstance(segment, dict) and segment.get("type") == "text":
                        parts.append(str(segment.get("text") or ""))
                    elif isinstance(segment, str):
                        parts.append(segment)
                text = "".join(parts).strip()
            elif isinstance(content, str):
                text = content.strip()
            else:
                text = str(content).strip() if content is not None else ""
            return role, text
        return "user", str(msg).strip()

    async def _call_gate_llm(
        self, gate_settings: dict, prompt: str, session_id: str
    ) -> str | None:
        """调用 LLM 执行门禁评估（异常时返回 None，由调用方按 skip 处理）。"""
        try:
            provider_id = str(gate_settings.get("gate_model") or "").strip()
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    session_id
                )
            llm_response_obj = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
                contexts=[],
                system_prompt="你是严格的 JSON 输出器。只输出一个 JSON 对象，不要输出任何其他内容。",
            )
            if llm_response_obj and llm_response_obj.completion_text:
                return llm_response_obj.completion_text.strip()
        except Exception as e:
            logger.error(f"[主动消息] 心动门 LLM 调用失败喵: {e}")
        return None

    def _parse_gate_json(self, raw: str) -> dict | None:
        """容错解析门禁 JSON 输出，无效时返回 None。"""
        text = raw.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
        try:
            data = json.loads(text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None

        decide = str(data.get("decide") or "").strip().lower()
        if decide not in {"send", "wait", "skip"}:
            return None

        result: dict = {
            "decide": decide,
            "reason": str(data.get("reason") or "").strip(),
        }

        impulse = data.get("impulse")
        result["topic"] = ""
        result["anchor"] = ""
        if isinstance(impulse, dict):
            result["topic"] = str(impulse.get("topic") or "").strip()
            result["anchor"] = str(impulse.get("anchor") or "").strip()
        result["angle"] = str(data.get("angle") or "").strip()

        raw_wait = data.get("wait_minutes")
        try:
            if raw_wait is None or str(raw_wait).strip() == "":
                wait_minutes = self.DEFAULT_WAIT_MINUTES
            else:
                wait_minutes = float(raw_wait)
            result["wait_minutes"] = max(1.0, min(wait_minutes, 2880.0))
        except Exception:
            result["wait_minutes"] = self.DEFAULT_WAIT_MINUTES
        return result

    async def _reschedule_wait(
        self,
        session_id: str,
        wait_minutes: float,
        topic: str,
        anchor: str,
        angle: str,
        reason: str,
    ) -> None:
        """暂存 wait 任务并调度到点直接发送。"""
        try:
            run_date = datetime.fromtimestamp(
                time.time() + wait_minutes * 60, tz=self.timezone
            )
            async with self.data_lock:
                payload = self.session_data.setdefault(session_id, {})
                payload["gate_pending"] = {
                    "trigger_time": time.time() + wait_minutes * 60,
                    "impulse": {"topic": topic or "", "anchor": anchor or ""},
                    "angle": angle or "",
                    "reason": reason or "",
                }
                await self._save_data_internal()

            self.scheduler.add_job(
                self._send_pending_proactive,
                "date",
                run_date=run_date,
                args=[session_id],
                id=session_id,
                replace_existing=True,
                misfire_grace_time=60,
            )
            logger.info(
                f"[主动消息] 心动门：已为 {self._get_session_log_str(session_id)} "
                f"安排 wait 任务喵，将在 {run_date.strftime('%Y-%m-%d %H:%M:%S')} 直接发送。"
            )
        except Exception as e:
            logger.error(f"[主动消息] 心动门安排 wait 任务失败喵: {e}")
            try:
                await self._schedule_next_chat_and_save(session_id)
            except Exception:
                pass

    def _format_gate_impulse_text(self, gate_impulse: dict) -> str:
        """将门禁评估结果格式化为注入生成阶段的参考文本。"""
        lines = ["[本次主动消息评估]"]
        reason = str(gate_impulse.get("reason") or "").strip()
        topic = str(gate_impulse.get("topic") or "").strip()
        anchor = str(gate_impulse.get("anchor") or "").strip()
        angle = str(gate_impulse.get("angle") or "").strip()
        if reason:
            lines.append(f"- 系统已决定发起本次主动消息，理由：{reason}")
        if topic:
            lines.append(f"- 话题：{topic}")
        if anchor:
            lines.append(f"- 锚点：{anchor}")
        if angle:
            lines.append(f"- 切入角度：{angle}")
        lines.append("请基于以上评估内容，结合人格设定，自然地说出开场白。")
        return "\n".join(lines)

    def _track_gate_decision(self, session_id: str, decide: str) -> None:
        """上报门禁决策遥测（仅统计，不包含对话内容）。"""
        if not (self.telemetry and self.telemetry.enabled):
            return
        try:
            self._track_task(
                asyncio.create_task(
                    self.telemetry.track_feature(
                        "gate_decision",
                        {"decide": decide, "session_type": "friend"},
                    )
                )
            )
        except Exception:
            pass
