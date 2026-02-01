# -*- coding: utf-8 -*-
"""
dialogue_viewer.py

用法示例:

    from dialogue_viewer import process_batch

    # json_obj 结构类似:
    # {
    #   "messages": [
    #       { "messages": [ { "role": "system", "content": "...", "tool_calls": None }, ... ] },
    #       ...
    #   ]
    # }

    structured, printer = process_batch(json_obj)

    # structured 是结构化后的对话数据 (纯 dict, 可直接 json.dump)
    # printer 是一个函数, 用来打印任意一条对话:
    printer(0)  # 打印第 0 条
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable


# ---------------------
# 数据结构定义
# ---------------------

@dataclass
class ToolCallView:
    name: str
    arguments: Dict[str, Any]


@dataclass
class TurnView:
    role: str
    content: str
    tool_calls: Optional[List[ToolCallView]] = None


@dataclass
class DialogueView:
    turns: List[TurnView]


# ---------------------
# 解析工具函数
# ---------------------

def _format_tool_calls(tool_calls_raw: Any) -> List[ToolCallView]:
    """
    将 Message.tool_calls 转成结构化的 ToolCallView 列表。

    兼容：
    1) openai 样式对象: tc.function.name, tc.function.arguments
    2) dict 样式:
       {
         "type": "function",
         "function": {"name": "...", "arguments": {...}}
       }
    """
    if not tool_calls_raw:
        return []

    formatted: List[ToolCallView] = []

    for tc in tool_calls_raw:
        # 情况 1: 对象形式
        func = getattr(tc, "function", None)
        if func is not None:
            name = getattr(func, "name", "")
            arguments = getattr(func, "arguments", {}) or {}
            formatted.append(ToolCallView(name=name, arguments=arguments))
            continue

        # 情况 2: dict 形式
        if isinstance(tc, dict):
            func = tc.get("function") or {}
            name = func.get("name", "")
            arguments = func.get("arguments", {}) or {}
            formatted.append(ToolCallView(name=name, arguments=arguments))
            continue

        # 其他形式可按需处理
        # print(f"[WARN] Unrecognized tool_call format: {tc!r}")

    return formatted


def _structure_single_dialogue(dialog_messages: List[Dict[str, Any]]) -> DialogueView:
    """
    输入: 一条样本的 messages 列表 (每个元素是 dict 或拥有 role/content/tool_calls 属性的对象)
    输出: DialogueView
    """
    turns: List[TurnView] = []

    for m in dialog_messages:
        # print(m)
        # input("Press Enter to continue...")
        if isinstance(m, dict):
            role = m["role"]
            content = m["content"]
            tool_calls_raw = m.get("tool_calls")
        else:
            role = getattr(m, "role")
            content = getattr(m, "content")
            tool_calls_raw = getattr(m, "tool_calls", None)

        tool_calls = _format_tool_calls(tool_calls_raw) if tool_calls_raw else None
        turns.append(TurnView(role=role, content=content, tool_calls=tool_calls))

    return DialogueView(turns=turns)


def _structure_batch(non_tensor_batch: Dict[str, Any]) -> List[DialogueView]:
    """
    输入: non_tensor_batch, 形如
        {
          "messages": [
            { "messages": [ {...}, {...}, ... ] },
            ...
          ]
        }
    输出: DialogueView 列表
    """
    batch_dialogs_raw = non_tensor_batch["messages"]
    structured: List[DialogueView] = []

    for sample in batch_dialogs_raw:
        dialog_messages = sample["messages"]
        dv = _structure_single_dialogue(dialog_messages)
        structured.append(dv)

    return structured


def _pretty_print_dialogue(dialogue: DialogueView,
                           max_content_len: int = 800) -> str:
    """
    以较易读的形式生成一条对话的字符串。
    如果内容被截断，会在该 turn 下方打印清晰的截断提示。
    """
    lines: List[str] = []
    for i, turn in enumerate(dialogue.turns):
        header = f"[{i:02d}] Role: {turn.role}"
        lines.append(header)

        content = turn.content or ""
        truncated = len(content) > max_content_len
        if truncated:
            content_show = content[:max_content_len]
        else:
            content_show = content

        if content_show.strip():
            lines.append("  Content:")
            for line in content_show.splitlines():
                lines.append("    " + line)

        # 截断提示
        if truncated:
            lines.append(
                f"  [CONTENT TRUNCATED] original_length={len(content)}, "
                f"shown={max_content_len}"
            )

        if turn.tool_calls:
            lines.append("  Tool Calls:")
            for tc in turn.tool_calls:
                lines.append(f"    - name: {tc.name}")
                try:
                    args_str = json.dumps(tc.arguments, ensure_ascii=False)
                except TypeError:
                    args_str = str(tc.arguments)
                lines.append(f"      arguments: {args_str}")

        lines.append("")  # 空行分隔

    return "\n".join(lines)



def process_batch(json_obj: Dict[str, Any],
                  max_content_len: int = 2000
                  ) -> Tuple[List[Dict[str, Any]], Callable[[int], None]]:
    """
    主入口。

    参数:
        json_obj: 你的 non_tensor_batch 对象, 至少包含:
                  { "messages": [ { "messages": [ {...}, ... ] }, ... ] }
        max_content_len: 打印时每个 turn 的 content 截断长度.

    返回:
        structured: 结构化后的对话数据 (列表, 内部是纯 dict, 可直接 json.dump)
        printer: 一个函数 printer(idx), 打印第 idx 条对话。
                 如果某个 turn 被截断，会在输出中带 [CONTENT TRUNCATED] 提示。
    """
    dialogues = _structure_batch(json_obj)

    structured_as_dicts: List[Dict[str, Any]] = []
    for dv in dialogues:
        structured_as_dicts.append({
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in (t.tool_calls or [])
                    ]
                }
                for t in dv.turns
            ]
        })

    def printer(idx: int) -> None:
        if idx < 0 or idx >= len(dialogues):
            raise IndexError(f"Index out of range: {idx}, total={len(dialogues)}")
        txt = _pretty_print_dialogue(dialogues[idx], max_content_len=max_content_len)
        print(txt)

    return structured_as_dicts, printer



def print_messages(non_tensor_batch, max_content_len = 5000, idx = None, extra_info = True):
    batch_dialogs_raw = non_tensor_batch["messages"]
    structured, printer = process_batch(non_tensor_batch, max_content_len=max_content_len)
    if idx is not None:
        printer(idx)
    else:
        for i in range(len(structured)):
            print(batch_dialogs_raw[i])
            print("=" * 80)
            print(f"Dialogue {i}")
            print("=" * 80)
            printer(i)
            if extra_info:
                print(non_tensor_batch['extra_info'][i])

# ---------------------
# 命令行用法 (可选)
# ---------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pretty print dialogue batch JSON.")
    parser.add_argument("json_file", type=str, help="Path to JSON file containing non_tensor_batch-like object.")
    parser.add_argument("--max-len", type=int, default=800, help="Max content length per turn.")
    parser.add_argument("--idx", type=int, default=None,
                        help="Index of dialogue to print. If omitted, print all.")
    args = parser.parse_args()

    with open(args.json_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    structured, printer = process_batch(obj, max_content_len=args.max_len)

    