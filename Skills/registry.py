"""
Skills/registry.py
==================
纯基础设施层 —— 零业务依赖，零 LangChain 导入。

提供：
  SkillMeta      — Skill 元数据 dataclass（含运行时统计）
  @skill         — 节点装饰器（计时 / 日志 / 异常包装）
  SkillRegistry  — 全局注册表（支持按名查找、批量打印）
  print_skill_registry / print_skill_stats — 调试工具
"""

from __future__ import annotations

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# ── 终端颜色 ──────────────────────────────────────────────────────────────────
_USE_COLOR = os.environ.get("LOG_NO_COLOR", "0") != "1"

_C: dict[str, str] = {
    k: (v if _USE_COLOR else "")
    for k, v in {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "purple": "\033[35m",
        "teal": "\033[36m",
        "blue": "\033[34m",
        "amber": "\033[33m",
        "gray": "\033[90m",
        "red": "\033[31m",
        "green": "\033[32m",
    }.items()
}

# skill name → ANSI color（由 medical_skills.py 中每个 SkillMeta 的 color_key 驱动）
_ANSI_MAP: dict[str, str] = {
    "purple": _C["purple"],
    "teal": _C["teal"],
    "blue": _C["blue"],
    "amber": _C["amber"],
    "gray": _C["gray"],
}


# =============================================================================
# SkillMeta
# =============================================================================

@dataclass
class SkillMeta:
    """
    描述一个 Agent Skill 节点的元数据。

    Parameters
    ----------
    name        : 节点唯一标识符，与 LangGraph add_node 的 key 保持一致。
    icon        : Unicode 图标，用于终端日志可视化。
    description : 单行功能描述（≤ 40 字）。
    version     : 语义化版本号，默认 "1.0.0"。
    color_key   : 对应 _ANSI_MAP 的颜色键，控制终端输出颜色。
    tags        : 可选标签列表，用于分类和过滤。
    prompts     : 该 Skill 使用的所有 Prompt 模板字典，键为业务语义名称。
                  类型标注为 dict[str, Any] 以避免在基础层引入 LangChain 依赖；
                  实际值由 medical_skills.py 注入 ChatPromptTemplate 实例。
    """
    name: str
    icon: str
    description: str
    version: str = "1.0.0"
    color_key: str = "gray"
    tags: list[str] = field(default_factory=list)
    prompts: dict[str, Any] = field(default_factory=dict)

    # 运行时统计（由 @skill 装饰器自动填充，不参与构造）
    call_count: int = field(default=0, init=False, repr=False)
    total_ms: float = field(default=0.0, init=False, repr=False)

    # ── 统计属性 ──────────────────────────────────────────────────────────────

    @property
    def avg_ms(self) -> Optional[float]:
        """平均单次耗时（ms），无调用记录时返回 None。"""
        return self.total_ms / self.call_count if self.call_count else None

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _color(self) -> str:
        return _ANSI_MAP.get(self.color_key, _C["gray"])

    # ── 日志方法 ──────────────────────────────────────────────────────────────

    def log_start(self) -> None:
        color = self._color()
        print(
            f"\n{color}{_C['bold']}[{self.icon}  Skill:{self.name}]{_C['reset']} "
            f"{_C['dim']}v{self.version} · {self.description}{_C['reset']}"
        )

    def log_step(self, msg: str, *, indent: int = 4) -> None:
        color = self._color()
        pad = " " * indent
        print(f"{color}{pad}▸{_C['reset']} {_C['dim']}{msg}{_C['reset']}")

    def log_end(self, elapsed_ms: float, ok: bool = True) -> None:
        color = self._color()
        status = f"{_C['green']}✔ done" if ok else f"{_C['red']}✘ failed"
        bar = _ms_bar(elapsed_ms)
        print(
            f"{color}{'└':>21}{_C['reset']} "
            f"{status}{_C['reset']}  "
            f"{_C['dim']}{bar}  {elapsed_ms:.0f} ms{_C['reset']}"
        )

    def get_prompt(self, key: str) -> Any:
        """
        按语义键取 Prompt 模板，键不存在时抛出清晰的 KeyError。

        Usage::
            prompt = SKILLS["retriever"].get_prompt("enhance")
        """
        if key not in self.prompts:
            available = list(self.prompts.keys())
            raise KeyError(
                f"[Skill:{self.name}] prompt key '{key}' 不存在，"
                f"可用键：{available}"
            )
        return self.prompts[key]


# =============================================================================
# @skill 装饰器
# =============================================================================

def skill(meta: SkillMeta) -> Callable:
    """
    节点装饰器：绑定 SkillMeta，自动记录耗时、调用次数与结构化日志。

    用法::

        @skill(SKILLS["my_node"])
        def my_node(state: MedicalState) -> dict:
            ...

    装饰后的函数签名与原函数完全一致，可直接传给 builder.add_node()。
    装饰器将 meta 挂载为 wrapper.skill_meta，方便外部检查。
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            meta.log_start()
            t0 = time.perf_counter()
            try:
                result = fn(state, *args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                meta.call_count += 1
                meta.total_ms += elapsed
                meta.log_end(elapsed, ok=True)
                return result
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                meta.log_end(elapsed, ok=False)
                raise RuntimeError(
                    f"[Skill:{meta.name}] 执行失败，耗时 {elapsed:.0f} ms"
                ) from exc

        wrapper.skill_meta = meta  # type: ignore[attr-defined]
        return wrapper

    return decorator


# =============================================================================
# SkillRegistry
# =============================================================================

class SkillRegistry:
    """
    全局 Skill 注册表。

    通常在 medical_skills.py 中实例化并导出一个单例 REGISTRY，
    graph.py 只需 ``from Skills.medical_skills import REGISTRY`` 即可。
    """

    def __init__(self) -> None:
        self._store: dict[str, SkillMeta] = {}

    def register(self, meta: SkillMeta) -> SkillMeta:
        """注册一个 SkillMeta，返回该实例（方便链式调用）。"""
        self._store[meta.name] = meta
        return meta

    def __getitem__(self, name: str) -> SkillMeta:
        if name not in self._store:
            raise KeyError(f"Skill '{name}' 未注册，已注册：{list(self._store)}")
        return self._store[name]

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def all(self) -> dict[str, SkillMeta]:
        return dict(self._store)

    # ── 调试工具 ──────────────────────────────────────────────────────────────

    def print_registry(self) -> None:
        """打印所有已注册 Skill 的摘要表格。"""
        sep = "─" * 68
        header = f"{'Skill':<20} {'Icon':<5} {'Ver':<9} {'Tags'}"
        print(f"\n{_C['bold']}Agent Skill Registry{_C['reset']}")
        print(sep)
        print(f"{_C['dim']}{header}{_C['reset']}")
        print(sep)
        for name, m in self._store.items():
            color = _ANSI_MAP.get(m.color_key, _C["gray"])
            tags = ", ".join(m.tags) if m.tags else "—"
            prompts_hint = f"  [{', '.join(m.prompts)}]" if m.prompts else ""
            print(
                f"{color}{_C['bold']}{m.name:<20}{_C['reset']}"
                f"{m.icon:<5} "
                f"{_C['dim']}v{m.version:<8} {tags}{prompts_hint}{_C['reset']}"
            )
        print(sep)

    def print_stats(self) -> None:
        """打印各 Skill 的运行时统计（调用次数、平均耗时）。"""
        sep = "─" * 52
        print(f"\n{_C['bold']}Skill Runtime Stats{_C['reset']}")
        print(sep)
        for name, m in self._store.items():
            color = _ANSI_MAP.get(m.color_key, _C["gray"])
            avg_str = f"{m.avg_ms:.0f} ms" if m.avg_ms is not None else "—"
            print(
                f"{color}{m.icon} {m.name:<20}{_C['reset']}"
                f"{_C['dim']}calls={m.call_count:<5} avg={avg_str}{_C['reset']}"
            )
        print(sep)


# =============================================================================
# 内部工具
# =============================================================================

def _ms_bar(ms: float, width: int = 10) -> str:
    """将耗时映射为 ASCII 进度条（0–3000 ms 区间）。"""
    filled = min(int(ms / 3000 * width), width)
    return "▓" * filled + "░" * (width - filled)
