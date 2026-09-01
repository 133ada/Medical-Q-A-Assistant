"""Canonical application configuration.

The legacy ``App.config`` package re-exports this module for backwards
compatibility. New code should import settings from ``App.Core.config``.
"""
# =============================================================================
# Memory 模块配置（CoverLayWindow.md 第三十三节原则）
#   · Redis   = 短期对话记忆
#   · 数据库   = 长期 Summary + 长期 Memory
# =============================================================================
import os
from pathlib import Path
from dotenv import load_dotenv


# Load the project-level .env before reading configuration values. Older local
# setups kept the file under App/, so retain that location as a fallback.
# Explicit process environment variables still take precedence.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for env_file in (PROJECT_ROOT / ".env", PROJECT_ROOT / "App" / ".env"):
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}

CONFIG_DEEPSEEK = {
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "max_iterations": 5,
}

CONFIG_ALIYUN = {
    "api_key": os.environ.get("ALIYUN_API_KEY", ""),
    "base_url": os.environ.get(
        "ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ),
    "model": os.environ.get("ALIYUN_EMBEDDING_MODEL", "text-embedding-v3"),
    "max_iterations": 5,
    "dimensions": 1024
}

CONFIG_MEMORY = {
    "short_term": {
        # 短期记忆最多保存的消息条数（是"条"，不是"轮"）
        "max_messages": 20,
        # 短期记忆 TTL：1 天（秒），每次写入刷新（滑动过期）
        "ttl_seconds": 86400,
        # 是否使用真实 Redis；Redis 不可用时是否降级为内存存储
        "use_redis": True,
        "fallback_to_memory": True,
    },
    "summary": {
        # 历史上下文累计 Token 数达到该阈值即触发 Summary
        "token_threshold": 10000,
        # Summary 触发后 Redis 保留的最近消息条数
        "keep_recent_messages": 20,
        # tiktoken 编码器（近似值；DeepSeek 无公开官方 tokenizer）
        "encoder_name": "cl100k_base",
    },
    "long_term": {
        # 长期记忆相关性检索 Top-K
        "top_k": 5,
        # 用药记忆检索 Top-K
        "medication_top_k": 3,
        # 进入 Prompt 的长期记忆条数上限
        "max_in_context": 5,
    },
}

# 短期记忆 Redis 地址（真实 Redis 服务器必填）
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")

# 长期记忆数据库连接串
#   PostgreSQL：postgresql+psycopg://user:password@host:port/dbname
#   本地兜底：  sqlite:///./Memory.db
# 通过环境变量 DATABASE_URL 覆盖（生产环境强烈建议 PostgreSQL）。
DATABASE_URL = os.environ.get(
    "DATABASE_URL", ""
)

# AUTH_TOKENS_JSON maps bearer tokens to server-side user ids.
# Anonymous requests remain allowed by default, but do not use Memory.
AUTH_REQUIRED = _env_bool("AUTH_REQUIRED", False)
AUTH_TOKENS_JSON = os.environ.get("AUTH_TOKENS_JSON", "{}")
JWT_SECRET = os.environ.get("JWT_SECRET", "")
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FRONTEND_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
    ).split(",")
    if origin.strip()
]


def missing_runtime_secrets() -> list[str]:
    """Return required Core secrets that are not configured."""
    missing: list[str] = []
    if not CONFIG_DEEPSEEK["api_key"]:
        missing.append("DEEPSEEK_API_KEY")
    if not CONFIG_ALIYUN["api_key"]:
        missing.append("ALIYUN_API_KEY")
    if not JWT_SECRET:
        missing.append("JWT_SECRET")
    return missing

# ── 短期记忆 ──────────────────────────────────────────────────────────────────
SHORT_TERM_MAX_MESSAGES = CONFIG_MEMORY["short_term"]["max_messages"]
SHORT_TERM_TTL_SECONDS = CONFIG_MEMORY["short_term"]["ttl_seconds"]
SHORT_TERM_USE_REDIS = CONFIG_MEMORY["short_term"]["use_redis"]
SHORT_TERM_FALLBACK_TO_MEMORY = CONFIG_MEMORY["short_term"]["fallback_to_memory"]

# ── Summary ───────────────────────────────────────────────────────────────────
SUMMARY_TOKEN_THRESHOLD = CONFIG_MEMORY["summary"]["token_threshold"]
SUMMARY_KEEP_RECENT_MESSAGES = CONFIG_MEMORY["summary"]["keep_recent_messages"]
TOKEN_ENCODER_NAME = CONFIG_MEMORY["summary"]["encoder_name"]

# ── 长期记忆 ──────────────────────────────────────────────────────────────────
LONG_TERM_TOP_K = CONFIG_MEMORY["long_term"]["top_k"]
LONG_TERM_MEDICATION_TOP_K = CONFIG_MEMORY["long_term"]["medication_top_k"]
LONG_TERM_MAX_IN_CONTEXT = CONFIG_MEMORY["long_term"]["max_in_context"]

