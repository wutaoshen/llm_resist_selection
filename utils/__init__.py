"""utils 工具包：提供API调用、速率限制、重试机制、日志等通用功能"""

from .rate_limiter import TokenBucketRateLimiter, get_rate_limiter
from .retry import RetryConfig, DEFAULT_RETRY_CONFIG, is_rate_limit_error, is_retryable_error
from .api_client import (
    call_api, call_dashscope_api, call_openai_api, parse_json_response,
    KIMI_BASE_URL, MULTIMODAL_MODELS, OPENAI_MODELS, OPENAI_API_KEY, OPENAI_BASE_URL
)
from .logging import Tee

__all__ = [
    # 速率限制
    'TokenBucketRateLimiter', 'get_rate_limiter',
    # 重试与错误分类
    'RetryConfig', 'DEFAULT_RETRY_CONFIG', 'is_rate_limit_error', 'is_retryable_error',
    # API 调用
    'call_api', 'call_dashscope_api', 'call_openai_api', 'parse_json_response',
    'KIMI_BASE_URL', 'MULTIMODAL_MODELS', 'OPENAI_MODELS', 'OPENAI_API_KEY', 'OPENAI_BASE_URL',
    # 日志
    'Tee',
]
