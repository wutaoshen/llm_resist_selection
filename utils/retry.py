"""重试机制模块：指数退避重试配置与错误分类"""

import random


class RetryConfig:
    """重试配置：支持指数退避和随机抖动"""

    def __init__(self, max_retries=5, base_delay=2.0, max_delay=60.0,
                 exponential_base=2.0, jitter=True):
        """
        初始化重试配置

        Args:
            max_retries: 最大重试次数
            base_delay: 基础延迟时间（秒）
            max_delay: 最大延迟时间（秒）
            exponential_base: 指数退避的基数
            jitter: 是否添加随机抖动
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt):
        """
        计算第N次重试的延迟时间

        Args:
            attempt: 当前尝试次数（从0开始）
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0.1, delay)


# 默认重试配置
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=5, base_delay=2.0, max_delay=60.0,
    exponential_base=2.0, jitter=True
)


# ==================== 错误分类 ====================

def is_rate_limit_error(error, status_code=None):
    """
    判断是否为速率限制错误

    Args:
        error: 异常对象或错误消息
        status_code: HTTP状态码
    """
    if status_code in [400, 429, 503]:
        return True
    error_str = str(error).lower()
    rate_limit_keywords = [
        'rate limit', 'rate_limit', 'ratelimit', 'too many requests',
        'quota exceeded', 'throttl', 'rpm', 'requests per minute',
        'concurrency', 'overloaded', 'capacity', 'busy',
        'try again later', '请求过于频繁', '超出限制', '限流', '频率限制'
    ]
    return any(keyword in error_str for keyword in rate_limit_keywords)


def is_retryable_error(error, status_code=None):
    """
    判断错误是否可重试

    Args:
        error: 异常对象或错误消息
        status_code: HTTP状态码
    """
    if is_rate_limit_error(error, status_code):
        return True
    if status_code and status_code >= 500:
        return True
    error_str = str(error).lower()
    retryable_keywords = [
        'timeout', 'connection', 'network', 'temporary',
        'unavailable', 'reset', 'broken pipe', 'eof', '超时', '连接'
    ]
    return any(keyword in error_str for keyword in retryable_keywords)
