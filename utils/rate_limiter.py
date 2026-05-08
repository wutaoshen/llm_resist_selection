"""速率限制器模块：令牌桶算法实现API请求频率控制"""

import time
import threading


class TokenBucketRateLimiter:
    """令牌桶速率限制器，控制API请求频率"""

    def __init__(self, rpm=600, burst_multiplier=0.8):
        """
        初始化速率限制器

        Args:
            rpm: 每分钟最大请求数
            burst_multiplier: 安全系数，默认0.8表示只使用80%的配额
        """
        self.rpm = rpm
        self.effective_rpm = int(rpm * burst_multiplier)
        self.tokens = self.effective_rpm
        self.max_tokens = self.effective_rpm
        self.refill_rate = self.effective_rpm / 60.0
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        print(f"    [RateLimiter] 初始化: RPM={rpm}, 有效RPM={self.effective_rpm}, "
              f"每秒补充={self.refill_rate:.2f}令牌")

    def _refill(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill_time = now

    def acquire(self, timeout=120):
        """
        获取一个令牌（阻塞直到获取成功或超时）

        Args:
            timeout: 最大等待时间（秒）

        Returns:
            bool: 是否成功获取令牌
        """
        start_time = time.time()
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                wait_time = (1 - self.tokens) / self.refill_rate
            if time.time() - start_time + wait_time > timeout:
                print(f"    [RateLimiter] 获取令牌超时 (等待>{timeout}s)")
                return False
            actual_wait = min(wait_time, 1.0)
            time.sleep(actual_wait)
        return False

    def get_status(self):
        """获取当前状态"""
        with self.lock:
            self._refill()
            return {
                'available_tokens': self.tokens,
                'max_tokens': self.max_tokens,
                'effective_rpm': self.effective_rpm
            }


# 全局速率限制器实例（按提供商分别管理）
_rate_limiters = {}
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(provider="dashscope", rpm=600):
    """
    获取指定提供商的速率限制器（单例模式）

    Args:
        provider: 提供商名称
        rpm: 每分钟请求限制
    """
    global _rate_limiters
    with _rate_limiter_lock:
        if provider not in _rate_limiters:
            _rate_limiters[provider] = TokenBucketRateLimiter(rpm=rpm)
        return _rate_limiters[provider]
