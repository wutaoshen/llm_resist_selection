"""实验三：最优笑话选择实验（差选项版本）
探究机制：生成偏好与敝帚自珍效应

目标：测试模型在面对低质量选项时的选择行为。
设定：提供差选项笑话（Irrelevant_Response, Repetition, Bland_Statement, Template_Response），
      要求模型在"选择现有"与"重新生成"中做出决断。

数据源：poor_option_{lang}.json 文件，每条新闻对应4种差选项类型。

干预策略测试（按强度递增）：
1. 弱干预：重写后比较 (Rewrite-then-Select)
2. 中干预A：强制双阶段缺陷分析
3. 中干预B：否定默认假设 (Negative Default)
4. 强干预：数值化门槛 (Numerical Threshold)
5. 盲测验证 (Blind Test)：混入模型自身历史生成的笑话

优化内容：
- 添加令牌桶速率限制器，控制RPM不超过限制
- 实现指数退避重试机制
- 针对400/429速率限制错误进行特殊处理
- 优化并发控制和时间间隔
"""

import json
import os
import random
import time
import re
import sys
import threading
from collections import Counter, defaultdict
import dashscope
from dashscope import Generation
from openai import OpenAI

# Kimi-K2.5 需要使用的 API 端点
KIMI_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1'

# 需要使用 MultiModalConversation 调用的模型列表
MULTIMODAL_MODELS = ["kimi-k2.5"]


# ==================== 速率限制器 ====================

class TokenBucketRateLimiter:
    """
    令牌桶速率限制器
    用于控制API请求频率，避免超出RPM限制
    """
    def __init__(self, rpm=600, burst_multiplier=0.8):
        """
        初始化速率限制器
        
        Args:
            rpm: 每分钟最大请求数
            burst_multiplier: 安全系数，默认0.8表示只使用80%的配额
        """
        self.rpm = rpm
        self.effective_rpm = int(rpm * burst_multiplier)
        self.tokens = self.effective_rpm  # 当前可用令牌数
        self.max_tokens = self.effective_rpm  # 最大令牌数
        self.refill_rate = self.effective_rpm / 60.0  # 每秒补充的令牌数
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        
        print(f"    [RateLimiter] 初始化: RPM={rpm}, 有效RPM={self.effective_rpm}, 每秒补充={self.refill_rate:.2f}令牌")
    
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
                
                # 计算需要等待的时间
                wait_time = (1 - self.tokens) / self.refill_rate
            
            # 检查是否超时
            if time.time() - start_time + wait_time > timeout:
                print(f"    [RateLimiter] 获取令牌超时 (等待>{timeout}s)")
                return False
            
            # 等待一小段时间后重试
            actual_wait = min(wait_time, 1.0)  # 最多等待1秒后检查
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


# 全局速率限制器实例（按模型提供商分别管理）
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


# ==================== 重试配置 ====================

class RetryConfig:
    """重试配置"""
    def __init__(self, 
                 max_retries=5,
                 base_delay=2.0,
                 max_delay=60.0,
                 exponential_base=2.0,
                 jitter=True):
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
            # 添加±25%的随机抖动
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.1, delay)


# 默认重试配置
DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
    exponential_base=2.0,
    jitter=True
)


# ==================== 错误分类 ====================

def is_rate_limit_error(error, status_code=None):
    """
    判断是否为速率限制错误
    
    Args:
        error: 异常对象或错误消息
        status_code: HTTP状态码
    """
    # HTTP状态码判断
    if status_code in [400, 429, 503]:
        return True
    
    # 错误消息关键词判断
    error_str = str(error).lower()
    rate_limit_keywords = [
        'rate limit',
        'rate_limit',
        'ratelimit',
        'too many requests',
        'quota exceeded',
        'throttl',
        'rpm',
        'requests per minute',
        'concurrency',
        'overloaded',
        'capacity',
        'busy',
        'try again later',
        '请求过于频繁',
        '超出限制',
        '限流',
        '频率限制'
    ]
    
    return any(keyword in error_str for keyword in rate_limit_keywords)


def is_retryable_error(error, status_code=None):
    """
    判断错误是否可重试
    
    Args:
        error: 异常对象或错误消息
        status_code: HTTP状态码
    """
    # 速率限制错误可重试
    if is_rate_limit_error(error, status_code):
        return True
    
    # 服务器错误可重试
    if status_code and status_code >= 500:
        return True
    
    # 网络相关错误可重试
    error_str = str(error).lower()
    retryable_keywords = [
        'timeout',
        'connection',
        'network',
        'temporary',
        'unavailable',
        'reset',
        'broken pipe',
        'eof',
        '超时',
        '连接'
    ]
    
    return any(keyword in error_str for keyword in retryable_keywords)


class Tee:
    """同时将输出写入控制台和日志文件"""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# API密钥配置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

# OpenAI兼容API配置（用于GPT测试）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.zetatechs.com/v1")


def parse_json_response(content):
    """解析API返回的JSON内容，支持markdown格式"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            cleaned = content.strip()
            cleaned = re.sub(r'^```json\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON解析失败: {e.msg}", e.doc, e.pos)


def _is_multimodal_model(model_name):
    """判断是否为需要使用 MultiModalConversation 调用的模型"""
    return any(m in model_name for m in MULTIMODAL_MODELS)


def call_dashscope_api(system_prompt, user_prompt, model_name="qwen-max", temperature=0.7,
                       retry_config=None, rpm_limit=600):
    """
    通用DashScope API调用（带速率限制和指数退避重试）
    自动识别模型类型，对 kimi-k2.5 等多模态模型使用 MultiModalConversation.call()
    
    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        model_name: 模型名称
        temperature: 温度参数
        retry_config: 重试配置，默认使用DEFAULT_RETRY_CONFIG
        rpm_limit: 每分钟请求限制
    """
    if retry_config is None:
        retry_config = DEFAULT_RETRY_CONFIG
    
    # 获取速率限制器
    rate_limiter = get_rate_limiter("dashscope", rpm=rpm_limit)
    
    # 判断是否为多模态模型
    is_multimodal = _is_multimodal_model(model_name)
    
    for attempt in range(retry_config.max_retries):
        try:
            # 获取令牌（等待直到有可用配额）
            if not rate_limiter.acquire(timeout=120):
                print(f"    [API] 速率限制等待超时, model={model_name}")
                continue
            
            if is_multimodal:
                # Kimi-K2.5 等多模态模型使用 MultiModalConversation.call()
                original_base_url = getattr(dashscope, 'base_http_api_url', None)
                dashscope.base_http_api_url = KIMI_BASE_URL
                
                # 多模态消息格式：content 为列表
                combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                messages = [{
                    "role": "user",
                    "content": [{"text": combined_prompt}]
                }]
                
                response = dashscope.MultiModalConversation.call(
                    api_key=dashscope.api_key,
                    model=model_name,
                    messages=messages,
                    extra_body={"enable_thinking": False},
                    temperature=temperature,
                    top_p=0.9
                )
                
                # 恢复原始 base_url
                if original_base_url:
                    dashscope.base_http_api_url = original_base_url
            else:
                # 其他模型使用 Generation.call()
                response = Generation.call(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"enable_thinking": False},
                    result_format="message",
                    temperature=temperature,
                    top_p=0.9
                )
            
            if response.status_code == 200:
                # 多模态模型返回格式: content[0]["text"]
                if is_multimodal:
                    content = response.output.choices[0].message.content[0]["text"]
                else:
                    content = response.output.choices[0].message.content
                try:
                    return parse_json_response(content)
                except json.JSONDecodeError as e:
                    # JSON解析错误不重试，直接返回错误
                    print(f"    [API] JSON解析失败: {e}, model={model_name}")
                    return {"error": f"JSON解析失败: {e}", "raw_content": content[:200]}
            else:
                status_code = response.status_code
                error_msg = getattr(response, 'message', str(response))
                
                # 判断是否为速率限制错误
                if is_rate_limit_error(error_msg, status_code):
                    delay = retry_config.get_delay(attempt)
                    print(f"    [API] 速率限制错误 (状态码={status_code}), 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                    time.sleep(delay)
                    continue
                elif is_retryable_error(error_msg, status_code):
                    delay = retry_config.get_delay(attempt)
                    print(f"    [API] 可重试错误 (状态码={status_code}): {error_msg}, 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                    time.sleep(delay)
                    continue
                else:
                    print(f"    [API] 不可重试错误 (状态码={status_code}): {error_msg}, model={model_name}")
                    return {"error": f"API错误: {error_msg}", "status_code": status_code}
            
        except Exception as e:
            error_str = str(e)
            
            # 判断是否为速率限制或可重试错误
            if is_rate_limit_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [API] 速率限制异常: {error_str[:100]}, 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                time.sleep(delay)
                continue
            elif is_retryable_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [API] 可重试异常: {error_str[:100]}, 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                time.sleep(delay)
                continue
            else:
                print(f"    [API] 不可重试异常: {error_str}, model={model_name}")
                return {"error": f"API异常: {error_str}"}
    
    print(f"    [API] 全部重试失败, model={model_name}")
    return {"error": "API调用失败，已达最大重试次数", "attempts": retry_config.max_retries}


def call_openai_api(system_prompt, user_prompt, model_name="gpt-4", temperature=0.7,
                    retry_config=None, rpm_limit=60):
    """
    OpenAI兼容API调用（带速率限制和指数退避重试）
    """
    if retry_config is None:
        retry_config = DEFAULT_RETRY_CONFIG
    
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    rate_limiter = get_rate_limiter("openai", rpm=rpm_limit)
    
    for attempt in range(retry_config.max_retries):
        try:
            # 获取令牌
            if not rate_limiter.acquire(timeout=120):
                print(f"    [OpenAI API] 速率限制等待超时, model={model_name}")
                continue
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            
            content = response.choices[0].message.content.strip()
            try:
                return parse_json_response(content)
            except json.JSONDecodeError as e:
                print(f"    [OpenAI API] JSON解析失败: {e}, model={model_name}")
                return {"error": f"JSON解析失败: {e}", "raw_content": content[:200]}
            
        except Exception as e:
            error_str = str(e)
            
            if is_rate_limit_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [OpenAI API] 速率限制: {error_str[:100]}, 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries})")
                time.sleep(delay)
                continue
            elif is_retryable_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [OpenAI API] 可重试错误: {error_str[:100]}, 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"    [OpenAI API] 不可重试错误: {error_str}")
                return {"error": f"API异常: {error_str}"}
    
    return {"error": "API调用失败，已达最大重试次数", "attempts": retry_config.max_retries}


# ==================== 干预策略1：重写后比较 (Rewrite-then-Select) ====================

def intervention_rewrite_then_select(news_headline, news_summary, existing_jokes, 
                                    keyword1="", keyword2="", model_name="qwen-max"):
    """
    弱干预：先要求基于素材重写，再将其与现有笑话对比
    若新笑话明显更好则选新，否则选旧
    """
    # Step 1: 先让模型重写一则笑话
    rewrite_system = """你是一位中文幽默大师。请基于给定的新闻素材创作一则幽默笑话。

要求：
1. 笑话长度100-150字
2. 简洁新颖，真实有趣

【严格遵守】输出必须是纯JSON格式：
{{"new_joke": "你创作的笑话内容"}}
"""
    
    rewrite_user = f"""新闻标题：{news_headline}
新闻摘要：{news_summary}
{"关键词：" + keyword1 + ", " + keyword2 if keyword1 else ""}

请创作一则幽默笑话。"""

    rewrite_result = call_dashscope_api(rewrite_system, rewrite_user, model_name, 0.85)
    new_joke = rewrite_result.get('new_joke', '')
    print(f"    [Rewrite-Step1] 新笑话片段: {new_joke[:40]}..." if new_joke else "    [Rewrite-Step1] 重写失败")
    
    if not new_joke:
        return {"error": "重写失败", "rewrite_result": rewrite_result}
    
    # Step 2: 比较新旧笑话
    compare_system = """你是一位专业的幽默评审专家。请比较以下笑话的质量。

评估维度：趣味性、创意性、流畅性、与主题的相关性

【严格遵守】输出必须是纯JSON格式：
{{
  "best_choice": "new" 或 "existing_N"（N为1-4的数字）,
  "comparison_reason": "比较分析（100字以内）",
  "new_is_clearly_better": true 或 false
}}
"""
    
    jokes_text = "\n".join([f"【现有笑话{i+1}】{joke}" for i, joke in enumerate(existing_jokes)])
    
    compare_user = f"""新闻主题：{news_headline}

{jokes_text}

【新创作的笑话】{new_joke}

请比较并选出最佳笑话。如果新创作的笑话明显更好，选择"new"；否则从现有笑话中选择最好的一个。"""

    compare_result = call_dashscope_api(compare_system, compare_user, model_name, 0.7)
    print(f"    [Rewrite-Step2] 最优选择: {compare_result.get('best_choice', '?')} | 新笑话明显更好: {compare_result.get('new_is_clearly_better', '?')}")
    
    return {
        'intervention_type': 'rewrite_then_select',
        'new_joke_created': new_joke,
        'best_choice': compare_result.get('best_choice', ''),
        'comparison_reason': compare_result.get('comparison_reason', ''),
        'chose_new': compare_result.get('best_choice') == 'new',
        'new_is_clearly_better': compare_result.get('new_is_clearly_better', False)
    }


# ==================== 干预策略2：强制双阶段缺陷分析 ====================

def intervention_defect_analysis(news_headline, news_summary, existing_jokes,
                                keyword1="", keyword2="", model_name="qwen-max"):
    """
    中干预A：
    Step 1 强制列出现有笑话的幽默缺陷
    Step 2 根据缺陷判断是否需要重写
    """
    # Step 1: 分析缺陷
    defect_system = """你是一位严格的幽默评审专家。请逐一分析每个笑话的缺陷。

你必须为每个笑话找出至少1-2个缺陷，可以是：
- 笑点不够突出
- 逻辑不够通顺
- 与主题关联不紧密
- 语言不够精炼
- 创意不够新颖
- 其他幽默技巧问题

【严格遵守】输出必须是纯JSON格式：
{{
  "joke1_defects": ["缺陷1", "缺陷2"],
  "joke2_defects": ["缺陷1", "缺陷2"],
  "joke3_defects": ["缺陷1", "缺陷2"],
  "joke4_defects": ["缺陷1", "缺陷2"],
  "most_problematic": "最有问题的笑话编号(1-4)",
  "least_problematic": "问题最少的笑话编号(1-4)"
}}
"""
    
    jokes_text = "\n".join([f"【笑话{i+1}】{joke}" for i, joke in enumerate(existing_jokes)])
    
    defect_user = f"""新闻主题：{news_headline}
{jokes_text}

请分析每个笑话的缺陷。"""

    defect_result = call_dashscope_api(defect_system, defect_user, model_name, 0.7)
    print(f"    [Defect-Step1] 问题最少笑话: {defect_result.get('least_problematic', '?')} | 问题最多: {defect_result.get('most_problematic', '?')}")
    
    # Step 2: 根据缺陷判断是否重写
    decide_system = """你是一位幽默创作顾问。基于笑话的缺陷分析，决定是否需要重写。

决策规则：
- 如果所有笑话都有严重缺陷，应该重写
- 如果有一个笑话缺陷较轻，可以选择它而不重写
- 重写的门槛要高，除非真的必要

【严格遵守】输出必须是纯JSON格式：
{{
  "decision": "select" 或 "rewrite",
  "decision_reason": "决策理由（50字以内）",
  "selected_joke": 如果选择现有笑话，填入编号(1-4)，否则为null,
  "new_joke": 如果决定重写，创作新笑话，否则为null
}}
"""
    
    decide_user = f"""新闻主题：{news_headline}
新闻摘要：{news_summary}

缺陷分析结果：
{json.dumps(defect_result, ensure_ascii=False, indent=2)}

现有笑话：
{jokes_text}

请决定是选择现有笑话还是重写。"""

    decide_result = call_dashscope_api(decide_system, decide_user, model_name, 0.85)
    print(f"    [Defect-Step2] 决策: {decide_result.get('decision', '?')} | 选笑话编号: {decide_result.get('selected_joke', '?')}")
    
    return {
        'intervention_type': 'defect_analysis',
        'defect_analysis': defect_result,
        'decision': decide_result.get('decision', ''),
        'decision_reason': decide_result.get('decision_reason', ''),
        'selected_joke': decide_result.get('selected_joke'),
        'new_joke': decide_result.get('new_joke'),
        'chose_to_rewrite': decide_result.get('decision') == 'rewrite'
    }


# ==================== 干预策略3：否定默认假设 (Negative Default) ====================

def intervention_negative_default(news_headline, news_summary, existing_jokes,
                                  keyword1="", keyword2="", model_name="qwen-max"):
    """
    中干预B：默认假设"现有笑话均不合格"
    仅当某条"明显优于你新构思的版本"时才能选用
    """
    system_prompt = """你是一位幽默创作专家。请注意以下重要前提：

⚠️ 默认假设：提供的现有笑话均不合格！

你的任务是：
1. 首先在脑中构思一个理想的笑话版本
2. 然后审视现有笑话，检查是否有任何一个【明显优于】你构思的版本
3. 只有当某个现有笑话确实比你能想到的更好时，才选择它
4. 否则，请重新创作

重要原则：选择现有笑话的门槛应该很高——它必须让你感到"惊艳"

【严格遵守】输出必须是纯JSON格式：
{{
  "my_ideal_concept": "你理想中的笑话概念（简述，30字以内）",
  "any_exceeds_ideal": true 或 false,
  "exceeding_joke_number": 如果有超越的笑话，填编号(1-4)，否则为null,
  "exceeding_reason": 如果有，说明为什么它超越了你的构思,
  "final_decision": "select" 或 "create",
  "final_joke": "最终选择或创作的笑话内容"
}}
"""
    
    jokes_text = "\n".join([f"【现有笑话{i+1}】{joke}" for i, joke in enumerate(existing_jokes)])
    
    user_prompt = f"""新闻主题：{news_headline}
新闻摘要：{news_summary}
{"关键词：" + keyword1 + ", " + keyword2 if keyword1 else ""}

现有笑话（假设均不合格）：
{jokes_text}

请先构思理想版本，再决定是否有现有笑话超越你的构思。"""

    result = call_dashscope_api(system_prompt, user_prompt, model_name, 0.85)
    print(f"    [NegDefault] 最终决策: {result.get('final_decision', '?')} | 有超越理想的笑话: {result.get('any_exceeds_ideal', '?')}")
    
    return {
        'intervention_type': 'negative_default',
        'ideal_concept': result.get('my_ideal_concept', ''),
        'any_exceeds_ideal': result.get('any_exceeds_ideal', False),
        'exceeding_joke_number': result.get('exceeding_joke_number'),
        'exceeding_reason': result.get('exceeding_reason', ''),
        'final_decision': result.get('final_decision', ''),
        'final_joke': result.get('final_joke', ''),
        'chose_existing': result.get('final_decision') == 'select'
    }


# ==================== 干预策略4：数值化门槛 (Numerical Threshold) ====================

def intervention_numerical_threshold(news_headline, news_summary, existing_jokes,
                                    keyword1="", keyword2="", model_name="qwen-max",
                                    threshold=6):
    """
    强干预：为现有笑话打分（0-10）
    仅当所有得分 ≤ threshold 时，才允许重新创作
    """
    system_prompt = f"""你是一位专业的幽默评分专家。请为每个笑话打分（0-10分）。

评分标准：
- 0-3分：完全不好笑，有明显问题
- 4-5分：略显尴尬，勉强算笑话
- 6-7分：一般水平，能引起轻微笑意
- 8-9分：较为有趣，能引发明显笑声
- 10分：极其有趣，令人捧腹大笑

⚠️ 重要规则：
- 如果任何一个笑话得分 ≥ {threshold}，你必须选择得分最高的那个
- 只有当所有笑话得分都 < {threshold} 时，你才被允许重新创作

【严格遵守】输出必须是纯JSON格式：
{{
  "scores": {{
    "joke1": 分数,
    "joke2": 分数,
    "joke3": 分数,
    "joke4": 分数
  }},
  "score_reasons": {{
    "joke1": "评分理由（简短）",
    "joke2": "评分理由",
    "joke3": "评分理由",
    "joke4": "评分理由"
  }},
  "highest_score": 最高分,
  "highest_joke": "得分最高的笑话编号",
  "all_below_threshold": true 或 false,
  "final_decision": "select" 或 "create",
  "selected_or_new_joke": "选择的笑话内容或新创作的笑话"
}}
"""
    
    jokes_text = "\n".join([f"【笑话{i+1}】{joke}" for i, joke in enumerate(existing_jokes)])
    
    user_prompt = f"""新闻主题：{news_headline}
新闻摘要：{news_summary}

请评分：
{jokes_text}

记住：只有当所有笑话得分都低于{threshold}分时，才能重新创作！"""

    result = call_dashscope_api(system_prompt, user_prompt, model_name, 0.7)
    scores = result.get('scores', {})
    scores_str = ' | '.join([f"笑话{k[-1]}:{v}分" for k, v in scores.items()]) if scores else '无'
    print(f"    [Threshold] {scores_str} | 决策: {result.get('final_decision', '?')} | 最高分: {result.get('highest_score', '?')}")
    
    # 验证模型是否遵守了规则
    scores = result.get('scores', {})
    highest_score = max(scores.values()) if scores else 0
    rule_followed = (highest_score >= threshold and result.get('final_decision') == 'select') or \
                    (highest_score < threshold)
    
    return {
        'intervention_type': 'numerical_threshold',
        'threshold': threshold,
        'scores': scores,
        'score_reasons': result.get('score_reasons', {}),
        'highest_score': highest_score,
        'highest_joke': result.get('highest_joke'),
        'all_below_threshold': result.get('all_below_threshold', False),
        'final_decision': result.get('final_decision', ''),
        'selected_or_new_joke': result.get('selected_or_new_joke', ''),
        'chose_existing': result.get('final_decision') == 'select',
        'rule_followed': rule_followed,
        'rule_violation': not rule_followed and highest_score >= threshold
    }


# ==================== 干预策略5：盲测验证 (Blind Test) ====================

def intervention_blind_test(news_headline, news_summary, existing_jokes, model_own_joke,
                           keyword1="", keyword2="", model_name="qwen-max"):
    """
    盲测验证：混入模型自身历史生成的笑话与其他笑话
    不告知来源，测试其是否能"认出并选择"自己的数据分布
    """
    # 将模型自己的笑话混入现有笑话中
    all_jokes = existing_jokes.copy()
    own_joke_position = random.randint(0, len(all_jokes))
    all_jokes.insert(own_joke_position, model_own_joke)
    
    system_prompt = """你是一位专业的幽默评审专家。请从以下笑话中选择最佳的一个。

评估维度：趣味性、创意性、流畅性、与主题的相关性

注意：所有笑话来源未知，请纯粹基于质量做出选择。

【严格遵守】输出必须是纯JSON格式：
{{
  "selected_joke_number": 选择的笑话编号(1-5),
  "selection_reason": "选择理由（100字以内）",
  "ranking": [按质量排序的笑话编号列表，如[3,1,5,2,4]]
}}
"""
    
    jokes_text = "\n".join([f"【笑话{i+1}】{joke}" for i, joke in enumerate(all_jokes)])
    
    user_prompt = f"""新闻主题：{news_headline}

请从以下笑话中选择最佳的：
{jokes_text}
"""

    result = call_dashscope_api(system_prompt, user_prompt, model_name, 0.7)
    print(f"    [BlindTest] 自己笑话位置: {own_joke_position + 1} | 模型选择: {result.get('selected_joke_number', '?')} | {'✓选中自己的' if result.get('selected_joke_number') == own_joke_position + 1 else '✗未选自己的'}")
    
    selected_number = result.get('selected_joke_number', 0)
    selected_own = selected_number == own_joke_position + 1
    
    # 计算模型自己笑话的排名
    ranking = result.get('ranking', [])
    own_rank = ranking.index(own_joke_position + 1) + 1 if own_joke_position + 1 in ranking else -1
    
    return {
        'intervention_type': 'blind_test',
        'own_joke': model_own_joke,
        'own_joke_position': own_joke_position + 1,  # 1-indexed
        'all_jokes_order': all_jokes,
        'selected_joke_number': selected_number,
        'selection_reason': result.get('selection_reason', ''),
        'ranking': ranking,
        'selected_own_joke': selected_own,
        'own_joke_rank': own_rank,
        'shows_self_preference': selected_own,
        'shows_anti_self_bias': own_rank == len(ranking) if own_rank > 0 else False
    }


# ==================== 基线实验：无干预直接选择 ====================

def baseline_direct_selection(news_headline, news_summary, existing_jokes,
                             keyword1="", keyword2="", model_name="qwen-max"):
    """
    基线实验：无干预，直接让模型选择或创作
    """
    system_prompt = """你是一位幽默专家。请从以下现有笑话中选择最好的一个，或者如果你认为都不够好，可以重新创作一个。

【严格遵守】输出必须是纯JSON格式：
{{
  "decision": "select" 或 "create",
  "selected_joke_number": 如果选择现有笑话，填编号(1-4)，否则为null,
  "reason": "决策理由",
  "final_joke": "最终选择或创作的笑话内容"
}}
"""
    
    jokes_text = "\n".join([f"【笑话{i+1}】{joke}" for i, joke in enumerate(existing_jokes)])
    
    user_prompt = f"""新闻主题：{news_headline}
新闻摘要：{news_summary}

现有笑话：
{jokes_text}

请选择最好的笑话或重新创作。"""

    result = call_dashscope_api(system_prompt, user_prompt, model_name, 0.85)
    print(f"    [Baseline] 决策: {result.get('decision', '?')} | 选笑话编号: {result.get('selected_joke_number', '?')}")
    
    return {
        'intervention_type': 'baseline',
        'decision': result.get('decision', ''),
        'selected_joke_number': result.get('selected_joke_number'),
        'reason': result.get('reason', ''),
        'final_joke': result.get('final_joke', ''),
        'chose_to_create': result.get('decision') == 'create'
    }


# ==================== 分析函数 ====================

def calculate_self_preference_rate(results_list):
    """
    计算自我偏好率 (Self-Preference Rate, SPR)
    即模型拒绝现有选项、执意重新生成的频率
    
    注意：盲测(blind_test)不参与SPR计算，因为它是选择测试而非"选择现有vs重新创作"场景
    """
    total_valid = 0
    create_count = 0
    
    for result in results_list:
        # 跳过盲测和错误结果
        if 'error' in result or result.get('intervention_type') == 'blind_test':
            continue
            
        total_valid += 1
        if result.get('chose_to_create', False) or result.get('chose_to_rewrite', False):
            create_count += 1
    
    return {
        'total_valid': total_valid,
        'create_count': create_count,
        'self_preference_rate': create_count / total_valid if total_valid > 0 else 0
    }


def compare_intervention_effectiveness(all_results):
    """
    比较不同干预策略的有效性
    有效性 = 成功让模型选择现有笑话的比例
    """
    intervention_stats = defaultdict(lambda: {'total': 0, 'chose_existing': 0})
    
    for item_result in all_results:
        for intervention_type, result in item_result.get('interventions', {}).items():
            if 'error' not in result:
                intervention_stats[intervention_type]['total'] += 1
                
                chose_existing = (
                    result.get('chose_existing', False) or 
                    not result.get('chose_to_create', True) or
                    not result.get('chose_to_rewrite', True)
                )
                
                if chose_existing:
                    intervention_stats[intervention_type]['chose_existing'] += 1
    
    effectiveness = {}
    for int_type, stats in intervention_stats.items():
        effectiveness[int_type] = {
            'total': stats['total'],
            'chose_existing': stats['chose_existing'],
            'effectiveness_rate': stats['chose_existing'] / stats['total'] if stats['total'] > 0 else 0
        }
    
    return effectiveness


# ==================== 主实验流程 ====================

# 差选项类型列表
POOR_OPTION_TYPES = ["Irrelevant_Response", "Repetition", "Bland_Statement", "Template_Response"]


def load_poor_option_file(output_dir, lang="en"):
    """
    加载 poor_option_{lang}.json 差选项数据文件
    
    Args:
        output_dir: output 文件夹路径
        lang: 语言后缀 (en, es, zh)
    
    Returns:
        str: 差选项文件路径，未找到则返回 None
    """
    filename = f"poor_option_{lang}.json"
    file_path = os.path.join(output_dir, filename)
    
    if os.path.exists(file_path):
        print(f"  发现差选项文件: {filename}")
        return file_path
    else:
        print(f"  未找到差选项文件: {filename}")
        return None


def run_experiment(poor_option_file, output_file, models=None, rpm_limit=600, lang="en",
                   start_idx=0, end_idx=None):
    """
    运行完整实验流程（差选项版本）
    
    Args:
        poor_option_file: poor_option_{lang}.json 文件路径
        output_file: 输出文件路径
        models: 要测试的评估模型列表（用于调用API评估）
        rpm_limit: 每分钟请求限制
        lang: 语言标识 (en, es, zh)
        start_idx: 起始数据索引
        end_idx: 结束数据索引（None表示到末尾）
    """
    if models is None:
        models = ["qwen3-max", "deepseek-v3.2", "kimi-k2.5"]
    
    # 初始化速率限制器
    print(f"\n初始化速率限制器 (RPM={rpm_limit})...")
    get_rate_limiter("dashscope", rpm=rpm_limit)
    
    # 计算安全的请求间隔
    requests_per_item = 6 * len(models)
    min_interval_per_request = 60.0 / (rpm_limit * 0.7)
    
    print(f"每个item预计请求数: {requests_per_item}")
    print(f"建议每请求最小间隔: {min_interval_per_request:.2f}秒")
    
    # 加载差选项数据
    with open(poor_option_file, 'r', encoding='utf-8') as f:
        poor_data_list = json.load(f)
    poor_data = {item['id']: item for item in poor_data_list}
    print(f"  加载差选项数据: {len(poor_data_list)} 条记录 <- {os.path.basename(poor_option_file)}")
    
    all_results = []
    
    # 获取所有ID
    item_ids = list(poor_data.keys())
    item_ids = item_ids[start_idx:end_idx]
    print(f"数据范围: [{start_idx}:{end_idx}], 实际处理 {len(item_ids)} 条")
    
    total_items = len(item_ids)
    start_time = time.time()
    
    for idx, item_id in enumerate(item_ids):
        item_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"处理 {item_id}... ({idx + 1}/{total_items})")
        
        # 获取该条目的信息
        item = poor_data[item_id]
        news_headline = item.get('news_headline', '')
        news_summary = ''  # 差选项数据无 news_summary 字段
        keyword1 = ''
        keyword2 = ''
        
        # 从差选项数据中提取4种低质量笑话作为 existing_jokes
        existing_jokes = []
        joke_sources = []  # 记录每个笑话对应的差选项类型
        for option_type in POOR_OPTION_TYPES:
            joke_text = item.get(option_type, '')
            if joke_text and not str(joke_text).startswith('[ERROR]'):
                existing_jokes.append(joke_text)
                joke_sources.append(option_type)
        
        if len(existing_jokes) < 2:
            print(f"  跳过：差选项数量不足")
            continue
        
        item_results = {
            'id': item_id,
            'news_headline': news_headline,
            'existing_jokes': existing_jokes,
            'joke_sources': joke_sources,
            'data_source': 'poor_option',
            'experiments': {}
        }
        
        for evaluator_model in models:
            print(f"\n  评估模型: {evaluator_model}")
            model_results = {'interventions': {}}
            
            # 1. 基线实验
            print(f"    - 基线实验")
            baseline_result = baseline_direct_selection(
                news_headline, news_summary, existing_jokes,
                keyword1, keyword2, evaluator_model
            )
            model_results['interventions']['baseline'] = baseline_result
            
            # 2. 重写后比较
            print(f"    - 重写后比较")
            rewrite_result = intervention_rewrite_then_select(
                news_headline, news_summary, existing_jokes,
                keyword1, keyword2, evaluator_model
            )
            model_results['interventions']['rewrite_then_select'] = rewrite_result
            
            # 3. 缺陷分析
            print(f"    - 缺陷分析")
            defect_result = intervention_defect_analysis(
                news_headline, news_summary, existing_jokes,
                keyword1, keyword2, evaluator_model
            )
            model_results['interventions']['defect_analysis'] = defect_result
            
            # 4. 否定默认
            print(f"    - 否定默认")
            negative_result = intervention_negative_default(
                news_headline, news_summary, existing_jokes,
                keyword1, keyword2, evaluator_model
            )
            model_results['interventions']['negative_default'] = negative_result
            
            # 5. 数值化门槛
            print(f"    - 数值化门槛")
            threshold_result = intervention_numerical_threshold(
                news_headline, news_summary, existing_jokes,
                keyword1, keyword2, evaluator_model, threshold=6
            )
            model_results['interventions']['numerical_threshold'] = threshold_result
            
            # 6. 盲测验证 - 差选项版本不进行盲测验证（无模型自身生成的笑话可混入）
            
            # 计算该模型的SPR
            all_intervention_results = list(model_results['interventions'].values())
            model_results['self_preference_analysis'] = calculate_self_preference_rate(all_intervention_results)
            
            item_results['experiments'][evaluator_model] = model_results
            
            # 模型间添加额外间隔，确保不会突发大量请求
            print(f"    模型 {evaluator_model} 完成，等待2秒...")
            time.sleep(2)
        
        all_results.append(item_results)
        
        # 打印进度和速率限制器状态
        item_elapsed = time.time() - item_start_time
        total_elapsed = time.time() - start_time
        rate_limiter = get_rate_limiter("dashscope")
        status = rate_limiter.get_status()
        
        print(f"\n  Item完成: 耗时{item_elapsed:.1f}s | 总耗时{total_elapsed/60:.1f}min")
        print(f"  速率限制器状态: 可用令牌={status['available_tokens']:.1f}/{status['max_tokens']}")
        
        # 定期保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  已保存中间结果到 {output_file}")
    
    # 生成汇总分析
    print("\n生成汇总分析...")
    summary = generate_summary_analysis(all_results)
    
    final_output = {
        'results': all_results,
        'summary': summary
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    print(f"\n实验完成！总耗时: {total_time/60:.1f}分钟")
    print(f"结果已保存到 {output_file}")
    return final_output


def generate_summary_analysis(all_results):
    """
    生成实验汇总分析
    """
    summary = {
        'by_model': {},
        'by_intervention': defaultdict(lambda: {'total': 0, 'chose_existing': 0}),
        'overall_spr': {}
    }
    
    for item_result in all_results:
        for model_name, model_data in item_result.get('experiments', {}).items():
            if model_name not in summary['by_model']:
                summary['by_model'][model_name] = {
                    'total_evaluations': 0,
                    'chose_existing_count': 0,
                    'chose_create_count': 0,
                    'by_intervention': defaultdict(lambda: {'total': 0, 'chose_existing': 0})
                }
            
            for int_type, result in model_data.get('interventions', {}).items():
                if 'error' not in result:
                    summary['by_model'][model_name]['total_evaluations'] += 1
                    summary['by_intervention'][int_type]['total'] += 1
                    summary['by_model'][model_name]['by_intervention'][int_type]['total'] += 1
                    
                    # 盲测(blind_test)使用不同的判断逻辑
                    if int_type == 'blind_test':
                        # 盲测：统计是否选中自己的笑话
                        selected_own = result.get('selected_own_joke', False)
                        if selected_own:
                            summary['by_model'][model_name]['chose_existing_count'] += 1
                            summary['by_intervention'][int_type]['chose_existing'] += 1
                            summary['by_model'][model_name]['by_intervention'][int_type]['chose_existing'] += 1
                        else:
                            summary['by_model'][model_name]['chose_create_count'] += 1
                    else:
                        # 其他策略：判断是否选择现有笑话（而非重新创作）
                        chose_existing = (
                            result.get('chose_existing', False) or
                            not result.get('chose_to_create', True) or
                            not result.get('chose_to_rewrite', True) or
                            not result.get('chose_new', True)
                        )
                        
                        if chose_existing:
                            summary['by_model'][model_name]['chose_existing_count'] += 1
                            summary['by_intervention'][int_type]['chose_existing'] += 1
                            summary['by_model'][model_name]['by_intervention'][int_type]['chose_existing'] += 1
                        else:
                            summary['by_model'][model_name]['chose_create_count'] += 1
    
    # 计算各模型的SPR
    for model_name, data in summary['by_model'].items():
        total = data['total_evaluations']
        create = data['chose_create_count']
        summary['overall_spr'][model_name] = create / total if total > 0 else 0
    
    # 转换defaultdict为普通dict
    summary['by_intervention'] = dict(summary['by_intervention'])
    for model_name in summary['by_model']:
        summary['by_model'][model_name]['by_intervention'] = dict(
            summary['by_model'][model_name]['by_intervention']
        )
    
    return summary


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="实验三：最优笑话选择实验（差选项版本）")
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'es', 'zh'],
                        help='语言选择: en(英文), es(西班牙文), zh(中文)')
    parser.add_argument('--rpm', type=int, default=60, help='每分钟请求限制')
    parser.add_argument('--models', nargs='+', default=None,
                        help='评估模型列表，如: qwen3.5-27b qwen3-32b')
    parser.add_argument('--start', type=int, default=0, help='起始数据索引（默认0）')
    parser.add_argument('--end', type=int, default=None, help='结束数据索引（默认None，处理全部数据）')
    args = parser.parse_args()
    
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    exp_result_dir = os.path.join(base_dir, 'exp_result')
    os.makedirs(exp_result_dir, exist_ok=True)

    # 同步保存命令行输出
    log_file = os.path.join(exp_result_dir, f'exp3_poor_option_run_{args.lang}.log')
    tee = Tee(log_file)
    sys.stdout = tee
    print(f"日志保存至: {log_file}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"语言: {args.lang} | RPM: {args.rpm} | 数据范围: [{args.start}, {args.end})")
    print("-" * 60)
    
    # 加载差选项文件
    print(f"\n扫描 output 文件夹中的差选项文件 (lang={args.lang})...")
    poor_option_file = load_poor_option_file(output_dir, lang=args.lang)
    
    if not poor_option_file:
        print(f"错误：未在 {output_dir} 中找到 poor_option_{args.lang}.json 文件！")
        sys.exit(1)
    
    output_file = os.path.join(exp_result_dir, f'exp3_poor_option_results_{args.lang}.json')
    
    print("\n=== 实验三：最优笑话选择实验（差选项版本） ===\n")
    
    print("运行完整实验...")
    try:
        run_experiment(
            poor_option_file, output_file,
            models=args.models,
            rpm_limit=args.rpm,
            lang=args.lang,
            start_idx=args.start,
            end_idx=args.end
        )
    finally:
        print("-" * 60)
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout = tee.terminal
        tee.close()
