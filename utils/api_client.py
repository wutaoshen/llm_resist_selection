"""API客户端模块：DashScope 和 OpenAI 兼容API的统一调用接口"""

import json
import os
import re
import time

import dashscope
from dashscope import Generation
from openai import OpenAI

from .rate_limiter import get_rate_limiter
from .retry import DEFAULT_RETRY_CONFIG, is_rate_limit_error, is_retryable_error

# ==================== 常量配置 ====================

# Kimi-K2.5 需要使用的 API 端点
KIMI_BASE_URL = 'https://dashscope.aliyuncs.com/api/v1'

# 需要使用 MultiModalConversation 调用的模型列表
MULTIMODAL_MODELS = ["kimi-k2.5", "kimi-k2.6", "qwen3.6-27b"]

# 需要使用 OpenAI 兼容 API 调用的模型列表（前缀匹配）
OPENAI_MODELS = ["gpt-5.4-2026-03-05-high", "gemini-3.1-pro-preview", "gpt-5.5"]

# 需要使用 DeepSeek 官方 OpenAI 兼容 API 调用的模型列表（前缀匹配）
DEEPSEEK_MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]

# API 密钥配置
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.zetatechs.com/v1")

# DeepSeek 官方 API 配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


# ==================== 辅助函数 ====================

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


# ==================== DashScope API 调用 ====================

def call_dashscope_api(system_prompt, user_prompt, model_name="qwen-max",
                       temperature=0.7, retry_config=None, rpm_limit=600,
                       enable_thinking=False):
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
        enable_thinking: 是否启用思考模式（DashScope平台参数），默认False（不启用）
    """
    if retry_config is None:
        retry_config = DEFAULT_RETRY_CONFIG

    rate_limiter = get_rate_limiter("dashscope", rpm=rpm_limit)
    is_multimodal = _is_multimodal_model(model_name)

    for attempt in range(retry_config.max_retries):
        try:
            if not rate_limiter.acquire(timeout=120):
                print(f"    [API] 速率限制等待超时, model={model_name}")
                continue

            if is_multimodal:
                original_base_url = getattr(dashscope, 'base_http_api_url', None)
                dashscope.base_http_api_url = KIMI_BASE_URL
                combined_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
                messages = [{"role": "user", "content": [{"text": combined_prompt}]}]
                extra_body = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
                response = dashscope.MultiModalConversation.call(
                    api_key=dashscope.api_key, model=model_name, messages=messages,
                    extra_body=extra_body, temperature=temperature, top_p=0.95,
                    timeout=120
                )
                if original_base_url:
                    dashscope.base_http_api_url = original_base_url
            else:
                extra_body = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
                response = Generation.call(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body=extra_body,
                    result_format="message", temperature=temperature, top_p=0.95,
                    timeout=120
                )

            if response.status_code == 200:
                if is_multimodal:
                    content = response.output.choices[0].message.content[0]["text"]
                else:
                    content = response.output.choices[0].message.content
                try:
                    return parse_json_response(content)
                except json.JSONDecodeError as e:
                    print(f"    [API] JSON解析失败: {e}, model={model_name}")
                    return {"error": f"JSON解析失败: {e}", "raw_content": content[:200]}
            else:
                status_code = response.status_code
                error_msg = getattr(response, 'message', str(response))
                if is_rate_limit_error(error_msg, status_code):
                    delay = retry_config.get_delay(attempt)
                    print(f"    [API] 速率限制错误 (状态码={status_code}), 等待{delay:.1f}s后重试 "
                          f"({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                    time.sleep(delay)
                    continue
                elif is_retryable_error(error_msg, status_code):
                    delay = retry_config.get_delay(attempt)
                    print(f"    [API] 可重试错误 (状态码={status_code}), 等待{delay:.1f}s后重试 "
                          f"({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                    time.sleep(delay)
                    continue
                else:
                    print(f"    [API] 不可重试错误 (状态码={status_code}): {error_msg}, model={model_name}")
                    return {"error": f"API错误: {error_msg}", "status_code": status_code}

        except Exception as e:
            error_str = str(e)
            if is_rate_limit_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [API] 速率限制异常: {error_str[:100]}, 等待{delay:.1f}s后重试 "
                      f"({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                time.sleep(delay)
                continue
            elif is_retryable_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [API] 可重试异常: {error_str[:100]}, 等待{delay:.1f}s后重试 "
                      f"({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                time.sleep(delay)
                continue
            else:
                print(f"    [API] 不可重试异常: {error_str}, model={model_name}")
                return {"error": f"API异常: {error_str}"}

    print(f"    [API] 全部重试失败, model={model_name}")
    return {"error": "API调用失败，已达最大重试次数", "attempts": retry_config.max_retries}


# ==================== 统一路由调用 ====================

def _is_openai_model(model_name):
    """判断是否为需要使用 OpenAI 兼容 API 调用的模型"""
    return any(model_name.startswith(prefix) for prefix in OPENAI_MODELS)


def _is_deepseek_model(model_name):
    """判断是否为需要使用 DeepSeek 官方 API 调用的模型"""
    return any(model_name.startswith(prefix) for prefix in DEEPSEEK_MODELS)


def call_api(system_prompt, user_prompt, model_name="qwen3-max",
             temperature=0.7, retry_config=None, rpm_limit=600,
             enable_thinking=False, reasoning_effort=None):
    """
    统一API调用路由：根据模型名称自动选择 DashScope / OpenAI / DeepSeek 调用方式

    - DEEPSEEK_MODELS 中的模型 -> call_openai_api()（使用 DeepSeek 官方 base_url / api_key）
    - OPENAI_MODELS 中的模型 -> call_openai_api()
    - 其他模型 -> call_dashscope_api()（内部自动区分普通/多模态）

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        model_name: 模型名称
        temperature: 温度参数
        retry_config: 重试配置
        rpm_limit: 每分钟请求限制
        enable_thinking: 是否启用思考模式（DashScope平台及非GPT的OpenAI兼容模型使用），默认False（不启用）
        reasoning_effort: 推理力度（GPT系列模型使用），可选 "low"/"medium"/"high"，默认None（不启用）
    """
    if _is_deepseek_model(model_name):
        print(f"    [路由] {model_name} -> DeepSeek API")
        return call_openai_api(
            system_prompt, user_prompt, model_name,
            temperature=temperature, retry_config=retry_config, rpm_limit=rpm_limit,
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort,
            api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL,
            rate_limiter_key="deepseek"
        )
    elif _is_openai_model(model_name):
        print(f"    [路由] {model_name} -> OpenAI API")
        return call_openai_api(
            system_prompt, user_prompt, model_name,
            temperature=temperature, retry_config=retry_config, rpm_limit=rpm_limit,
            enable_thinking=enable_thinking, reasoning_effort=reasoning_effort
        )
    else:
        print(f"    [路由] {model_name} -> DashScope API")
        return call_dashscope_api(
            system_prompt, user_prompt, model_name,
            temperature=temperature, retry_config=retry_config, rpm_limit=rpm_limit,
            enable_thinking=enable_thinking
        )


# ==================== OpenAI 兼容 API 调用 ====================

def call_openai_api(system_prompt, user_prompt, model_name="gpt-4",
                    temperature=0.7, retry_config=None, rpm_limit=60,
                    enable_thinking=False, reasoning_effort=None,
                    api_key=None, base_url=None, rate_limiter_key="openai"):
    """
    OpenAI兼容API调用（带速率限制和指数退避重试）
    根据模型名称自动选择思考模式参数：
    - GPT系列模型：使用 reasoning_effort 参数
    - DeepSeek 模型：根据 enable_thinking 传递 extra_body={"thinking": {"type": "enabled|disabled"}}
    - 其他模型（如Gemini）：使用 extra_body={"enable_thinking": True}

    Args:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
        model_name: 模型名称
        temperature: 温度参数
        retry_config: 重试配置
        rpm_limit: 每分钟请求限制
        enable_thinking: 是否启用思考模式（非GPT、非DeepSeek模型使用），默认False（不启用）
        reasoning_effort: 推理力度（GPT模型使用），可选 "low"/"medium"/"high"，默认None（不启用）
        api_key: 可选，自定义 API 密钥（用于 DeepSeek 等非默认端点）。为 None 时用 OPENAI_API_KEY
        base_url: 可选，自定义 base_url。为 None 时用 OPENAI_BASE_URL
        rate_limiter_key: 速率限制器的名称（不同服务商使用独立的速率限制器）
    """
    if retry_config is None:
        retry_config = DEFAULT_RETRY_CONFIG

    effective_api_key = api_key if api_key else OPENAI_API_KEY
    effective_base_url = base_url if base_url else OPENAI_BASE_URL
    client = OpenAI(api_key=effective_api_key, base_url=effective_base_url, timeout=120)
    rate_limiter = get_rate_limiter(rate_limiter_key, rpm=rpm_limit)

    for attempt in range(retry_config.max_retries):
        try:
            if not rate_limiter.acquire(timeout=120):
                print(f"    [OpenAI API] 速率限制等待超时, model={model_name}")
                continue

            is_gpt = "gpt" in model_name.lower()
            is_deepseek = _is_deepseek_model(model_name)
            create_params = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
            }
            if is_gpt:
                # GPT系列模型：使用 reasoning_effort 参数
                if reasoning_effort is not None:
                    create_params["reasoning_effort"] = reasoning_effort
            elif is_deepseek:
                # DeepSeek 官方模型：通过 enable_thinking 控制思考开关
                # True -> enabled, False -> disabled, None -> 不传递
                if enable_thinking is not None:
                    thinking_type = "enabled" if enable_thinking else "disabled"
                    create_params["extra_body"] = {"thinking": {"type": thinking_type}}
            else:
                # 非GPT模型（如Gemini）：使用 extra_body enable_thinking
                if enable_thinking is not None:
                    create_params["extra_body"] = {"enable_thinking": enable_thinking}
            response = client.chat.completions.create(**create_params)

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
                print(f"    [OpenAI API] 速率限制: {error_str[:100]}, 等待{delay:.1f}s后重试 "
                      f"({attempt + 1}/{retry_config.max_retries})")
                time.sleep(delay)
                continue
            elif is_retryable_error(e):
                delay = retry_config.get_delay(attempt)
                print(f"    [OpenAI API] 可重试错误: {error_str[:100]}, 等待{delay:.1f}s后重试 "
                      f"({attempt + 1}/{retry_config.max_retries})")
                time.sleep(delay)
                continue
            else:
                print(f"    [OpenAI API] 不可重试错误: {error_str}")
                return {"error": f"API异常: {error_str}"}

    return {"error": "API调用失败，已达最大重试次数", "attempts": retry_config.max_retries}
