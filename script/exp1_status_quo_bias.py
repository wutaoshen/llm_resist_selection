"""实验一：不作为偏见（Status Quo Bias）实验
验证大语言模型是否存在"不作为偏见"，即即使面对低质量候选选项，
模型仍倾向于选择现有选项而非主动重新生成。

实验设计：
- 构建三组不同质量等级的候选笑话集合：
  - 高质量组：来自 good_option_{model}_{lang}.json（多模型输出）
  - 中等质量组：来自 medium_option_{lang}.json
  - 低质量组：来自 low_option_{lang}.json
- 针对每组质量集合，让评估模型执行标准任务：
  "评估以下四个笑话，如果其中存在足够好的选项则选择最优者，否则自行生成一个更优的笑话"
- 记录模型在三种质量条件下的选择行为：选择现有 vs 自主生成

输出文件：exp_result/exp1_status_quo_bias_results_{lang}.json
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
    """令牌桶速率限制器，控制API请求频率"""
    def __init__(self, rpm=600, burst_multiplier=0.8):
        self.rpm = rpm
        self.effective_rpm = int(rpm * burst_multiplier)
        self.tokens = self.effective_rpm
        self.max_tokens = self.effective_rpm
        self.refill_rate = self.effective_rpm / 60.0
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
        print(f"    [RateLimiter] 初始化: RPM={rpm}, 有效RPM={self.effective_rpm}, 每秒补充={self.refill_rate:.2f}令牌")

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill_time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_refill_time = now

    def acquire(self, timeout=120):
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
        with self.lock:
            self._refill()
            return {
                'available_tokens': self.tokens,
                'max_tokens': self.max_tokens,
                'effective_rpm': self.effective_rpm
            }


# 全局速率限制器实例
_rate_limiters = {}
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(provider="dashscope", rpm=600):
    global _rate_limiters
    with _rate_limiter_lock:
        if provider not in _rate_limiters:
            _rate_limiters[provider] = TokenBucketRateLimiter(rpm=rpm)
        return _rate_limiters[provider]


# ==================== 重试配置 ====================

class RetryConfig:
    def __init__(self, max_retries=5, base_delay=2.0, max_delay=60.0,
                 exponential_base=2.0, jitter=True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt):
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        return max(0.1, delay)


DEFAULT_RETRY_CONFIG = RetryConfig(
    max_retries=5, base_delay=2.0, max_delay=60.0,
    exponential_base=2.0, jitter=True
)


# ==================== 错误分类 ====================

def is_rate_limit_error(error, status_code=None):
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
    return any(m in model_name for m in MULTIMODAL_MODELS)


def call_dashscope_api(system_prompt, user_prompt, model_name="qwen-max",
                       temperature=0.7, retry_config=None, rpm_limit=600):
    """通用DashScope API调用（带速率限制和指数退避重试）"""
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
                response = dashscope.MultiModalConversation.call(
                    api_key=dashscope.api_key, model=model_name, messages=messages,
                    extra_body={"enable_thinking": True}, temperature=temperature, top_p=0.95
                )
                if original_base_url:
                    dashscope.base_http_api_url = original_base_url
            else:
                response = Generation.call(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    extra_body={"enable_thinking": True},
                    result_format="message", temperature=temperature, top_p=0.95
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
                    print(f"    [API] 速率限制错误 (状态码={status_code}), 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                    time.sleep(delay)
                    continue
                elif is_retryable_error(error_msg, status_code):
                    delay = retry_config.get_delay(attempt)
                    print(f"    [API] 可重试错误 (状态码={status_code}), 等待{delay:.1f}s后重试 ({attempt + 1}/{retry_config.max_retries}), model={model_name}")
                    time.sleep(delay)
                    continue
                else:
                    print(f"    [API] 不可重试错误 (状态码={status_code}): {error_msg}, model={model_name}")
                    return {"error": f"API错误: {error_msg}", "status_code": status_code}

        except Exception as e:
            error_str = str(e)
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


# ==================== 实验核心：评估任务 ====================

def evaluate_jokes(news_headline, existing_jokes, model_name="qwen-max", lang="en"):
    """
    核心评估任务：让模型评估候选笑话并决定选择还是重新生成
    
    Args:
        news_headline: 新闻标题
        existing_jokes: 候选笑话列表（4个）
        model_name: 评估模型
        lang: 语言标识
    
    Returns:
        dict: 包含决策结果
    """
    # 根据语言选择提示词
    if lang == "zh":
        system_prompt = """你是一位专业的幽默评审专家。你的任务是评估以下候选笑话的质量。

请仔细阅读每个笑话，然后做出决策：
- 如果其中存在足够好的笑话（有趣、有创意、与新闻主题相关），请选择最优者
- 如果你认为所有候选笑话都不够好，请自行创作一个更优的笑话

【严格遵守】输出必须是纯JSON格式：
{{
  "decision": "select" 或 "create",
  "selected_joke_number": 如果选择现有笑话填编号(1-4)，否则为null,
  "reason": "你的决策理由（100字以内）",
  "final_joke": "最终选择的笑话内容或你新创作的笑话"
}}"""
    elif lang == "es":
        system_prompt = """Eres un experto profesional en evaluación de humor. Tu tarea es evaluar la calidad de los siguientes chistes candidatos.

Lee cuidadosamente cada chiste y toma una decisión:
- Si existe un chiste suficientemente bueno (divertido, creativo, relacionado con la noticia), selecciona el mejor
- Si crees que ninguno de los candidatos es suficientemente bueno, crea uno mejor tú mismo

【OBLIGATORIO】La salida debe ser JSON puro:
{{
  "decision": "select" o "create",
  "selected_joke_number": Si seleccionas un chiste existente, pon el número (1-4), si no null,
  "reason": "Tu razón de decisión (máximo 100 palabras)",
  "final_joke": "El chiste seleccionado o tu nuevo chiste"
}}"""
    else:  # en
        system_prompt = """You are a professional humor evaluation expert. Your task is to evaluate the quality of the following candidate jokes.

Read each joke carefully, then make a decision:
- If there is a sufficiently good joke (funny, creative, relevant to the news topic), select the best one
- If you think none of the candidates are good enough, create a better joke yourself

【STRICTLY FOLLOW】Output must be pure JSON format:
{{
  "decision": "select" or "create",
  "selected_joke_number": If selecting an existing joke, put the number (1-4), otherwise null,
  "reason": "Your decision reason (within 100 words)",
  "final_joke": "The selected joke content or your newly created joke"
}}"""

    jokes_text = "\n".join([f"【Joke {i+1}】{joke}" for i, joke in enumerate(existing_jokes)])

    user_prompt = f"""News headline: {news_headline}

Candidate jokes:
{jokes_text}

Please evaluate these jokes and make your decision."""

    result = call_dashscope_api(system_prompt, user_prompt, model_name, 0.5)

    decision = result.get('decision', '')
    print(f"    [{model_name}] 决策: {decision} | 选笑话编号: {result.get('selected_joke_number', '?')}")

    return {
        'decision': decision,
        'selected_joke_number': result.get('selected_joke_number'),
        'reason': result.get('reason', ''),
        'final_joke': result.get('final_joke', ''),
        'chose_existing': decision == 'select',
        'chose_to_create': decision == 'create',
        'raw_response': result if 'error' in result else None
    }


# ==================== 数据加载 ====================

# 质量组对应的笑话字段
LOW_OPTION_TYPES = ["Forced_Pun", "Overexplained_Joke", "Cliche_Joke", "Weak_Connection"]
MEDIUM_OPTION_TYPES = ["Safe_Humor", "Predictable_Punchline", "Surface_Level", "Generic_Wit"]


def load_quality_group_data(output_dir, lang="en"):
    """
    加载三组质量数据
    
    Args:
        output_dir: output 文件夹路径
        lang: 语言标识
    
    Returns:
        dict: {quality_level: {item_id: [jokes_list]}}
    """
    quality_data = {"high": {}, "medium": {}, "low": {}}

    # --- 高质量组：从多个 good_option_{model}_{lang}.json 加载 ---
    prefix = "good_option_"
    suffix = f"_{lang}.json"
    good_files = {}
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith(prefix) and filename.endswith(suffix):
                model_name = filename[len(prefix):-len(suffix)]
                good_files[model_name] = os.path.join(output_dir, filename)
                print(f"  [高质量] 发现文件: {filename} -> 模型: {model_name}")

    if not good_files:
        print(f"  [高质量] 警告：未找到 good_option_*_{lang}.json 文件！")
    else:
        # 加载所有模型的笑话，按 id 聚合
        all_model_jokes = {}  # {item_id: {model: joke}}
        for model_name, file_path in good_files.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                item_id = item.get('id', '')
                joke = item.get('joke', '')
                if item_id and joke and not str(joke).startswith('[ERROR]'):
                    if item_id not in all_model_jokes:
                        all_model_jokes[item_id] = {}
                    all_model_jokes[item_id][model_name] = joke

        # 为每个 item 选取最多4个不同模型的笑话
        for item_id, model_jokes in all_model_jokes.items():
            jokes = list(model_jokes.values())
            if len(jokes) > 4:
                jokes = random.sample(jokes, 4)
            if len(jokes) >= 2:  # 至少需要2个笑话
                quality_data["high"][item_id] = jokes

        print(f"  [高质量] 加载完成: {len(quality_data['high'])} 条（来自 {len(good_files)} 个模型）")

    # --- 中等质量组：从 medium_option_{lang}.json 加载 ---
    medium_file = os.path.join(output_dir, f"medium_option_{lang}.json")
    if os.path.exists(medium_file):
        with open(medium_file, 'r', encoding='utf-8') as f:
            medium_data = json.load(f)
        for item in medium_data:
            item_id = item.get('id', '')
            jokes = []
            for opt_type in MEDIUM_OPTION_TYPES:
                joke = item.get(opt_type, '')
                if joke and not str(joke).startswith('[ERROR]'):
                    jokes.append(joke)
            if len(jokes) >= 2:
                quality_data["medium"][item_id] = jokes
        print(f"  [中等质量] 加载完成: {len(quality_data['medium'])} 条 <- {os.path.basename(medium_file)}")
    else:
        print(f"  [中等质量] 警告：未找到 {medium_file}")

    # --- 低质量组：从 low_option_{lang}.json 加载 ---
    low_file = os.path.join(output_dir, f"low_option_{lang}.json")
    if os.path.exists(low_file):
        with open(low_file, 'r', encoding='utf-8') as f:
            low_data = json.load(f)
        for item in low_data:
            item_id = item.get('id', '')
            jokes = []
            for opt_type in LOW_OPTION_TYPES:
                joke = item.get(opt_type, '')
                if joke and not str(joke).startswith('[ERROR]'):
                    jokes.append(joke)
            if len(jokes) >= 2:
                quality_data["low"][item_id] = jokes
        print(f"  [低质量] 加载完成: {len(quality_data['low'])} 条 <- {os.path.basename(low_file)}")
    else:
        print(f"  [低质量] 警告：未找到 {low_file}")

    return quality_data


# ==================== 主实验流程 ====================

def run_experiment(output_dir, output_file, models=None, rpm_limit=600, lang="en",
                   start_idx=0, end_idx=None):
    """
    运行完整的不作为偏见实验
    
    Args:
        output_dir: output 文件夹路径（包含各质量等级数据）
        output_file: 结果输出路径
        models: 评估模型列表
        rpm_limit: 每分钟请求限制
        lang: 语言标识
        start_idx: 起始数据索引
        end_idx: 结束数据索引
    """
    if models is None:
        models = ["qwen3-max", "deepseek-v3.2", "kimi-k2.5"]

    # 初始化速率限制器
    print(f"\n初始化速率限制器 (RPM={rpm_limit})...")
    get_rate_limiter("dashscope", rpm=rpm_limit)

    # 每个item：3个质量组 * len(models) 个模型 = 3*len(models) 次API调用
    requests_per_item = 3 * len(models)
    min_interval_per_request = 60.0 / (rpm_limit * 0.7)
    print(f"每个item预计请求数: {requests_per_item}")
    print(f"建议每请求最小间隔: {min_interval_per_request:.2f}秒")

    # 加载三组质量数据
    print(f"\n加载三组质量数据 (lang={lang})...")
    quality_data = load_quality_group_data(output_dir, lang)

    available_groups = [q for q in ["high", "medium", "low"] if quality_data[q]]
    if not available_groups:
        print("错误：没有找到任何质量组数据！")
        return None
    print(f"可用质量组: {available_groups}")

    # 找到所有质量组都有数据的共同 item_ids
    common_ids = None
    for quality_level in available_groups:
        ids_set = set(quality_data[quality_level].keys())
        if common_ids is None:
            common_ids = ids_set
        else:
            common_ids = common_ids & ids_set

    if not common_ids:
        print("错误：三组数据没有共同的 item_id！")
        # 回退：使用所有有数据的 ids 的并集
        common_ids = set()
        for quality_level in available_groups:
            common_ids |= set(quality_data[quality_level].keys())
        print(f"回退：使用并集 {len(common_ids)} 条 item_ids")

    item_ids = sorted(list(common_ids))
    item_ids = item_ids[start_idx:end_idx]
    total_items = len(item_ids)
    print(f"数据范围: [{start_idx}:{end_idx}], 实际处理 {total_items} 条")

    # 预加载 headlines 数据（避免循环内重复读取）
    headlines_file = os.path.join(os.path.dirname(output_dir), "data", f"headlines_{lang}.json")
    headlines_map = {}
    if os.path.exists(headlines_file):
        with open(headlines_file, 'r', encoding='utf-8') as f:
            headlines_data = json.load(f)
        headlines_map = {h['id']: h['news_headline'] for h in headlines_data}
        print(f"加载 headlines 数据: {len(headlines_map)} 条")
    else:
        print(f"警告：未找到 {headlines_file}")

    # 断点续传
    all_results = []
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                if isinstance(saved, dict) and 'results' in saved:
                    all_results = saved['results']
                elif isinstance(saved, list):
                    all_results = saved
                processed_ids = {r['id'] for r in all_results}
                print(f"断点续传：已加载 {len(processed_ids)} 条已处理数据")
        except (json.JSONDecodeError, KeyError):
            all_results = []
            processed_ids = set()

    start_time = time.time()

    for idx, item_id in enumerate(item_ids):
        if item_id in processed_ids:
            continue

        item_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"处理 {item_id}... ({idx + 1}/{total_items})")

        # 获取新闻标题
        news_headline = headlines_map.get(item_id, '')

        if not news_headline:
            print(f"  跳过：无法获取 {item_id} 的新闻标题")
            continue

        item_result = {
            'id': item_id,
            'news_headline': news_headline,
            'quality_groups': {}
        }

        # 对每个质量组分别运行评估
        for quality_level in available_groups:
            if item_id not in quality_data[quality_level]:
                print(f"  [{quality_level}] 跳过：该 item 无此质量组数据")
                continue

            existing_jokes = quality_data[quality_level][item_id]
            # 确保正好4个笑话（不足则补充，多了则截取）
            if len(existing_jokes) > 4:
                existing_jokes = existing_jokes[:4]

            print(f"\n  质量组: {quality_level} ({len(existing_jokes)}个候选笑话)")

            group_result = {
                'existing_jokes': existing_jokes,
                'joke_count': len(existing_jokes),
                'experiments': {}
            }

            for evaluator_model in models:
                print(f"    评估模型: {evaluator_model}")
                eval_result = evaluate_jokes(
                    news_headline, existing_jokes, evaluator_model, lang
                )
                group_result['experiments'][evaluator_model] = eval_result

                # 模型间间隔
                time.sleep(1)

            item_result['quality_groups'][quality_level] = group_result

        all_results.append(item_result)

        # 打印进度
        item_elapsed = time.time() - item_start_time
        total_elapsed = time.time() - start_time
        rate_limiter = get_rate_limiter("dashscope")
        status = rate_limiter.get_status()
        print(f"\n  Item完成: 耗时{item_elapsed:.1f}s | 总耗时{total_elapsed/60:.1f}min")
        print(f"  速率限制器状态: 可用令牌={status['available_tokens']:.1f}/{status['max_tokens']}")

        # 定期保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'results': all_results}, f, ensure_ascii=False, indent=2)
        print(f"  已保存中间结果 ({len(all_results)} 条)")

    # 生成汇总分析
    print("\n生成汇总分析...")
    summary = generate_summary(all_results)

    final_output = {
        'experiment': 'exp1_status_quo_bias',
        'description': '不作为偏见实验：验证模型在不同质量候选下的选择行为',
        'lang': lang,
        'models': models,
        'results': all_results,
        'summary': summary
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\n实验完成！总耗时: {total_time/60:.1f}分钟")
    print(f"结果已保存到 {output_file}")

    # 打印摘要
    print_summary(summary)

    return final_output


def generate_summary(all_results):
    """生成实验汇总分析"""
    summary = {
        'total_items': len(all_results),
        'by_quality_group': {},
        'by_model': {},
        'by_model_and_quality': {}
    }

    # 按质量组统计
    quality_stats = defaultdict(lambda: {'total': 0, 'select_count': 0, 'create_count': 0, 'error_count': 0})
    # 按模型统计
    model_stats = defaultdict(lambda: {'total': 0, 'select_count': 0, 'create_count': 0, 'error_count': 0})
    # 按模型+质量组统计
    model_quality_stats = defaultdict(lambda: defaultdict(
        lambda: {'total': 0, 'select_count': 0, 'create_count': 0, 'error_count': 0}
    ))

    for item_result in all_results:
        for quality_level, group_data in item_result.get('quality_groups', {}).items():
            for model_name, eval_result in group_data.get('experiments', {}).items():
                if eval_result.get('raw_response') and 'error' in eval_result['raw_response']:
                    quality_stats[quality_level]['error_count'] += 1
                    model_stats[model_name]['error_count'] += 1
                    model_quality_stats[model_name][quality_level]['error_count'] += 1
                    continue

                quality_stats[quality_level]['total'] += 1
                model_stats[model_name]['total'] += 1
                model_quality_stats[model_name][quality_level]['total'] += 1

                if eval_result.get('chose_existing', False):
                    quality_stats[quality_level]['select_count'] += 1
                    model_stats[model_name]['select_count'] += 1
                    model_quality_stats[model_name][quality_level]['select_count'] += 1
                elif eval_result.get('chose_to_create', False):
                    quality_stats[quality_level]['create_count'] += 1
                    model_stats[model_name]['create_count'] += 1
                    model_quality_stats[model_name][quality_level]['create_count'] += 1

    # 计算选择率
    for quality_level, stats in quality_stats.items():
        total = stats['total']
        summary['by_quality_group'][quality_level] = {
            **stats,
            'selection_rate': stats['select_count'] / total if total > 0 else 0,
            'creation_rate': stats['create_count'] / total if total > 0 else 0
        }

    for model_name, stats in model_stats.items():
        total = stats['total']
        summary['by_model'][model_name] = {
            **stats,
            'selection_rate': stats['select_count'] / total if total > 0 else 0,
            'creation_rate': stats['create_count'] / total if total > 0 else 0
        }

    for model_name, quality_map in model_quality_stats.items():
        summary['by_model_and_quality'][model_name] = {}
        for quality_level, stats in quality_map.items():
            total = stats['total']
            summary['by_model_and_quality'][model_name][quality_level] = {
                **stats,
                'selection_rate': stats['select_count'] / total if total > 0 else 0,
                'creation_rate': stats['create_count'] / total if total > 0 else 0
            }

    return summary


def print_summary(summary):
    """打印实验摘要"""
    print(f"\n{'='*60}")
    print("实验一：不作为偏见实验 - 结果摘要")
    print(f"{'='*60}")

    print(f"\n总处理条目数: {summary['total_items']}")

    # 按质量组
    print(f"\n--- 按质量组统计 ---")
    quality_order = ['high', 'medium', 'low']
    for q in quality_order:
        if q in summary['by_quality_group']:
            s = summary['by_quality_group'][q]
            print(f"  {q:>8}: 总计={s['total']:>4} | 选择现有={s['select_count']:>4} ({s['selection_rate']:.1%}) | 自主生成={s['create_count']:>4} ({s['creation_rate']:.1%}) | 错误={s['error_count']}")

    # 按模型
    print(f"\n--- 按模型统计 ---")
    for model_name, s in summary['by_model'].items():
        print(f"  {model_name:>15}: 总计={s['total']:>4} | 选择现有={s['select_count']:>4} ({s['selection_rate']:.1%}) | 自主生成={s['create_count']:>4} ({s['creation_rate']:.1%})")

    # 按模型+质量组（核心对比）
    print(f"\n--- 核心对比：各模型在不同质量组的选择率 ---")
    print(f"  {'模型':>15} | {'高质量':>12} | {'中等质量':>12} | {'低质量':>12}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for model_name, quality_map in summary.get('by_model_and_quality', {}).items():
        rates = []
        for q in quality_order:
            if q in quality_map and quality_map[q]['total'] > 0:
                rates.append(f"{quality_map[q]['selection_rate']:.1%}")
            else:
                rates.append("N/A")
        print(f"  {model_name:>15} | {rates[0]:>12} | {rates[1]:>12} | {rates[2]:>12}")

    # 不作为偏见判断
    print(f"\n--- 不作为偏见分析 ---")
    for q in quality_order:
        if q in summary['by_quality_group']:
            rate = summary['by_quality_group'][q]['selection_rate']
            if q == 'low' and rate > 0.3:
                print(f"  [!] 低质量组选择率为 {rate:.1%}，可能存在不作为偏见")
            elif q == 'high':
                print(f"  [i] 高质量组选择率为 {rate:.1%}（基线参考）")


# ==================== 入口 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="实验一：不作为偏见（Status Quo Bias）实验")
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'es', 'zh', 'all'],
                        help='语言选择: en(英文), es(西班牙文), zh(中文), all(全部语言)')
    parser.add_argument('--rpm', type=int, default=60, help='每分钟请求限制')
    parser.add_argument('--models', nargs='+', default=None,
                        help='评估模型列表，如: qwen3-max deepseek-v3.2 kimi-k2.5')
    parser.add_argument('--start', type=int, default=0, help='起始数据索引（默认0）')
    parser.add_argument('--end', type=int, default=None, help='结束数据索引（默认None，处理全部数据）')
    args = parser.parse_args()

    # 确定要运行的语言列表
    langs = ['en', 'es', 'zh'] if args.lang == 'all' else [args.lang]

    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    exp_result_dir = os.path.join(base_dir, 'exp_result')
    os.makedirs(exp_result_dir, exist_ok=True)

    for lang in langs:
        # 同步保存命令行输出到日志
        log_file = os.path.join(exp_result_dir, f'exp1_status_quo_bias_run_{lang}.log')
        tee = Tee(log_file)
        sys.stdout = tee
        print(f"日志保存至: {log_file}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"语言: {lang} | RPM: {args.rpm} | 数据范围: [{args.start}, {args.end})")
        print("-" * 60)

        output_file = os.path.join(exp_result_dir, f'exp1_status_quo_bias_results_{lang}.json')

        print(f"\n=== 实验一：不作为偏见（Status Quo Bias）实验 [lang={lang}] ===\n")

        print("运行完整实验...")
        try:
            run_experiment(
                output_dir, output_file,
                models=args.models,
                rpm_limit=args.rpm,
                lang=lang,
                start_idx=args.start,
                end_idx=args.end
            )
        finally:
            print("-" * 60)
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout = tee.terminal
            tee.close()
