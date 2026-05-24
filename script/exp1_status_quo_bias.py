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
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 sys.path 以支持 utils 导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import call_api, get_rate_limiter, Tee

# 全局打印锁，避免并发打印交错
_print_lock = threading.Lock()


def _safe_print(msg):
    """线程安全打印"""
    with _print_lock:
        print(msg)


# ==================== 实验核心：评估任务 ====================

def evaluate_jokes(news_headline, existing_jokes, model_name="qwen-max", lang="en",
                   enable_thinking=False, rpm_limit=600, reasoning_effort=None):
    """
    核心评估任务：让模型评估候选笑话并决定选择还是重新生成

    Args:
        news_headline: 新闻标题
        existing_jokes: 候选笑话列表（4个）
        model_name: 评估模型
        lang: 语言标识
        enable_thinking: 是否启用思考模式（透传给 call_api），默认 False
        rpm_limit: 每分钟请求限制（透传给 call_api），默认 600
        reasoning_effort: 推理力度（GPT系列模型使用），可选 "low"/"medium"/"high"，默认 None

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

    result = call_api(system_prompt, user_prompt, model_name, 0.5,
                      enable_thinking=enable_thinking, rpm_limit=rpm_limit,
                      reasoning_effort=reasoning_effort)

    decision = result.get('decision', '')
    _safe_print(f"    [{model_name}] 决策: {decision} | 选笑话编号: {result.get('selected_joke_number', '?')}")

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

        for item_id, model_jokes in all_model_jokes.items():
            jokes = list(model_jokes.values())
            if len(jokes) > 4:
                jokes = random.sample(jokes, 4)
            if len(jokes) >= 2:
                quality_data["high"][item_id] = jokes

        print(f"  [高质量] 加载完成: {len(quality_data['high'])} 条（来自 {len(good_files)} 个模型）")

    # --- 中等质量组 ---
    medium_file = os.path.join(output_dir, f"medium_option_{lang}.json")
    if os.path.exists(medium_file):
        with open(medium_file, 'r', encoding='utf-8') as f:
            medium_data = json.load(f)
        for item in medium_data:
            item_id = item.get('id', '')
            jokes = [item.get(t, '') for t in MEDIUM_OPTION_TYPES
                     if item.get(t, '') and not str(item.get(t, '')).startswith('[ERROR]')]
            if len(jokes) >= 2:
                quality_data["medium"][item_id] = jokes
        print(f"  [中等质量] 加载完成: {len(quality_data['medium'])} 条 <- {os.path.basename(medium_file)}")
    else:
        print(f"  [中等质量] 警告：未找到 {medium_file}")

    # --- 低质量组 ---
    low_file = os.path.join(output_dir, f"low_option_{lang}.json")
    if os.path.exists(low_file):
        with open(low_file, 'r', encoding='utf-8') as f:
            low_data = json.load(f)
        for item in low_data:
            item_id = item.get('id', '')
            jokes = [item.get(t, '') for t in LOW_OPTION_TYPES
                     if item.get(t, '') and not str(item.get(t, '')).startswith('[ERROR]')]
            if len(jokes) >= 2:
                quality_data["low"][item_id] = jokes
        print(f"  [低质量] 加载完成: {len(quality_data['low'])} 条 <- {os.path.basename(low_file)}")
    else:
        print(f"  [低质量] 警告：未找到 {low_file}")

    return quality_data


# ==================== 主实验流程 ====================

def run_experiment(output_dir, output_file, models=None, rpm_limit=600, lang="en",
                   start_idx=0, end_idx=None, enable_thinking=False, reasoning_effort=None):
    """运行完整的不作为偏见实验"""
    if models is None:
        models = ["qwen3.6-27b"] 
#"qwen3-max", "deepseek-v3.2", "kimi-k2.5"
#"qwen3.6-max-preview", "gpt-5.4-2026-03-05-high", 
#"kimi-k2.6",  "gemini-3.1-pro-preview", "deepseek-v4-flash", "deepseek-v4-pro"
    print(f"\n初始化速率限制器 (RPM={rpm_limit})...")
    # 预热所有可能用到的限流器，使 --rpm 对 DashScope/DeepSeek/OpenAI 路径均生效
    for rl_key in ("dashscope", "deepseek", "openai"):
        get_rate_limiter(rl_key, rpm=rpm_limit)

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

    # 找到共同 item_ids
    common_ids = None
    for quality_level in available_groups:
        ids_set = set(quality_data[quality_level].keys())
        common_ids = ids_set if common_ids is None else (common_ids & ids_set)

    if not common_ids:
        print("错误：三组数据没有共同的 item_id！")
        common_ids = set()
        for quality_level in available_groups:
            common_ids |= set(quality_data[quality_level].keys())
        print(f"回退：使用并集 {len(common_ids)} 条 item_ids")

    item_ids = sorted(list(common_ids))[start_idx:end_idx]
    total_items = len(item_ids)
    print(f"数据范围: [{start_idx}:{end_idx}], 实际处理 {total_items} 条")

    # 预加载 headlines 数据
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

        news_headline = headlines_map.get(item_id, '')
        if not news_headline:
            print(f"  跳过：无法获取 {item_id} 的新闻标题")
            continue

        item_result = {'id': item_id, 'news_headline': news_headline, 'quality_groups': {}}

        # 预构建本 item 所有任务：(quality_level, model, jokes)
        tasks = []
        for quality_level in available_groups:
            if item_id not in quality_data[quality_level]:
                print(f"  [{quality_level}] 跳过：该 item 无此质量组数据")
                continue
            existing_jokes = quality_data[quality_level][item_id][:4]
            print(f"  质量组: {quality_level} ({len(existing_jokes)}个候选笑话)")
            item_result['quality_groups'][quality_level] = {
                'existing_jokes': existing_jokes,
                'joke_count': len(existing_jokes),
                'experiments': {}
            }
            for evaluator_model in models:
                tasks.append((quality_level, evaluator_model, existing_jokes))

        # 并发执行：不同质量组 × 不同模型互不依赖，完全可并行
        max_workers = min(len(tasks), max(len(models) * 2, 4))
        print(f"  并发执行 {len(tasks)} 个评估任务 (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(evaluate_jokes, news_headline, jokes, model, lang,
                                enable_thinking, rpm_limit, reasoning_effort):
                (quality_level, model)
                for quality_level, model, jokes in tasks
            }
            for future in as_completed(future_map):
                quality_level, model = future_map[future]
                try:
                    eval_result = future.result(timeout=600)
                except Exception as e:
                    _safe_print(f"    [{model}] 评估异常: {e}")
                    eval_result = {
                        'decision': '',
                        'selected_joke_number': None,
                        'reason': '',
                        'final_joke': '',
                        'chose_existing': False,
                        'chose_to_create': False,
                        'raw_response': {'error': str(e)}
                    }
                item_result['quality_groups'][quality_level]['experiments'][model] = eval_result

        all_results.append(item_result)

        # 打印进度
        item_elapsed = time.time() - item_start_time
        total_elapsed = time.time() - start_time
        print(f"\n  Item完成: 耗时{item_elapsed:.1f}s | 总耗时{total_elapsed/60:.1f}min")
        for rl_key in ("dashscope", "deepseek", "openai"):
            status = get_rate_limiter(rl_key).get_status()
            print(f"  [{rl_key}] 可用令牌={status['available_tokens']:.1f}/{status['max_tokens']}")

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

    quality_stats = defaultdict(lambda: {'total': 0, 'select_count': 0, 'create_count': 0, 'error_count': 0})
    model_stats = defaultdict(lambda: {'total': 0, 'select_count': 0, 'create_count': 0, 'error_count': 0})
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

    print(f"\n--- 按质量组统计 ---")
    quality_order = ['high', 'medium', 'low']
    for q in quality_order:
        if q in summary['by_quality_group']:
            s = summary['by_quality_group'][q]
            print(f"  {q:>8}: 总计={s['total']:>4} | 选择现有={s['select_count']:>4} ({s['selection_rate']:.1%}) "
                  f"| 自主生成={s['create_count']:>4} ({s['creation_rate']:.1%}) | 错误={s['error_count']}")

    print(f"\n--- 按模型统计 ---")
    for model_name, s in summary['by_model'].items():
        print(f"  {model_name:>15}: 总计={s['total']:>4} | 选择现有={s['select_count']:>4} ({s['selection_rate']:.1%}) "
              f"| 自主生成={s['create_count']:>4} ({s['creation_rate']:.1%})")

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
    parser.add_argument('--lang', type=str, default='es', choices=['en', 'es', 'zh', 'all'],
                        help='语言选择: en(英文), es(西班牙文), zh(中文), all(全部语言)')
    parser.add_argument('--rpm', type=int, default=1200, help='每分钟请求限制')
    parser.add_argument('--models', nargs='+', default=None,
                        help='评估模型列表，如: qwen3-max deepseek-v3.2 kimi-k2.5')
    parser.add_argument('--start', type=int, default=0, help='起始数据索引（默认0）')
    parser.add_argument('--end', type=int, default=None, help='结束数据索引（默认None，处理全部数据）')
    parser.add_argument('--thinking', action='store_true',
                        help='启用思考模式（默认关闭）')
    parser.add_argument('--reasoning_effort', type=str, default='medium',
                        choices=['low', 'medium', 'high'],
                        help='GPT系列模型的推理力度: low/medium/high（默认 medium）')
    parser.add_argument('--suffix', type=str, default='',
                        help='输出文件名后缀，用于区分不同批次（如 _thinking / _no_thinking）')
    args = parser.parse_args()

    langs = ['en', 'es', 'zh'] if args.lang == 'all' else [args.lang]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'output')
    exp_result_dir = os.path.join(base_dir, 'exp_result')
    os.makedirs(exp_result_dir, exist_ok=True)

    for lang in langs:
        log_file = os.path.join(exp_result_dir, f'exp1_status_quo_bias_run_{lang}{args.suffix}.log')
        tee = Tee(log_file)
        sys.stdout = tee
        print(f"日志保存至: {log_file}")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"语言: {lang} | RPM: {args.rpm} | 数据范围: [{args.start}, {args.end}) | 思考模式: {args.thinking} | 推理力度: {args.reasoning_effort}")
        print("-" * 60)

        output_file = os.path.join(exp_result_dir, f'exp1_status_quo_bias_results_{lang}{args.suffix}.json')

        print(f"\n=== 实验一：不作为偏见（Status Quo Bias）实验 [lang={lang}] ===\n")

        print("运行完整实验...")
        try:
            run_experiment(
                output_dir, output_file,
                models=args.models,
                rpm_limit=args.rpm,
                lang=lang,
                start_idx=args.start,
                end_idx=args.end,
                enable_thinking=args.thinking,
                reasoning_effort=args.reasoning_effort
            )
        finally:
            print("-" * 60)
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout = tee.terminal
            tee.close()
