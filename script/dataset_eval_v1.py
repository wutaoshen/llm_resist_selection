"""数据集质量评估脚本

使用 GPT-5.5 对 exp1_status_quo_bias.py 实验使用的笑话数据集进行多维度打分。

评估维度（每项 0-10，0 最低，10 最高）：
  - news_relevance : 新闻相关性，笑话与新闻标题的关联程度
  - humor          : 幽默程度，笑话的有趣程度和娱乐价值
  - creativity     : 创新程度，笑话的原创性和创意水平
  - conciseness    : 简洁程度，表达的精炼程度和清晰度

数据来源（与 exp1 对齐）：
  - 高质量组：output/good_option_{model}_{lang}.json（多个模型）
  - 中等质量组：output/medium_option_{lang}.json
  - 低质量组：output/low_option_{lang}.json

输出文件：exp_result/dataset_eval_results_{lang}.json
日志文件：exp_result/dataset_eval_run_{lang}.log
"""

import json
import math
import os
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到 sys.path 以支持 utils 导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import call_openai_api, get_rate_limiter, Tee

# 可选依赖：scipy（用于更精确的 p 值；未安装时回退到正态近似）
try:
    from scipy import stats as _scipy_stats  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _scipy_stats = None
    _HAS_SCIPY = False

# 复用 exp1 中的质量组字段定义
LOW_OPTION_TYPES = ["Forced_Pun", "Overexplained_Joke", "Cliche_Joke", "Weak_Connection"]
MEDIUM_OPTION_TYPES = ["Safe_Humor", "Predictable_Punchline", "Surface_Level", "Generic_Wit"]

# 评分维度
SCORE_DIMENSIONS = ["news_relevance", "humor", "creativity", "conciseness"]

# 全局打印锁
_print_lock = threading.Lock()


def _safe_print(msg):
    with _print_lock:
        print(msg)


# ==================== 评估任务 ====================

def build_eval_prompt(lang: str):
    """根据语言返回评估用的 system_prompt"""
    if lang == "zh":
        return """你是一位专业的笑话质量评估专家。请对给定笑话从以下四个维度打分（0-10 分，0 最低，10 最高，允许小数）：
- news_relevance : 新闻相关性，笑话与新闻标题的关联程度
- humor          : 幽默程度，笑话的有趣程度和娱乐价值
- creativity     : 创新程度，笑话的原创性和创意水平
- conciseness    : 简洁程度，表达的精炼程度和清晰度

【严格遵守】只输出纯 JSON，不要任何解释、不要 markdown 代码块：
{"news_relevance": <数值>, "humor": <数值>, "creativity": <数值>, "conciseness": <数值>}"""
    elif lang == "es":
        return """Eres un experto profesional en evaluación de chistes. Puntúa el chiste dado en las siguientes cuatro dimensiones (0-10, 0 mínimo, 10 máximo, se permiten decimales):
- news_relevance : Relevancia con la noticia
- humor          : Nivel de humor y entretenimiento
- creativity     : Originalidad y creatividad
- conciseness    : Concisión y claridad de expresión

【OBLIGATORIO】Devuelve SOLO JSON puro, sin explicaciones ni bloques markdown:
{"news_relevance": <num>, "humor": <num>, "creativity": <num>, "conciseness": <num>}"""
    else:
        return """You are a professional joke quality evaluator. Rate the given joke on the following four dimensions (0-10, 0 is lowest, 10 is highest, decimals allowed):
- news_relevance : How relevant the joke is to the news headline
- humor          : How funny / entertaining the joke is
- creativity     : Originality and creative level of the joke
- conciseness    : Conciseness and clarity of expression

[STRICT] Output ONLY pure JSON, no explanation, no markdown fences:
{"news_relevance": <num>, "humor": <num>, "creativity": <num>, "conciseness": <num>}"""


def evaluate_single_joke(news_headline, joke_text, model_name, lang, rpm_limit):
    """调用 GPT-5.5 评估单条笑话，返回四维评分 dict 或 error dict"""
    system_prompt = build_eval_prompt(lang)

    if lang == "zh":
        user_prompt = f"新闻标题：{news_headline}\n\n笑话：{joke_text}\n\n请按要求输出 JSON 评分。"
    elif lang == "es":
        user_prompt = f"Titular: {news_headline}\n\nChiste: {joke_text}\n\nDevuelve la puntuación JSON según las instrucciones."
    else:
        user_prompt = f"News headline: {news_headline}\n\nJoke: {joke_text}\n\nReturn the JSON scores as instructed."

    # 直接使用 OpenAI 兼容接口调用 GPT-5.5（不走 call_api 路由，避免模型前缀列表维护）
    result = call_openai_api(
        system_prompt, user_prompt,
        model_name=model_name,
        temperature=0.2,
        rpm_limit=rpm_limit,
        reasoning_effort=None,
        enable_thinking=None,
    )

    if isinstance(result, dict) and "error" in result:
        return {"error": result.get("error", "unknown"), "raw": result}

    # 校验四维都存在且可转 float
    scores = {}
    for dim in SCORE_DIMENSIONS:
        val = result.get(dim)
        try:
            scores[dim] = float(val)
        except (TypeError, ValueError):
            return {"error": f"missing_or_invalid_dim: {dim}", "raw": result}
    return scores


# ==================== 数据加载 ====================

def load_dataset(output_dir, lang):
    """
    加载三组质量笑话数据。

    返回：
        list[dict]，每项结构：
            {
                "item_id": str,
                "quality_group": "high" | "medium" | "low",
                "source": str,          # 来源模型 或 类型字段名
                "joke_index": int,      # 该 item 在该质量组下的序号
                "joke": str
            }
    """
    tasks = []

    # --- 高质量组：多个 good_option_{model}_{lang}.json ---
    prefix = "good_option_"
    suffix = f"_{lang}.json"
    good_files = {}
    if os.path.exists(output_dir):
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith(prefix) and filename.endswith(suffix):
                model_name = filename[len(prefix):-len(suffix)]
                good_files[model_name] = os.path.join(output_dir, filename)
                print(f"  [高质量] 发现文件: {filename} -> {model_name}")

    if not good_files:
        print(f"  [高质量] 警告：未找到 good_option_*_{lang}.json")
    else:
        # 高质量组 source=模型名本身已唯一，joke_index 固定为 0，保证 task_key 与模型文件集合无关
        high_count = 0
        for model_name, file_path in sorted(good_files.items()):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                item_id = item.get("id", "")
                joke = item.get("joke", "")
                if item_id and joke and not str(joke).startswith("[ERROR]"):
                    tasks.append({
                        "item_id": item_id,
                        "quality_group": "high",
                        "source": model_name,
                        "joke_index": 0,
                        "joke": joke,
                    })
                    high_count += 1
        print(f"  [高质量] 加载完成: {high_count} 条笑话")

    # --- 中等质量组 ---
    medium_file = os.path.join(output_dir, f"medium_option_{lang}.json")
    if os.path.exists(medium_file):
        with open(medium_file, "r", encoding="utf-8") as f:
            medium_data = json.load(f)
        count = 0
        for item in medium_data:
            item_id = item.get("id", "")
            if not item_id:
                continue
            for idx, t in enumerate(MEDIUM_OPTION_TYPES):
                joke = item.get(t, "")
                if joke and not str(joke).startswith("[ERROR]"):
                    tasks.append({
                        "item_id": item_id,
                        "quality_group": "medium",
                        "source": t,
                        "joke_index": idx,
                        "joke": joke,
                    })
                    count += 1
        print(f"  [中等质量] 加载完成: {count} 条笑话")
    else:
        print(f"  [中等质量] 警告：未找到 {medium_file}")

    # --- 低质量组 ---
    low_file = os.path.join(output_dir, f"low_option_{lang}.json")
    if os.path.exists(low_file):
        with open(low_file, "r", encoding="utf-8") as f:
            low_data = json.load(f)
        count = 0
        for item in low_data:
            item_id = item.get("id", "")
            if not item_id:
                continue
            for idx, t in enumerate(LOW_OPTION_TYPES):
                joke = item.get(t, "")
                if joke and not str(joke).startswith("[ERROR]"):
                    tasks.append({
                        "item_id": item_id,
                        "quality_group": "low",
                        "source": t,
                        "joke_index": idx,
                        "joke": joke,
                    })
                    count += 1
        print(f"  [低质量] 加载完成: {count} 条笑话")
    else:
        print(f"  [低质量] 警告：未找到 {low_file}")

    return tasks


def load_headlines(base_dir, lang):
    """加载 headlines 映射"""
    headlines_file = os.path.join(base_dir, "data", f"headlines_{lang}.json")
    if not os.path.exists(headlines_file):
        print(f"  警告：未找到 headlines 文件 {headlines_file}")
        return {}
    with open(headlines_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {h["id"]: h["news_headline"] for h in data}


# ==================== 断点续传 ====================

def _task_key(t):
    """任务唯一标识"""
    return f"{t['item_id']}|{t['quality_group']}|{t['source']}|{t['joke_index']}"


def load_existing_results(output_file):
    """加载已有结果，返回 (results_list, processed_keys_set)

    仅将「有有效 scores」的记录视为已处理；失败记录（包含 error 或 scores 为 None）
    会保留在 results 中，但不计入 processed，下次运行时会自动重试，
    重试成功后新记录会追加，旧错误记录在最终 summary 生成前会被同 key 覆盖。
    """
    if not os.path.exists(output_file):
        return [], set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict) and "results" in saved:
            results = saved["results"]
        elif isinstance(saved, list):
            results = saved
        else:
            return [], set()
        processed = {
            _task_key(r) for r in results
            if r.get("scores") is not None and "error" not in r
        }
        # 丢弃未成功的旧记录，避免与新重试结果冲突
        results = [r for r in results if _task_key(r) in processed]
        err_dropped = 0
        try:
            err_dropped = len(saved.get("results", saved)) - len(results) if isinstance(saved, (dict, list)) else 0
        except Exception:
            err_dropped = 0
        print(f"断点续传：已加载 {len(processed)} 条成功记录"
              + (f"（丢弃 {err_dropped} 条失败记录待重试）" if err_dropped > 0 else ""))
        return results, processed
    except (json.JSONDecodeError, KeyError, TypeError):
        print("断点续传：已有结果文件解析失败，将重新开始")
        return [], set()


# ==================== 主流程 ====================

def run(output_dir, output_file, base_dir, model_name="gpt-5.5",
        lang="en", rpm_limit=600, concurrency=8, start_idx=0, end_idx=None,
        save_interval=20):
    print(f"\n初始化 OpenAI 速率限制器 (RPM={rpm_limit})...")
    get_rate_limiter("openai", rpm=rpm_limit)

    print(f"\n加载笑话数据集 (lang={lang})...")
    all_tasks = load_dataset(output_dir, lang)
    headlines_map = load_headlines(base_dir, lang)
    print(f"headlines 加载: {len(headlines_map)} 条")
    print(f"原始笑话总数: {len(all_tasks)}")

    # 过滤无 headline 的 item
    all_tasks = [t for t in all_tasks if headlines_map.get(t["item_id"])]
    # 切片
    all_tasks = all_tasks[start_idx:end_idx]
    print(f"本次处理任务数（切片后 [{start_idx}:{end_idx}]）: {len(all_tasks)}")

    # 断点续传
    results, processed = load_existing_results(output_file)
    pending = [t for t in all_tasks if _task_key(t) not in processed]
    print(f"待评估任务: {len(pending)}")

    if not pending:
        print("没有待评估任务，直接生成汇总。")
        summary = generate_summary(results)
        _save(output_file, results, summary, model_name, lang)
        print_summary(summary)
        return

    start_time = time.time()
    completed_count = 0
    lock = threading.Lock()

    def _work(task):
        headline = headlines_map.get(task["item_id"], "")
        return evaluate_single_joke(headline, task["joke"], model_name, lang, rpm_limit)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_map = {executor.submit(_work, t): t for t in pending}
        for future in as_completed(future_map):
            task = future_map[future]
            try:
                res = future.result()
            except Exception as e:
                res = {"error": f"exception: {e}"}

            record = {
                "item_id": task["item_id"],
                "quality_group": task["quality_group"],
                "source": task["source"],
                "joke_index": task["joke_index"],
                "joke": task["joke"],
                "news_headline": headlines_map.get(task["item_id"], ""),
            }
            if "error" in res:
                record["error"] = res["error"]
                record["scores"] = None
                _safe_print(f"  [ERR] {_task_key(task)} -> {res['error']}")
            else:
                record["scores"] = res
                _safe_print(f"  [OK ] {_task_key(task)} -> "
                            f"rel={res['news_relevance']:.1f} hum={res['humor']:.1f} "
                            f"crt={res['creativity']:.1f} cnc={res['conciseness']:.1f}")

            with lock:
                results.append(record)
                completed_count += 1
                if completed_count % save_interval == 0:
                    _save(output_file, results, None, model_name, lang)
                    elapsed = time.time() - start_time
                    _safe_print(f"  ... 已完成 {completed_count}/{len(pending)} "
                                f"(用时 {elapsed/60:.1f}min) 已中间保存")

    # 最终汇总并保存
    summary = generate_summary(results)
    _save(output_file, results, summary, model_name, lang)
    total_time = time.time() - start_time
    print(f"\n评估完成！本次新增 {completed_count} 条，总记录 {len(results)} 条，总耗时 {total_time/60:.1f} 分钟")
    print_summary(summary)


def _save(output_file, results, summary, model_name, lang):
    payload = {
        "experiment": "dataset_eval",
        "model": model_name,
        "lang": lang,
        "score_dimensions": SCORE_DIMENSIONS,
        "results": results,
    }
    if summary is not None:
        payload["summary"] = summary
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ==================== 汇总统计 ====================

def _stat_block(values):
    """返回 {mean,max,min,count}"""
    if not values:
        return {"count": 0, "mean": None, "max": None, "min": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 3),
        "max": round(max(values), 3),
        "min": round(min(values), 3),
    }


def _bucketize(values, bins=(0, 2, 4, 6, 8, 10.0001)):
    """整体分布：按区间统计数量（末桶闭区间 [8,10]）"""
    def _label(lo, hi):
        return f"[{lo:.0f},{hi:.0f})" if hi < 10 else f"[{lo:.0f},10]"
    dist = {}
    for i in range(len(bins) - 1):
        dist[_label(bins[i], bins[i + 1])] = 0
    for v in values:
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if lo <= v < hi:
                dist[_label(lo, hi)] += 1
                break
    return dist


def _bucketize_z(values):
    """z-score 分布：按标准差区间统计数量"""
    labels = ["(-inf,-2)", "[-2,-1)", "[-1,0)", "[0,1)", "[1,2)", "[2,+inf)"]
    edges = [-float("inf"), -2, -1, 0, 1, 2, float("inf")]
    dist = {lab: 0 for lab in labels}
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                dist[labels[i]] += 1
                break
    return dist


# ---------- 通用数值辅助 ----------

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _var_sample(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _std_pop(xs):
    n = len(xs)
    if n < 1:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def _two_sided_p_from_z(z):
    return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))


def _rank(xs):
    """返回 xs 的秩（1-based，并列取平均秩）"""
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j + 2) / 2.0  # 1-based 平均秩
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


# ---------- 统计检验 ----------

def _welch_t_test(xs, ys):
    """Welch's t-test（双尾），返回 {t, df, p}"""
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        return None
    m1, m2 = _mean(xs), _mean(ys)
    v1, v2 = _var_sample(xs), _var_sample(ys)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return {"t": 0.0, "df": None, "p": 1.0}
    t = (m1 - m2) / se
    num = (v1 / n1 + v2 / n2) ** 2
    den = (v1 ** 2) / (n1 ** 2 * (n1 - 1)) + (v2 ** 2) / (n2 ** 2 * (n2 - 1))
    df = num / den if den > 0 else None
    if _HAS_SCIPY and df is not None:
        try:
            p = float(2 * (1 - _scipy_stats.t.cdf(abs(t), df)))
        except Exception:
            p = _two_sided_p_from_z(t)
    else:
        p = _two_sided_p_from_z(t)
    return {
        "t": round(t, 4),
        "df": round(df, 2) if df is not None else None,
        "p": round(p, 6),
    }


def _mann_whitney_u(xs, ys):
    """Mann-Whitney U 检验（双尾），返回 {u, z, p}"""
    n1, n2 = len(xs), len(ys)
    if n1 < 1 or n2 < 1:
        return None
    if _HAS_SCIPY:
        try:
            stat, pval = _scipy_stats.mannwhitneyu(xs, ys, alternative="two-sided")
            return {"u": round(float(stat), 4), "p": round(float(pval), 6)}
        except Exception:
            pass
    combined = [(v, 0) for v in xs] + [(v, 1) for v in ys]
    combined.sort(key=lambda p: p[0])
    N = len(combined)
    ranks = [0.0] * N
    tie_sum = 0
    i = 0
    while i < N:
        j = i
        while j + 1 < N and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        t = j - i + 1
        if t > 1:
            tie_sum += t ** 3 - t
        i = j + 1
    r1 = sum(r for r, pair in zip(ranks, combined) if pair[1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mean_u = n1 * n2 / 2.0
    if N > 1 and tie_sum > 0:
        var_u = (n1 * n2 / 12.0) * ((N + 1) - tie_sum / (N * (N - 1)))
    else:
        var_u = n1 * n2 * (N + 1) / 12.0
    if var_u <= 0:
        return {"u": round(u, 4), "z": 0.0, "p": 1.0}
    z = (u1 - mean_u) / math.sqrt(var_u)
    p = _two_sided_p_from_z(z)
    return {"u": round(u, 4), "z": round(z, 4), "p": round(p, 6)}


def _spearman_corr(xs, ys):
    """Spearman 秩相关，返回 {rho, p}"""
    n = len(xs)
    if n != len(ys) or n < 3:
        return None
    if _HAS_SCIPY:
        try:
            rho, pval = _scipy_stats.spearmanr(xs, ys)
            return {"rho": round(float(rho), 4), "p": round(float(pval), 6)}
        except Exception:
            pass
    rx, ry = _rank(xs), _rank(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    if den == 0:
        return {"rho": 0.0, "p": 1.0}
    rho = num / den
    if abs(rho) >= 1.0:
        return {"rho": round(rho, 4), "p": 0.0}
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    p = _two_sided_p_from_z(t)
    return {"rho": round(rho, 4), "p": round(p, 6)}


def _run_significance(group_per_dim, group_composite):
    """对三组成对比较：high-vs-low / high-vs-medium / medium-vs-low"""
    out = {
        "method": "Welch's t-test + Mann-Whitney U (two-sided)",
        "p_backend": "scipy" if _HAS_SCIPY else "normal_approx",
        "pairs": {},
    }
    pairs = [("high", "low"), ("high", "medium"), ("medium", "low")]
    for g1, g2 in pairs:
        if g1 not in group_per_dim or g2 not in group_per_dim:
            continue
        key = f"{g1}_vs_{g2}"
        out["pairs"][key] = {}
        for d in SCORE_DIMENSIONS:
            xs = group_per_dim[g1].get(d, [])
            ys = group_per_dim[g2].get(d, [])
            if not xs or not ys:
                continue
            out["pairs"][key][d] = {
                "n1": len(xs), "n2": len(ys),
                "mean1": round(_mean(xs), 4),
                "mean2": round(_mean(ys), 4),
                "diff": round(_mean(xs) - _mean(ys), 4),
                "welch_t": _welch_t_test(xs, ys),
                "mann_whitney": _mann_whitney_u(xs, ys),
            }
        cs1, cs2 = group_composite.get(g1, []), group_composite.get(g2, [])
        if cs1 and cs2:
            out["pairs"][key]["composite"] = {
                "n1": len(cs1), "n2": len(cs2),
                "mean1": round(_mean(cs1), 4),
                "mean2": round(_mean(cs2), 4),
                "diff": round(_mean(cs1) - _mean(cs2), 4),
                "welch_t": _welch_t_test(cs1, cs2),
                "mann_whitney": _mann_whitney_u(cs1, cs2),
            }
    return out


def _run_correlations(overall_per_dim):
    """四个维度两两 Spearman 相关矩阵"""
    out = {
        "method": "Spearman",
        "p_backend": "scipy" if _HAS_SCIPY else "normal_approx",
        "matrix": {},
    }
    lengths = [len(overall_per_dim.get(d, [])) for d in SCORE_DIMENSIONS]
    if not lengths or min(lengths) < 3:
        return out
    for d1 in SCORE_DIMENSIONS:
        out["matrix"][d1] = {}
        for d2 in SCORE_DIMENSIONS:
            if d1 == d2:
                out["matrix"][d1][d2] = {"rho": 1.0, "p": 0.0}
            else:
                out["matrix"][d1][d2] = _spearman_corr(overall_per_dim[d1], overall_per_dim[d2])
    return out


def generate_summary(results):
    """生成统计分析：各维度总体 + 分质量组 + 整体分数分布"""
    summary = {
        "total_records": len(results),
        "valid_records": 0,
        "error_records": 0,
        "overall": {},
        "by_quality_group": {},
        "distribution_of_overall_score": {},
    }

    overall_per_dim = defaultdict(list)              # {dim: [values]}
    group_per_dim = defaultdict(lambda: defaultdict(list))  # {group: {dim: [values]}}
    group_composite = defaultdict(list)              # {group: [原始 composite]}
    overall_scores = []  # 每条笑话的四维平均（作为综合得分）

    for r in results:
        scores = r.get("scores")
        if not scores:
            summary["error_records"] += 1
            continue
        # 校验维度完整
        try:
            vals = [float(scores[d]) for d in SCORE_DIMENSIONS]
        except (KeyError, TypeError, ValueError):
            summary["error_records"] += 1
            continue
        summary["valid_records"] += 1
        g = r.get("quality_group", "unknown")
        for d, v in zip(SCORE_DIMENSIONS, vals):
            overall_per_dim[d].append(v)
            group_per_dim[g][d].append(v)
        composite_val = sum(vals) / len(vals)
        overall_scores.append(composite_val)
        group_composite[g].append(composite_val)

    # 整体各维度统计
    for d in SCORE_DIMENSIONS:
        summary["overall"][d] = _stat_block(overall_per_dim[d])
    summary["overall"]["composite"] = _stat_block(overall_scores)

    # 分质量组统计
    for g, dim_map in group_per_dim.items():
        block = {}
        all_vals = []
        for d in SCORE_DIMENSIONS:
            block[d] = _stat_block(dim_map[d])
            all_vals.extend(dim_map[d])
        # 组内综合分
        composite = [sum(dim_map[d][i] for d in SCORE_DIMENSIONS) / 4
                     for i in range(len(dim_map[SCORE_DIMENSIONS[0]]))]
        block["composite"] = _stat_block(composite)
        summary["by_quality_group"][g] = block

    # 整体综合分分布
    summary["distribution_of_overall_score"] = _bucketize(overall_scores)

    # ---------- z-score 标准化综合分（保留原 composite 不变） ----------
    mu = {d: _mean(overall_per_dim[d]) for d in SCORE_DIMENSIONS}
    sigma = {d: (_std_pop(overall_per_dim[d]) or 1.0) for d in SCORE_DIMENSIONS}

    composite_z_overall = []
    group_composite_z = defaultdict(list)
    for r in results:
        scores = r.get("scores")
        if not scores:
            continue
        try:
            vals = {d: float(scores[d]) for d in SCORE_DIMENSIONS}
        except (KeyError, TypeError, ValueError):
            continue
        z_vals = [(vals[d] - mu[d]) / sigma[d] for d in SCORE_DIMENSIONS]
        cz = sum(z_vals) / len(z_vals)
        g = r.get("quality_group", "unknown")
        composite_z_overall.append(cz)
        group_composite_z[g].append(cz)

    summary["overall_zscore"] = {
        "method": "per-dim z-score (population std) then mean across dims",
        "mu": {d: round(mu[d], 4) for d in SCORE_DIMENSIONS},
        "sigma": {d: round(sigma[d], 4) for d in SCORE_DIMENSIONS},
        "composite_z": _stat_block(composite_z_overall),
    }
    summary["distribution_of_composite_z"] = _bucketize_z(composite_z_overall)

    for g, cz_list in group_composite_z.items():
        if g in summary["by_quality_group"]:
            summary["by_quality_group"][g]["composite_z"] = _stat_block(cz_list)

    # ---------- 显著性检验 & 相关性 ----------
    # 原始 composite 的检验（与展示的均值口径一致）
    summary["significance_tests"] = _run_significance(group_per_dim, group_composite)
    # 额外补充：基于 z-composite 的组间检验（仅 composite 行）
    z_sig = _run_significance(defaultdict(lambda: defaultdict(list)), group_composite_z)
    summary["significance_tests_zcomposite"] = {
        "method": z_sig.get("method"),
        "p_backend": z_sig.get("p_backend"),
        "pairs": {
            k: {"composite": v["composite"]} for k, v in z_sig.get("pairs", {}).items()
            if "composite" in v
        },
    }
    summary["correlations"] = _run_correlations(overall_per_dim)

    return summary


def print_summary(summary):
    print(f"\n{'=' * 60}")
    print("数据集评估 - 汇总分析")
    print(f"{'=' * 60}")
    print(f"总记录: {summary['total_records']} | 有效: {summary['valid_records']} | 错误: {summary['error_records']}")

    print("\n--- 各维度总体统计 ---")
    dims = SCORE_DIMENSIONS + ["composite"]
    header = f"  {'维度':<20}" + "".join(f"{k:>10}" for k in ("count", "mean", "max", "min"))
    print(header)
    for d in dims:
        s = summary["overall"].get(d, {})
        print(f"  {d:<20}{str(s.get('count','-')):>10}"
              f"{str(s.get('mean','-')):>10}{str(s.get('max','-')):>10}{str(s.get('min','-')):>10}")

    print("\n--- 分质量组对比（均值） ---")
    groups_order = ["high", "medium", "low"]
    groups_present = [g for g in groups_order if g in summary["by_quality_group"]] \
                     + [g for g in summary["by_quality_group"] if g not in groups_order]
    head = f"  {'维度':<20}" + "".join(f"{g:>12}" for g in groups_present)
    print(head)
    for d in dims:
        row = f"  {d:<20}"
        for g in groups_present:
            s = summary["by_quality_group"][g].get(d, {})
            row += f"{str(s.get('mean', '-')):>12}"
        print(row)

    print("\n--- 综合得分分布 ---")
    for bucket, cnt in summary["distribution_of_overall_score"].items():
        print(f"  {bucket:>10} : {cnt}")

    # ---------- z-score 统计 ----------
    oz = summary.get("overall_zscore")
    if oz:
        print("\n--- z-score 标准化参数（各维度 mu / sigma）---")
        for d in SCORE_DIMENSIONS:
            print(f"  {d:<20} mu={oz['mu'].get(d, '-'):<8} sigma={oz['sigma'].get(d, '-'):<8}")
        cz = oz.get("composite_z", {})
        print(f"  composite_z 统计: count={cz.get('count','-')} mean={cz.get('mean','-')} "
              f"max={cz.get('max','-')} min={cz.get('min','-')}")

        dist_z = summary.get("distribution_of_composite_z", {})
        if dist_z:
            print("\n--- composite_z 分布（标准差区间）---")
            for bucket, cnt in dist_z.items():
                print(f"  {bucket:>12} : {cnt}")

        # 分组 composite_z
        present_groups = [g for g in groups_present if "composite_z" in summary["by_quality_group"].get(g, {})]
        if present_groups:
            print("\n--- 分质量组 composite_z 对比（均值）---")
            head = f"  {'指标':<14}" + "".join(f"{g:>12}" for g in present_groups)
            print(head)
            row = f"  {'composite_z':<14}"
            for g in present_groups:
                s = summary["by_quality_group"][g].get("composite_z", {})
                row += f"{str(s.get('mean', '-')):>12}"
            print(row)

    # ---------- 显著性检验 ----------
    sig = summary.get("significance_tests")
    if sig and sig.get("pairs"):
        print(f"\n--- 显著性检验（{sig.get('method','')}，p_backend={sig.get('p_backend','')}）---")
        print("  每行列出一个维度/综合分在该组对下的 均值差、Welch t 的 p、Mann-Whitney U 的 p")
        dims = SCORE_DIMENSIONS + ["composite"]
        for pair_key, dim_map in sig["pairs"].items():
            print(f"  [{pair_key}]")
            header = f"    {'维度':<18}{'n1':>6}{'n2':>6}{'diff':>10}{'t_p':>12}{'u_p':>12}"
            print(header)
            for d in dims:
                entry = dim_map.get(d)
                if not entry:
                    continue
                tp = entry.get("welch_t", {}) or {}
                up = entry.get("mann_whitney", {}) or {}
                tp_s = f"{tp.get('p', '-')}" if tp else "-"
                up_s = f"{up.get('p', '-')}" if up else "-"
                print(f"    {d:<18}{entry['n1']:>6}{entry['n2']:>6}"
                      f"{entry['diff']:>10}{tp_s:>12}{up_s:>12}")

    # ---------- Spearman 相关矩阵 ----------
    cor = summary.get("correlations")
    if cor and cor.get("matrix"):
        print(f"\n--- 四维 Spearman 相关矩阵（p_backend={cor.get('p_backend','')}）---")
        dims = SCORE_DIMENSIONS
        header = f"    {'':<18}" + "".join(f"{d:>16}" for d in dims)
        print(header)
        for d1 in dims:
            row = f"    {d1:<18}"
            for d2 in dims:
                cell = cor["matrix"].get(d1, {}).get(d2)
                if not cell:
                    row += f"{'-':>16}"
                else:
                    cell_str = f"{cell.get('rho', '-')}(p={cell.get('p', '-')})"
                    row += f"{cell_str:>16}"
            print(row)


# ==================== 入口 ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="使用 GPT-5.5 对实验一笑话数据集进行多维度质量评估")
    parser.add_argument("--lang", type=str, default="en", choices=["en", "es", "zh", "all"],
                        help="语言: en / es / zh / all")
    parser.add_argument("--model", type=str, default="gpt-5.5",
                        help="评估模型名称（默认 gpt-5.5）")
    parser.add_argument("--rpm", type=int, default=600, help="每分钟请求限制")
    parser.add_argument("--concurrency", type=int, default=8, help="并发线程数")
    parser.add_argument("--start", type=int, default=0, help="起始任务索引")
    parser.add_argument("--end", type=int, default=None, help="结束任务索引")
    parser.add_argument("--save-interval", type=int, default=20, help="每 N 条保存一次中间结果")
    args = parser.parse_args()

    langs = ["en", "es", "zh"] if args.lang == "all" else [args.lang]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "output")
    exp_result_dir = os.path.join(base_dir, "exp_result")
    os.makedirs(exp_result_dir, exist_ok=True)

    for lang in langs:
        log_file = os.path.join(exp_result_dir, f"dataset_eval_run_{lang}.log")
        tee = Tee(log_file)
        sys.stdout = tee
        try:
            print(f"日志保存至: {log_file}")
            print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"语言: {lang} | 模型: {args.model} | RPM: {args.rpm} | 并发: {args.concurrency}")
            print(f"数据范围: [{args.start}, {args.end})")
            print("-" * 60)

            output_file = os.path.join(exp_result_dir, f"dataset_eval_results_{lang}.json")

            run(
                output_dir=output_dir,
                output_file=output_file,
                base_dir=base_dir,
                model_name=args.model,
                lang=lang,
                rpm_limit=args.rpm,
                concurrency=args.concurrency,
                start_idx=args.start,
                end_idx=args.end,
                save_interval=args.save_interval,
            )
        finally:
            print("-" * 60)
            print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            sys.stdout = tee.terminal
            tee.close()
