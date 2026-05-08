"""针对 exp1_status_quo_bias_results_en.json 中 qwen3.6-max-preview 模型
的缺失结果进行补全。

遍历所有实验条目，识别缺少 qwen3.6-max-preview 评估结果的条目，
使用与原实验相同的参数和逻辑重新运行评估并补全。
复用 exp1_status_quo_bias.py 的评估逻辑与汇总函数，确保与原实验条件完全一致。
"""

import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 项目根目录加入 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 将 script 目录也加入 sys.path，支持直接 import
sys.path.insert(0, os.path.join(BASE_DIR, "script"))

from utils import get_rate_limiter, Tee
from exp1_status_quo_bias import evaluate_jokes, generate_summary, print_summary


TARGET_MODEL = "qwen3.6-max-preview"
LANG = "en"

_print_lock = threading.Lock()


def _safe_print(msg):
    with _print_lock:
        print(msg)


def is_missing_or_error(eval_result):
    """判断评估结果是否缺失或为 API 异常。"""
    if eval_result is None:
        return True
    if not isinstance(eval_result, dict):
        return True
    raw = eval_result.get("raw_response")
    if isinstance(raw, dict) and "error" in raw:
        return True
    # 额外防御：决策字段全空也视为异常
    decision = eval_result.get("decision", "")
    if not decision and not eval_result.get("final_joke"):
        return True
    return False


def collect_missing_tasks(results):
    """收集所有缺少 TARGET_MODEL 结果的任务：(item_index, item_id, quality_level, news_headline, existing_jokes)"""
    tasks = []
    for idx, item in enumerate(results):
        quality_groups = item.get("quality_groups", {})
        news_headline = item.get("news_headline", "")
        for quality_level, group in quality_groups.items():
            experiments = group.get("experiments", {})
            # 模型结果不存在 或 存在但为错误结果
            eres = experiments.get(TARGET_MODEL)
            if is_missing_or_error(eres):
                existing_jokes = group.get("existing_jokes", [])
                if not news_headline or not existing_jokes:
                    _safe_print(f"  [WARN] {item.get('id')} {quality_level} 缺少输入数据，跳过")
                    continue
                tasks.append((idx, item.get("id", ""), quality_level,
                              news_headline, existing_jokes))
    return tasks


def rerun_one(task):
    item_idx, item_id, quality_level, news_headline, existing_jokes = task
    try:
        result = evaluate_jokes(news_headline, existing_jokes,
                                model_name=TARGET_MODEL, lang=LANG)
        _safe_print(f"  [OK] {item_id} / {quality_level}: decision={result.get('decision')}")
        return (item_idx, quality_level, result, None)
    except Exception as e:
        _safe_print(f"  [FAIL] {item_id} / {quality_level}: {e}")
        err = {
            "decision": "",
            "selected_joke_number": None,
            "reason": "",
            "final_joke": "",
            "chose_existing": False,
            "chose_to_create": False,
            "raw_response": {"error": f"重跑异常: {e}"},
        }
        return (item_idx, quality_level, err, str(e))


def main(input_file, rpm=600, max_workers=8, save_interval=50):
    if not os.path.exists(input_file):
        print(f"文件不存在: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "results" not in data:
        print("结果文件格式异常：缺少 results 字段")
        return

    results = data["results"]
    print(f"加载 results: {len(results)} 条")

    # 初始化速率限制器（DashScope 通道）
    get_rate_limiter("dashscope", rpm=rpm)
    print(f"DashScope 通道 RPM={rpm}")

    # 收集待补全任务
    tasks = collect_missing_tasks(results)
    print(f"待补全 {TARGET_MODEL} 条目: {len(tasks)} 个")
    if not tasks:
        print("无需要重跑的条目。")
        return

    # 并发执行
    start_time = time.time()
    done = 0
    fail_cnt = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(rerun_one, t) for t in tasks]
        for fut in as_completed(futures):
            item_idx, quality_level, new_eval, err = fut.result()
            # 写回到原 results 对应位置
            try:
                results[item_idx]["quality_groups"][quality_level]["experiments"][TARGET_MODEL] = new_eval
            except Exception as e:
                _safe_print(f"  [写入失败] idx={item_idx} {quality_level}: {e}")
            if err:
                fail_cnt += 1
            done += 1
            if done % 20 == 0 or done == len(tasks):
                elapsed = time.time() - start_time
                _safe_print(f"  进度 {done}/{len(tasks)}  失败={fail_cnt}  耗时 {elapsed/60:.1f}min")
            # 定期落盘
            if done % save_interval == 0:
                with open(input_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                _safe_print(f"  [定期保存] 已写入中间结果")

    # 确保 models 列表包含目标模型
    if "models" in data and TARGET_MODEL not in data["models"]:
        data["models"].append(TARGET_MODEL)
        print(f"已将 {TARGET_MODEL} 添加到 models 列表")

    # 重新生成 summary
    print("\n重新生成 summary ...")
    new_summary = generate_summary(results)
    data["summary"] = new_summary

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已写回: {input_file}")

    # 输出最终摘要
    print_summary(new_summary)
    print(f"\n本次重跑完成，总耗时 {(time.time()-start_time)/60:.1f} min，失败 {fail_cnt} 条")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "exp_result", "exp1_status_quo_bias_results_en.json"))
    parser.add_argument("--rpm", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file = os.path.join(base_dir, "exp_result", "fix_qwen36_missing_en.log")
    tee = Tee(log_file)
    sys.stdout = tee
    try:
        print(f"=== 补全 qwen3.6-max-preview 缺失条目 ===")
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输入文件: {args.input}")
        print(f"RPM={args.rpm}  workers={args.workers}")
        main(args.input, rpm=args.rpm,
             max_workers=args.workers, save_interval=args.save_interval)
    finally:
        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout = tee.terminal
        tee.close()
