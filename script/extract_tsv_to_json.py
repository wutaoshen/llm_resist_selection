"""
从 raw_data 目录下的 TSV 文件提取数据并按语言分别保存为 JSON 文件。
3种语言，包括验证集和测试集。
"""
import csv
import json
import os

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "raw_data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# 六个源文件配置
SOURCE_FILES = {
    "en": [
        os.path.join(RAW_DATA_DIR, "mwahaha_dev_tasks-ab_v2", "task-a-en.tsv"),
        os.path.join(RAW_DATA_DIR, "mwahaha_test", "task-a-en.tsv"),
    ],
    "zh": [
        os.path.join(RAW_DATA_DIR, "mwahaha_dev_tasks-ab_v2", "task-a-zh.tsv"),
        os.path.join(RAW_DATA_DIR, "mwahaha_test", "task-a-zh.tsv"),
    ],
    "es": [
        os.path.join(RAW_DATA_DIR, "mwahaha_dev_tasks-ab_v2", "task-a-es.tsv"),
        os.path.join(RAW_DATA_DIR, "mwahaha_test", "task-a-es.tsv"),
    ],
}

OUTPUT_FILES = {
    "en": os.path.join(OUTPUT_DIR, "headlines_en.json"),
    "zh": os.path.join(OUTPUT_DIR, "headlines_zh.json"),
    "es": os.path.join(OUTPUT_DIR, "headlines_es.json"),
}


def extract_language(lang: str, file_paths: list[str]) -> list[dict]:
    """提取指定语言的所有数据，过滤 headline 为 '-' 的行。"""
    results = []
    seen_ids = set()

    for file_path in file_paths:
        with open(file_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                headline = row.get("headline", "").strip()
                item_id = row.get("id", "").strip()
                if headline == "-" or not headline:
                    continue
                if item_id in seen_ids:
                    print(f"  [警告] 重复 id: {item_id}，跳过")
                    continue
                seen_ids.add(item_id)
                results.append({"id": item_id, "news_headline": headline})

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for lang, file_paths in SOURCE_FILES.items():
        print(f"\n处理语言: {lang}")
        for fp in file_paths:
            print(f"  读取: {fp}")

        items = extract_language(lang, file_paths)
        print(f"  共提取 {len(items)} 条有效数据")

        output_path = OUTPUT_FILES[lang]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  已保存到: {output_path}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()
