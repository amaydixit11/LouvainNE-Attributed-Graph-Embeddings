#!/usr/bin/env python3
"""
Exports all benchmark results from JSON to CSV and TXT formats.
Ensures that every single result is captured in a human-readable and machine-processable format.
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent
EXPORT_DIR = REPO_ROOT / "results" / "all_exports"

def write_txt(path: Path, title: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("=" * len(title) + "\n")
        f.write(str(data))

def write_csv(path: Path, headers: List[str], rows: List[List[Any]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def export_json_to_csv_txt(json_path: Path, filename_base: str, flatten_key: str = None):
    if not json_path.exists():
        return

    data = json.loads(json_path.read_text(encoding="utf-8"))

    # Handle list of results vs single result
    results = data if isinstance(data, list) else [data]

    # If there is a flatten key (e.g., 'results' -> 'improved'), extract those
    processed_data = []
    for item in results:
        if flatten_key:
            # For LouvainNE reports, results are often in item['results']['improved']
            # We'll try to extract 'baseline' and 'improved' separately
            if 'results' in item:
                res = item['results']
                for prefix, key in [("baseline", "baseline"), ("improved", "improved")]:
                    if key in res:
                        row = {"dataset": item.get("dataset", "unknown"), "type": prefix}
                        row.update(res[key])
                        processed_data.append(row)
            else:
                processed_data.append(item)
        else:
            processed_data.append(item)

    if not processed_data:
        return

    # Extract headers
    headers = sorted(list(set().union(*(d.keys() for d in processed_data))))
    rows = [[d.get(h, "N/A") for h in headers] for d in processed_data]

    # Write CSV
    write_csv(EXPORT_DIR / f"{filename_base}.csv", headers, rows)

    # Write TXT
    txt_content = ""
    for d in processed_data:
        txt_content += f"--- Entry ---\n"
        for k, v in d.items():
            txt_content += f"{k}: {v}\n"
        txt_content += "\n"
    write_txt(EXPORT_DIR / f"{filename_base}.txt", f"Export of {filename_base}", txt_content)

def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Exporting all results to {EXPORT_DIR}...")

    # 1. Main Benchmark Summary
    export_json_to_csv_txt(
        REPO_ROOT / "results" / "benchmark_summary.json",
        "main_benchmark_summary",
        flatten_key="results"
    )

    # 2. OGB Summary
    export_json_to_csv_txt(
        REPO_ROOT / "results" / "ogb_benchmark_summary.json",
        "ogb_benchmark_summary",
        flatten_key="results"
    )

    # 3. Scalability Results
    export_json_to_csv_txt(
        REPO_ROOT / "results" / "scalability" / "scalability_results.json",
        "scalability_results"
    )

    # 4. Optimization Results
    export_json_to_csv_txt(
        REPO_ROOT / "results" / "optimization" / "optimization_results.json",
        "optimization_results"
    )

    # 5. Individual Dataset Results
    for ds in ["Cora", "CiteSeer", "PubMed", "BlogCatalog"]:
        json_path = REPO_ROOT / "results" / ds / "comparison_results.json"
        export_json_to_csv_txt(
            json_path,
            f"dataset_{ds.lower()}_results",
            flatten_key="results"
        )

    # 6. Copy existing MD reports to TXT for convenience
    md_files = [
        "results/benchmark_summary.md",
        "results/comprehensive_benchmark_report.md",
        "results/ogb_sota_comparison.md"
    ]
    for md_path_str in md_files:
        md_path = REPO_ROOT / md_path_str
        if md_path.exists():
            txt_name = md_path.name.replace(".md", ".txt")
            (EXPORT_DIR / txt_name).write_text(md_path.read_text(), encoding="utf-8")

    print(f"Successfully exported all data to {EXPORT_DIR}")

if __name__ == "__main__":
    main()
