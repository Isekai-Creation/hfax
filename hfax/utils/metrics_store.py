"""Persistent storage and reporting helpers for benchmark metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
DEFAULT_DB_PATH = DEFAULT_DIR / "metrics.json"
DEFAULT_README_PATH = DEFAULT_DIR / "README.md"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_records(db_path: Path | None = None) -> list[dict[str, Any]]:
    db_path = db_path or DEFAULT_DB_PATH
    if not db_path.exists():
        return []
    with db_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_records(
    entries: Iterable[dict[str, Any]],
    *,
    run_meta: dict[str, Any],
    db_path: Path | None = None,
    readme_path: Path | None = None,
) -> None:
    db_path = db_path or DEFAULT_DB_PATH
    readme_path = readme_path or DEFAULT_README_PATH
    _ensure_dir(db_path.parent)

    records = load_records(db_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    for entry in entries:
        record = {
            "timestamp": timestamp,
            "script": run_meta.get("script"),
            "run_type": run_meta.get("run_type"),
            "tpu_type": run_meta.get("tpu_type"),
            "quant_method": run_meta.get("quant_method"),
            "batch_size": run_meta.get("batch_size"),
            "benchmark_runs": run_meta.get("benchmark_runs"),
            "warmup_runs": run_meta.get("warmup_runs"),
            "phase": entry.get("phase"),
            "mode": entry.get("label"),
            "token_count": entry.get("token_count"),
            "runs": entry.get("runs"),
            "metrics": entry.get("metrics_summary"),
            "notes": entry.get("notes"),
        }
        records.append(record)

    with db_path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)

    generate_readme(records, readme_path)
    generate_html(records, readme_path.with_suffix(".html"))


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def generate_readme(records: list[dict[str, Any]], output_path: Path | None = None) -> None:
    output_path = output_path or DEFAULT_README_PATH
    _ensure_dir(output_path.parent)

    header_lines = [
        "# Benchmark Results",
        "",
        "This file is auto-generated from `benchmarks/metrics.json`.",
        "",
    ]

    if not records:
        header_lines.append("No benchmark data recorded yet.\n")
        output_path.write_text("\n".join(header_lines), encoding="utf-8")
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        run_type = record.get("run_type", "unknown")
        grouped[run_type].append(record)

    lines = list(header_lines)
    
    columns = [
        "Date (UTC)", "Script", "Phase", "Mode", "TPU", "Token Count", 
        "Batch Size", "Runs", "Avg Total s", "Avg First Token s", 
        "Avg Post-First s", "Avg Tokens/s", "Avg Decode Tokens/s", 
        "Avg Pre-First Tokens/s", "Notes"
    ]
    
    table_header = "| " + " | ".join(columns) + " |"
    table_divider = "| " + " | ".join(["---"] * len(columns)) + " |"

    for run_type in sorted(grouped):
        lines.append(f"## {run_type.capitalize()} Runs")
        lines.append("")
        lines.append(table_header)
        lines.append(table_divider)

        for record in grouped[run_type]:
            metrics = record.get("metrics", {}) or {}
            row_data = [
                record.get('timestamp', ''),
                record.get('script', ''),
                record.get('phase', ''),
                record.get('mode', ''),
                record.get('tpu_type', ''),
                record.get('token_count', ''),
                record.get('batch_size', ''),
                record.get('runs', ''),
                _format_value(metrics.get('avg_total')),
                _format_value(metrics.get('avg_first')),
                _format_value(metrics.get('avg_post')),
                _format_value(metrics.get('avg_tps')),
                _format_value(metrics.get('avg_decode_tps')),
                _format_value(metrics.get('avg_prefirst_tps')),
                record.get('notes', ''),
            ]
            lines.append("| " + " | ".join(map(str, row_data)) + " |")
        lines.append("")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")

def generate_html(records: list[dict[str, Any]], output_path: Path) -> None:
    html_template = '''
<!DOCTYPE html>
<html>
<head>
<title>Benchmark Results</title>
<style>
  body {{ font-family: sans-serif; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; }}
  th {{ background-color: #f2f2f2; cursor: pointer; }}
  tr:nth-child(even) {{ background-color: #f9f9f9; }}
</style>
</head>
<body>
<h1>Benchmark Results</h1>
{tables}
<script>
function sortTable(table, col, isNumeric) {{  
  let rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
  switching = true;
  dir = "asc";
  while (switching) {{
    switching = false;
    rows = table.rows;
    for (i = 1; i < (rows.length - 1); i++) {{
      shouldSwitch = false;
      x = rows[i].getElementsByTagName("TD")[col];
      y = rows[i + 1].getElementsByTagName("TD")[col];
      let xContent = isNumeric ? parseFloat(x.innerHTML) || 0 : x.innerHTML.toLowerCase();
      let yContent = isNumeric ? parseFloat(y.innerHTML) || 0 : y.innerHTML.toLowerCase();
      if (dir == "asc") {{
        if (xContent > yContent) {{
          shouldSwitch = true;
          break;
        }}
      }} else if (dir == "desc") {{
        if (xContent < yContent) {{
          shouldSwitch = true;
          break;
        }}
      }}
    }}
    if (shouldSwitch) {{
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true;
      switchcount++;
    }} else {{
      if (switchcount == 0 && dir == "asc") {{
        dir = "desc";
        switching = true;
      }}
    }}
  }}
}}

document.querySelectorAll('th').forEach((th, colIndex) => {{
    th.addEventListener('click', () => {{
        const table = th.closest('table');
        const isNumeric = !isNaN(parseFloat(table.rows[1].cells[colIndex].innerHTML));
        sortTable(table, colIndex, isNumeric);
    }});
}});
</script>
</body>
</html>
'''
    
    if not records:
        output_path.write_text(html_template.format(tables="<p>No benchmark data recorded yet.</p>"), encoding="utf-8")
        return

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        run_type = record.get("run_type", "unknown")
        grouped[run_type].append(record)

    tables_html = ""
    columns = [
        "Date (UTC)", "Script", "Phase", "Mode", "TPU", "Token Count", 
        "Batch Size", "Runs", "Avg Total s", "Avg First Token s", 
        "Avg Post-First s", "Avg Tokens/s", "Avg Decode Tokens/s", 
        "Avg Pre-First Tokens/s", "Notes"
    ]

    for run_type in sorted(grouped):
        tables_html += f"<h2>{run_type.capitalize()} Runs</h2>"
        tables_html += "<table><thead><tr>"
        for col in columns:
            tables_html += f"<th>{col}</th>"
        tables_html += "</tr></thead><tbody>"

        for record in grouped[run_type]:
            metrics = record.get("metrics", {}) or {}
            row_data = [
                record.get('timestamp', ''),
                record.get('script', ''),
                record.get('phase', ''),
                record.get('mode', ''),
                record.get('tpu_type', ''),
                record.get('token_count', ''),
                record.get('batch_size', ''),
                record.get('runs', ''),
                _format_value(metrics.get('avg_total')),
                _format_value(metrics.get('avg_first')),
                _format_value(metrics.get('avg_post')),
                _format_value(metrics.get('avg_tps')),
                _format_value(metrics.get('avg_decode_tps')),
                _format_value(metrics.get('avg_prefirst_tps')),
                record.get('notes', ''),
            ]
            tables_html += "<tr>"
            for cell in row_data:
                tables_html += f"<td>{cell}</td>"
            tables_html += "</tr>"
        tables_html += "</tbody></table>"

    output_path.write_text(html_template.format(tables=tables_html), encoding="utf-8")