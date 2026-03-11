import json
import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None

MODEL_NAME = "llama-3.1-8b-instant"
MAX_COMPLETION_TOKENS = 50
SAFE_ANALYSIS_DEFAULT = {
    "x_column": "",
    "y_column": "",
    "aggregation": "sum",
    "chart_type": "bar",
}


def _extract_json_object(text):
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _safe_chat(user_content):
    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": user_content[:1200]}],
            temperature=0,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def interpret_query(prompt, columns):
    trimmed_columns = [str(column)[:50] for column in columns[:30]]
    compact_columns = ", ".join(trimmed_columns)

    instruction = (
        "Return JSON only with keys x_column, y_column, aggregation, chart_type. "
        "Use only listed columns. Aggregation must be one of sum, mean, count, max, min. "
        "Chart_type must be one of bar, line, pie, scatter, histogram, treemap, funnel, heatmap, waterfall, gauge. "
        "Choose columns that best match the query meaning and the provided schema labels. "
        f"Columns: {compact_columns}. "
        f"Query: {str(prompt)[:300]}"
    )

    text = _safe_chat(instruction)
    parsed = _extract_json_object(text)

    result = SAFE_ANALYSIS_DEFAULT.copy()
    if isinstance(parsed, dict):
        for key in result:
            value = parsed.get(key)
            if isinstance(value, str):
                result[key] = value.strip()

    if not result["x_column"] and columns:
        result["x_column"] = str(columns[0])
    if not result["y_column"] and columns:
        result["y_column"] = str(columns[-1])

    return result


def generate_insight(prompt, dimension, aggregation, metrics):
    safe_metrics = {
        "row_count": int(metrics.get("row_count", 0)),
        "total": round(float(metrics.get("total", 0.0)), 2),
        "max": round(float(metrics.get("max", 0.0)), 2),
        "avg": round(float(metrics.get("avg", 0.0)), 2),
    }

    instruction = (
        "Give one short business insight in under 25 words. "
        f"Query: {str(prompt)[:220]}. "
        f"Dimension: {str(dimension)[:60]}. "
        f"Aggregation: {str(aggregation)[:20]}. "
        f"Summary: row_count={safe_metrics['row_count']}, total={safe_metrics['total']}, "
        f"max={safe_metrics['max']}, avg={safe_metrics['avg']}."
    )

    text = _safe_chat(instruction).strip()
    if text:
        return text

    return (
        f"{aggregation.title()} analysis across {dimension} shows total {safe_metrics['total']:.2f}, "
        f"peak {safe_metrics['max']:.2f}, average {safe_metrics['avg']:.2f}."
    )
