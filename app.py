import re
import sqlite3
import uuid
from io import BytesIO

import pandas as pd
import streamlit as st

from charts import generate_chart
from llm_engine import generate_insight, interpret_query


st.set_page_config(page_title="AI BI Dashboard", layout="wide")


st.markdown(
    """
<style>
.stApp {
    background:
        radial-gradient(circle at top left, rgba(32, 93, 160, 0.10), transparent 28%),
        linear-gradient(180deg, #eef3f8 0%, #f7f9fc 100%);
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

.section-card {
    background: linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%);
    padding: 22px;
    border-radius: 18px;
    border: 1px solid rgba(24, 53, 88, 0.08);
    box-shadow: 0 18px 40px rgba(20, 41, 61, 0.08);
    margin-bottom: 18px;
}

.hero-card {
    background: linear-gradient(120deg, #153b66 0%, #1f6aa5 100%);
    color: white;
    padding: 28px;
    border-radius: 22px;
    margin-bottom: 18px;
    box-shadow: 0 20px 44px rgba(21, 59, 102, 0.22);
}

.hero-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}

.hero-subtitle {
    font-size: 0.98rem;
    opacity: 0.9;
    margin: 0;
}

.meta-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}

.meta-tile {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 16px;
    padding: 14px 16px;
}

.meta-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.78;
}

.meta-value {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 6px;
}

.section-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #17324d;
    margin-bottom: 12px;
}

.kpi-note {
    font-size: 0.85rem;
    color: #5a6f84;
}
</style>
""",
    unsafe_allow_html=True,
)


if "history" not in st.session_state:
    st.session_state.history = []
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None


def normalize_identifier(value):
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(value).strip())


def quote_identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


@st.cache_resource(show_spinner=False)
def prepare_database(file_bytes, file_name):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    table_name = f"dataset_{uuid.uuid4().hex[:8]}"
    file_buffer = BytesIO(file_bytes)

    preview_df = None
    row_count = 0
    numeric_columns = []
    categorical_columns = []
    date_columns = []
    columns = []

    for chunk_index, chunk in enumerate(pd.read_csv(file_buffer, chunksize=50000)):
        chunk.columns = [normalize_identifier(column) for column in chunk.columns]
        detected_dates = []
        for column in chunk.columns:
            if column in numeric_columns:
                continue
            if chunk[column].dtype == "object" or "date" in column.lower() or "time" in column.lower():
                parsed = pd.to_datetime(chunk[column], errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    chunk[column] = parsed.dt.strftime("%Y-%m-%d")
                    detected_dates.append(column)
        if chunk_index == 0:
            preview_df = chunk.head(5).copy()
            columns = list(chunk.columns)
            numeric_columns = list(chunk.select_dtypes(include="number").columns)
            categorical_columns = [column for column in chunk.columns if column not in numeric_columns]
            date_columns = detected_dates[:]
        else:
            for column in detected_dates:
                if column not in date_columns:
                    date_columns.append(column)

        chunk.to_sql(table_name, connection, if_exists="append", index=False)
        row_count += len(chunk)

    if preview_df is None:
        preview_df = pd.DataFrame()

    schema = {
        "table_name": table_name,
        "dataset_name": file_name,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "date_columns": date_columns,
        "row_count": row_count,
        "column_count": len(columns),
    }
    return connection, preview_df, schema


def resolve_column(name, columns):
    if not name:
        return None

    cleaned_name = re.sub(r"\s*\((numeric|categorical)\)\s*$", "", str(name).strip(), flags=re.IGNORECASE)
    exact_map = {column.lower(): column for column in columns}
    lowered = cleaned_name.lower()
    if lowered in exact_map:
        return exact_map[lowered]

    normalized = lowered.replace("_", " ").replace("-", " ")
    for column in columns:
        candidate = column.lower().replace("_", " ").replace("-", " ")
        if candidate == normalized:
            return column

    for column in columns:
        candidate = column.lower().replace("_", " ").replace("-", " ")
        if normalized in candidate or candidate in normalized:
            return column

    return None


def choose_default_x(schema):
    if schema["categorical_columns"]:
        return schema["categorical_columns"][0]
    return schema["columns"][0]


def choose_default_y(schema):
    if schema["numeric_columns"]:
        return schema["numeric_columns"][0]
    return schema["columns"][0]


def tokenize_text(value):
    return [token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token]


def expand_tokens(tokens):
    synonym_groups = {
        "revenue": {"sales", "income", "earnings"},
        "sales": {"revenue", "income"},
        "cost": {"spend", "expense", "price"},
        "profit": {"margin", "gain"},
        "customer": {"client", "buyer", "user"},
        "users": {"customers", "clients"},
        "date": {"day", "month", "year", "time"},
        "time": {"date", "timeline", "trend"},
        "category": {"type", "group", "segment", "class"},
        "segment": {"group", "category"},
        "region": {"country", "state", "city", "area", "location"},
        "location": {"region", "city", "country", "state"},
        "quantity": {"units", "count", "volume"},
        "count": {"quantity", "number", "total"},
        "score": {"rating", "rank", "points"},
        "amount": {"value", "total"},
        "value": {"amount", "metric"},
    }
    expanded = set(tokens)
    for token in list(tokens):
        expanded.update(synonym_groups.get(token, set()))
    return expanded


def get_column_tokens(column_name):
    raw_tokens = tokenize_text(column_name)
    singular_tokens = {token[:-1] if token.endswith("s") and len(token) > 3 else token for token in raw_tokens}
    return expand_tokens(set(raw_tokens) | singular_tokens)


def find_exact_token_column(columns, target_tokens):
    target_set = {token.lower() for token in target_tokens if token}
    for column in columns:
        column_tokens = set(tokenize_text(column))
        if column_tokens.intersection(target_set):
            return column
    return None


def score_column_match(prompt_tokens, column_name):
    column_tokens = get_column_tokens(column_name)
    overlap = prompt_tokens.intersection(column_tokens)
    phrase = " ".join(tokenize_text(column_name))
    score = len(overlap) * 3

    if phrase and phrase in " ".join(sorted(prompt_tokens)):
        score += 4

    for token in column_tokens:
        if token in prompt_tokens:
            score += 1

    return score


def pick_best_column(candidates, prompt_tokens, fallback=None):
    scored = [(score_column_match(prompt_tokens, column), column) for column in candidates]
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1]
    return fallback


def infer_columns_from_prompt(prompt, schema):
    prompt_tokens = expand_tokens(set(tokenize_text(prompt)))
    numeric_columns = schema["numeric_columns"]
    categorical_columns = schema["categorical_columns"]
    prompt_text = str(prompt).lower()
    split_markers = [" by ", " per ", " across ", " for each "]

    group_hint = ""
    metric_hint = prompt_text
    for marker in split_markers:
        if marker in prompt_text:
            metric_hint, group_hint = prompt_text.split(marker, 1)
            break

    metric_tokens = expand_tokens(set(tokenize_text(metric_hint)))
    group_tokens = expand_tokens(set(tokenize_text(group_hint)))

    explicit_metric_keywords = {
        "roi": ["roi", "return", "investment"],
        "revenue": ["revenue", "sales", "income"],
        "profit": ["profit", "margin", "gain"],
        "cost": ["cost", "expense", "spend", "price"],
        "click": ["click", "clicks"],
        "lead": ["lead", "leads"],
        "conversion": ["conversion", "conversions"],
        "impression": ["impression", "impressions"],
        "engagement": ["engagement", "score", "rating"],
        "duration": ["duration", "time", "days", "hours"],
    }

    inferred_y = None
    for keyword, aliases in explicit_metric_keywords.items():
        if keyword in prompt_text:
            inferred_y = find_exact_token_column(numeric_columns, aliases)
            if inferred_y:
                break

    if inferred_y is None:
        inferred_y = pick_best_column(numeric_columns, metric_tokens, fallback=None)

    inferred_x = pick_best_column(categorical_columns, group_tokens, fallback=None)

    if inferred_y is None:
        inferred_y = pick_best_column(numeric_columns, prompt_tokens, fallback=None)

    if inferred_x is None:
        inferred_x = pick_best_column(categorical_columns, prompt_tokens, fallback=None)

    if inferred_x is None and "trend" in prompt_tokens:
        inferred_x = pick_best_column(categorical_columns, {"date", "time", "month", "year"}, fallback=None)

    return inferred_x, inferred_y


def infer_aggregation_from_prompt(prompt):
    prompt_text = str(prompt).lower()
    if any(token in prompt_text for token in ["average", "avg", "mean"]):
        return "mean"
    if any(token in prompt_text for token in ["count", "number of", "how many"]):
        return "count"
    if any(token in prompt_text for token in ["highest", "top", "maximum", "max", "best"]):
        return "max"
    if any(token in prompt_text for token in ["lowest", "minimum", "min", "worst"]):
        return "min"
    return "sum"


def infer_limit(prompt):
    match = re.search(r"\b(top|bottom)\s+(\d+)\b", str(prompt).lower())
    if match:
        direction = "desc" if match.group(1) == "top" else "asc"
        count = max(1, min(int(match.group(2)), 25))
        return count, direction

    prompt_text = str(prompt).lower()
    if "top" in prompt_text or "highest" in prompt_text or "best" in prompt_text:
        return 10, "desc"
    if "bottom" in prompt_text or "lowest" in prompt_text or "worst" in prompt_text:
        return 10, "asc"
    return None, "desc"


def infer_date_grain(prompt, x_column, schema):
    if x_column not in schema["date_columns"]:
        return None

    prompt_text = str(prompt).lower()
    if "year" in prompt_text:
        return "year"
    if "month" in prompt_text:
        return "month"
    if "day" in prompt_text or "daily" in prompt_text:
        return "day"
    if any(token in prompt_text for token in ["trend", "over time", "timeline"]):
        return "month"
    return None


def is_follow_up_prompt(prompt):
    prompt_text = str(prompt).lower().strip()
    markers = ["now", "also", "instead", "same", "that", "those", "switch", "change", "make it", "show it"]
    return any(marker in prompt_text for marker in markers)


def coerce_analysis(prompt, analysis, schema):
    columns = schema["columns"]
    numeric_columns = schema["numeric_columns"]
    inferred_x, inferred_y = infer_columns_from_prompt(prompt, schema)
    llm_x = resolve_column(analysis.get("x_column"), columns)
    llm_y = resolve_column(analysis.get("y_column"), columns)

    safe_x = inferred_x or llm_x or choose_default_x(schema)
    safe_y = inferred_y or llm_y
    aggregation = infer_aggregation_from_prompt(prompt)
    llm_aggregation = str(analysis.get("aggregation", "")).lower()
    if llm_aggregation in {"sum", "mean", "count", "max", "min"} and aggregation == "sum":
        aggregation = llm_aggregation

    chart_type = str(analysis.get("chart_type", "bar")).lower()
    if chart_type not in {"bar", "line", "pie", "scatter", "histogram", "treemap", "funnel", "heatmap", "waterfall", "gauge"}:
        chart_type = "bar"

    if aggregation in {"sum", "mean", "max", "min"}:
        if safe_y not in numeric_columns:
            safe_y = choose_default_y(schema)
    elif safe_y is None:
        safe_y = choose_default_y(schema)

    return {
        "x_column": safe_x,
        "y_column": safe_y,
        "aggregation": aggregation,
        "chart_type": chart_type,
        "source_x": "prompt" if inferred_x else "llm" if llm_x else "default",
        "source_y": "prompt" if inferred_y else "llm" if llm_y else "default",
    }


def merge_follow_up_analysis(prompt, analysis, schema):
    previous = st.session_state.get("last_analysis")
    if not previous or not is_follow_up_prompt(prompt):
        return analysis

    merged = dict(analysis)
    prompt_text = str(prompt).lower()

    if analysis["source_x"] == "default" and previous.get("x_column") in schema["columns"]:
        merged["x_column"] = previous["x_column"]
        merged["source_x"] = "follow_up"

    if analysis["source_y"] == "default" and previous.get("y_column") in schema["columns"]:
        merged["y_column"] = previous["y_column"]
        merged["source_y"] = "follow_up"

    if analysis["aggregation"] == "sum" and previous.get("aggregation") and not any(
        token in prompt_text for token in ["sum", "total", "average", "avg", "mean", "count", "highest", "lowest", "min", "max"]
    ):
        merged["aggregation"] = previous["aggregation"]

    if analysis["chart_type"] == "bar" and previous.get("chart_type") and not any(
        token in prompt_text for token in ["bar", "line", "pie", "scatter", "histogram", "treemap", "funnel", "heatmap", "waterfall", "gauge"]
    ):
        merged["chart_type"] = previous["chart_type"]

    return merged


def fetch_filter_options(connection, table_name, column_name):
    query = f"""
        SELECT {quote_identifier(column_name)} AS value
        FROM {quote_identifier(table_name)}
        WHERE {quote_identifier(column_name)} IS NOT NULL
        GROUP BY {quote_identifier(column_name)}
        ORDER BY COUNT(*) DESC, {quote_identifier(column_name)}
        LIMIT 25
    """
    values = pd.read_sql_query(query, connection)["value"].tolist()
    return values


def build_where_clause(filters):
    if not filters:
        return "", []

    clauses = []
    params = []
    for column_name, selected_values in filters.items():
        if not selected_values:
            continue
        placeholders = ", ".join(["?"] * len(selected_values))
        clauses.append(f"{quote_identifier(column_name)} IN ({placeholders})")
        params.extend(selected_values)

    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), params


def query_dataset_preview(connection, table_name, where_clause, params):
    query = f"SELECT * FROM {quote_identifier(table_name)}{where_clause} LIMIT 5"
    return pd.read_sql_query(query, connection, params=params)


def query_filtered_row_count(connection, table_name, where_clause, params):
    query = f"SELECT COUNT(*) AS total_rows FROM {quote_identifier(table_name)}{where_clause}"
    return int(pd.read_sql_query(query, connection, params=params).iloc[0]["total_rows"])


def aggregate_data(connection, schema, x_column, y_column, aggregation, filters, date_grain=None, result_limit=None, sort_direction="desc"):
    table_name = schema["table_name"]
    where_clause, params = build_where_clause(filters)
    safe_x = resolve_column(x_column, schema["columns"]) or choose_default_x(schema)
    safe_y = resolve_column(y_column, schema["columns"]) or choose_default_y(schema)
    group_expression = quote_identifier(safe_x)
    order_direction = "DESC" if sort_direction == "desc" else "ASC"
    x_alias = safe_x

    if date_grain and safe_x in schema["date_columns"]:
        if date_grain == "year":
            group_expression = f"strftime('%Y', {quote_identifier(safe_x)})"
            x_alias = f"{safe_x}_year"
        elif date_grain == "month":
            group_expression = f"strftime('%Y-%m', {quote_identifier(safe_x)})"
            x_alias = f"{safe_x}_month"
        elif date_grain == "day":
            group_expression = f"strftime('%Y-%m-%d', {quote_identifier(safe_x)})"
            x_alias = f"{safe_x}_day"

    limit_clause = f" LIMIT {result_limit}" if result_limit else ""

    try:
        if aggregation == "count":
            query = f"""
                SELECT {group_expression} AS {quote_identifier(x_alias)},
                       COUNT(*) AS "Count"
                FROM {quote_identifier(table_name)}
                {where_clause}
                GROUP BY {group_expression}
                ORDER BY "Count" {order_direction}
                {limit_clause}
            """
            return pd.read_sql_query(query, connection, params=params), "Count", x_alias, safe_y

        if safe_y in schema["numeric_columns"]:
            sql_agg = {"sum": "SUM", "mean": "AVG", "max": "MAX", "min": "MIN"}.get(aggregation, "SUM")
            query = f"""
                SELECT {group_expression} AS {quote_identifier(x_alias)},
                       {sql_agg}(CAST({quote_identifier(safe_y)} AS REAL)) AS {quote_identifier(safe_y)}
                FROM {quote_identifier(table_name)}
                {where_clause}
                GROUP BY {group_expression}
                ORDER BY {quote_identifier(safe_y)} {order_direction}
                {limit_clause}
            """
            return pd.read_sql_query(query, connection, params=params), safe_y, x_alias, safe_y
    except Exception:
        pass

    fallback_x = choose_default_x(schema)
    fallback_y = choose_default_y(schema)
    try:
        query = f"""
            SELECT {quote_identifier(fallback_x)} AS {quote_identifier(fallback_x)},
                   SUM(CAST({quote_identifier(fallback_y)} AS REAL)) AS {quote_identifier(fallback_y)}
            FROM {quote_identifier(table_name)}
            {where_clause}
            GROUP BY {quote_identifier(fallback_x)}
            ORDER BY {quote_identifier(fallback_y)} DESC
        """
        return pd.read_sql_query(query, connection, params=params), fallback_y, fallback_x, fallback_y
    except Exception:
        fallback = pd.DataFrame({fallback_x: ["All Data"], "Count": [query_filtered_row_count(connection, table_name, where_clause, params)]})
        return fallback, "Count", fallback_x, fallback_y


def build_summary_metrics(dataframe, value_column):
    numeric_series = pd.to_numeric(dataframe[value_column], errors="coerce").fillna(0)
    return {
        "row_count": int(len(dataframe)),
        "total": float(numeric_series.sum()),
        "max": float(numeric_series.max()) if not numeric_series.empty else 0.0,
        "avg": float(numeric_series.mean()) if not numeric_series.empty else 0.0,
    }


def format_number(value):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.2f}"


def infer_auto_chart(x_column, aggregation, grouped_data):
    lowered_name = x_column.lower()
    unique_groups = grouped_data[x_column].nunique(dropna=False) if x_column in grouped_data.columns else 0

    if "date" in lowered_name or "time" in lowered_name:
        return "line"
    if aggregation == "count" and unique_groups <= 6:
        return "pie"
    if unique_groups <= 8:
        return "bar"
    if unique_groups <= 20:
        return "line"
    return "bar"


def build_supporting_views(grouped_data, x_column, value_column):
    ranked = grouped_data.sort_values(value_column, ascending=False).head(10).reset_index(drop=True)
    try:
        trend_ready = grouped_data.sort_values(x_column).reset_index(drop=True)
    except Exception:
        trend_ready = grouped_data.copy()
    return ranked, trend_ready


def build_resolution_note(analysis, prompt, schema):
    inferred_x, inferred_y = infer_columns_from_prompt(prompt, schema)
    if analysis["source_x"] == "follow_up" or analysis["source_y"] == "follow_up":
        return "This follow-up query reused the previous dashboard context and updated only the requested change."
    if analysis["source_x"] == "default" or analysis["source_y"] == "default":
        return "The query was partially ambiguous, so safe default fields were used."
    if analysis["source_x"] == "llm" or analysis["source_y"] == "llm":
        return "The query relied on model interpretation for part of the field mapping."
    if inferred_x and inferred_y:
        return "The query fields were matched directly from your wording and dataset schema."
    return "The query was resolved using a mix of schema matching and safe fallbacks."


def compute_confidence(analysis):
    source_weights = {"prompt": 0.9, "follow_up": 0.85, "llm": 0.7, "default": 0.45}
    x_score = source_weights.get(analysis["source_x"], 0.45)
    y_score = source_weights.get(analysis["source_y"], 0.45)
    return round((x_score + y_score) / 2, 2)


def build_query_interpretation(analysis, value_column, auto_chart):
    return pd.DataFrame(
        [
            {"Component": "Grouping Field", "Selection": analysis["x_column"]},
            {"Component": "Metric Field", "Selection": value_column},
            {"Component": "Aggregation", "Selection": analysis["aggregation"]},
            {"Component": "Chart Type", "Selection": auto_chart if auto_chart else analysis["chart_type"]},
            {"Component": "Grouping Source", "Selection": analysis["source_x"]},
            {"Component": "Metric Source", "Selection": analysis["source_y"]},
        ]
    )


def explain_chart_choice(chart_type, x_column, grouped_data):
    unique_groups = grouped_data[x_column].nunique(dropna=False) if x_column in grouped_data.columns else 0
    if chart_type == "line":
        return "Line chart was chosen because the grouping field behaves like a trend or time sequence."
    if chart_type == "pie":
        return "Pie chart was chosen because the result has a small number of categories suitable for share comparison."
    if chart_type == "bar":
        return f"Bar chart was chosen because it compares {unique_groups} grouped categories clearly."
    if chart_type == "scatter":
        return "Scatter chart was chosen for point-based comparison between fields."
    if chart_type == "histogram":
        return "Histogram was chosen to show the distribution of a numeric measure."
    return "The chart was selected as the safest visual for the resolved query."


st.sidebar.title("Dashboard Controls")

uploaded = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV file to load it into the analytics database and start exploring.")
    st.stop()

file_bytes = uploaded.getvalue()
connection, preview_df, schema = prepare_database(file_bytes, uploaded.name)

st.markdown(
    f"""
<div class="hero-card">
    <div class="hero-title">Power BI Style Analytics Report</div>
    <p class="hero-subtitle">Upload a dataset, store it in a SQL-backed engine, and generate interactive dashboards from natural language queries.</p>
    <div class="meta-strip">
        <div class="meta-tile">
            <div class="meta-label">Dataset</div>
            <div class="meta-value">{schema["dataset_name"]}</div>
        </div>
        <div class="meta-tile">
            <div class="meta-label">Rows</div>
            <div class="meta-value">{schema["row_count"]:,}</div>
        </div>
        <div class="meta-tile">
            <div class="meta-label">Columns</div>
            <div class="meta-value">{schema["column_count"]:,}</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("### Filters")
filterable_columns = []
for column in schema["categorical_columns"]:
    options = fetch_filter_options(connection, schema["table_name"], column)
    if 1 < len(options) <= 25:
        filterable_columns.append((column, options))

active_filters = {}
for filter_column, options in filterable_columns[:3]:
    selected_values = st.sidebar.multiselect(filter_column, options)
    if selected_values:
        active_filters[filter_column] = selected_values

st.sidebar.markdown("### Chart Options")
chart_override = st.sidebar.selectbox(
    "Chart Type",
    ["Auto", "Bar", "Line", "Pie", "Scatter", "Histogram", "Treemap", "Funnel", "Heatmap", "Waterfall", "Gauge"],
)
show_architecture = st.sidebar.toggle("Show Architecture", value=True)

st.markdown('<div class="section-title">Quick Analysis Templates</div>', unsafe_allow_html=True)
col_a, col_b, col_c = st.columns(3, gap="small")

if col_a.button("Revenue by Category"):
    prompt = f"Show {choose_default_y(schema)} by {choose_default_x(schema)}"
elif col_b.button("Count by Category"):
    prompt = f"Show count by {choose_default_x(schema)}"
elif col_c.button("Average Metric"):
    prompt = f"Show average {choose_default_y(schema)} by {choose_default_x(schema)}"
else:
    prompt = st.text_input("Ask a question")

with st.expander("Example Questions", expanded=True):
    st.markdown(
        f"""
- Show {choose_default_y(schema)} by {choose_default_x(schema)}
- Show average {choose_default_y(schema)} by {choose_default_x(schema)}
- Show count by {choose_default_x(schema)}
- Which {choose_default_x(schema)} has the highest {choose_default_y(schema)}?
- Show top 5 {choose_default_x(schema)} by {choose_default_y(schema)}
- Show bottom 5 {choose_default_x(schema)} by {choose_default_y(schema)}
- Now show the same analysis as a pie chart
- Instead, make it average by {choose_default_x(schema)}
"""
    )

where_clause, where_params = build_where_clause(active_filters)
filtered_preview = query_dataset_preview(connection, schema["table_name"], where_clause, where_params)
filtered_row_count = query_filtered_row_count(connection, schema["table_name"], where_clause, where_params)

overview_col, preview_col = st.columns([1.35, 1], gap="large")

with overview_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    profile_left, profile_mid, profile_right = st.columns(3)
    profile_left.metric("Numeric Fields", len(schema["numeric_columns"]))
    profile_mid.metric("Categorical Fields", len(schema["categorical_columns"]))
    profile_right.metric("Rows After Filters", filtered_row_count)
    st.markdown(
        f'<div class="kpi-note">Available columns: {", ".join(schema["columns"][:8])}'
        + (" ..." if len(schema["columns"]) > 8 else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with preview_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(filtered_preview if not filtered_preview.empty else preview_df, use_container_width=True, height=220)
    st.markdown("</div>", unsafe_allow_html=True)

if show_architecture:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Architecture Flow</div>', unsafe_allow_html=True)
    flow_a, flow_b, flow_c, flow_d = st.columns(4)
    flow_a.metric("1", "Prompt")
    flow_b.metric("2", "LLM Parse")
    flow_c.metric("3", "SQLite Query")
    flow_d.metric("4", "Plotly Dashboard")
    st.caption(
        "User query is mapped to schema fields, validated against the uploaded dataset, aggregated with SQL over SQLite, and rendered as interactive visuals."
    )
    st.markdown("</div>", unsafe_allow_html=True)


if prompt:
    st.session_state.history.append(prompt)

    with st.spinner("AI analyzing data..."):
        schema_columns = [
            f"{column} ({'numeric' if column in schema['numeric_columns'] else 'categorical'})"
            for column in schema["columns"]
        ]
        raw_analysis = interpret_query(prompt, schema_columns)
        analysis = coerce_analysis(prompt, raw_analysis, schema)
        analysis = merge_follow_up_analysis(prompt, analysis, schema)
        result_limit, sort_direction = infer_limit(prompt)
        date_grain = infer_date_grain(prompt, analysis["x_column"], schema)

        data, value_column, x_column, y_column = aggregate_data(
            connection,
            schema,
            analysis["x_column"],
            analysis["y_column"],
            analysis["aggregation"],
            active_filters,
            date_grain=date_grain,
            result_limit=result_limit,
            sort_direction=sort_direction,
        )

        auto_chart = infer_auto_chart(x_column, analysis["aggregation"], data)
        chart_type = chart_override.lower() if chart_override != "Auto" else auto_chart

        fig = generate_chart(data, chart_type, x_column, value_column)
        ranked_data, trend_data = build_supporting_views(data, x_column, value_column)
        metrics = build_summary_metrics(data, value_column)
        insight = generate_insight(prompt, x_column, analysis["aggregation"], metrics)
        resolution_note = build_resolution_note(analysis, prompt, schema)
        query_interpretation = build_query_interpretation(analysis, value_column, auto_chart)
        confidence_score = compute_confidence(analysis)
        chart_reason = explain_chart_choice(chart_type, x_column, data)

    st.session_state.last_analysis = {
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": analysis["aggregation"],
        "chart_type": chart_type,
    }

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows in Visual", metrics["row_count"])
    k2.metric("Total", format_number(metrics["total"]))
    k3.metric("Highest", format_number(metrics["max"]))
    k4.metric("Average", format_number(metrics["avg"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Primary Visual</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2.2, 1], gap="large")

    with col1:
        st.plotly_chart(fig, use_container_width=True, key="primary_visual_chart")

    with col2:
        st.subheader("AI Insight")
        st.write(insight)
        st.info(resolution_note)
        if confidence_score < 0.6:
            st.warning(f"Low mapping confidence: {confidence_score:.0%}. Refine the metric or grouping field for a more precise result.")
        else:
            st.success(f"Mapping confidence: {confidence_score:.0%}")
        st.caption(
            f"Using x: {x_column} ({analysis['source_x']}), y: {value_column} ({analysis['source_y']}), agg: {analysis['aggregation']}"
        )
        st.caption(f"Auto chart recommendation: {auto_chart}")
        st.caption(chart_reason)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis Workspace</div>', unsafe_allow_html=True)
    tab_overview, tab_supporting, tab_details, tab_export = st.tabs(
        ["Interpretation", "Supporting Visuals", "Detail Table", "Export"]
    )

    with tab_overview:
        st.dataframe(query_interpretation, use_container_width=True, hide_index=True)
        meta_left, meta_right = st.columns(2)
        with meta_left:
            if date_grain:
                st.caption(f"Time grouping applied: {date_grain}")
        with meta_right:
            if result_limit:
                limit_label = "top" if sort_direction == "desc" else "bottom"
                st.caption(f"Result shaping applied: {limit_label} {result_limit}")

    with tab_supporting:
        support_left, support_right = st.columns(2, gap="large")
        with support_left:
            st.markdown("Top Categories")
            st.plotly_chart(
                generate_chart(ranked_data, "bar", x_column, value_column),
                use_container_width=True,
                key="top_categories_chart",
            )
        with support_right:
            st.markdown("Trend / Pattern View")
            secondary_chart = "line" if trend_data[x_column].nunique(dropna=False) > 2 else "bar"
            st.plotly_chart(
                generate_chart(trend_data, secondary_chart, x_column, value_column),
                use_container_width=True,
                key="trend_pattern_chart",
            )

    with tab_details:
        st.dataframe(data, use_container_width=True, height=320)

    with tab_export:
        st.download_button(
            label="Download Visual Data CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="dashboard_visual_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            label="Download Query Interpretation CSV",
            data=query_interpretation.to_csv(index=False).encode("utf-8"),
            file_name="query_interpretation.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Use this section during judging to show exactly how the query was translated into dashboard logic.")

    st.markdown("</div>", unsafe_allow_html=True)


st.sidebar.markdown("### Query History")
for query in st.session_state.history[-10:]:
    st.sidebar.write(query)
