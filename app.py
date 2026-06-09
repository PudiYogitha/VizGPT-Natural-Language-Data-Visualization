import json
from typing import Any
from ollama import chat
import re
import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib


SYSTEM_MESSAGE = """
You are a data visualization assistant.

IMPORTANT:
Use ONLY the exact column names provided.
Never invent column names.
Return JSON only.
"""

SUPPORTED_CHARTS = {"bar", "line", "scatter", "histogram", "pie", "box", "area"}


def sample_datasets() -> dict[str, pd.DataFrame]:
    return {
        "Iris": px.data.iris(),
        "Tips": px.data.tips(),
        "Gapminder": px.data.gapminder(),
    }


def build_prompt(nl: str, columns: list[str]) -> list[dict[str, str]]:
    schema = ", ".join(columns)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": (
                f"Dataset columns: {schema}.\n"
                f"User request: {nl}\n"
                "Return JSON only."
            ),
        },
    ]


def extract_json_object(content: str) -> str:
    start = content.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")

    bracket_count = 0
    for idx in range(start, len(content)):
        if content[idx] == "{":
            bracket_count += 1
        elif content[idx] == "}":
            bracket_count -= 1
            if bracket_count == 0:
                return content[start: idx + 1]

    raise ValueError("No complete JSON object found in model response.")


def clean_json_text(json_text: str) -> str:
    json_text = json_text.strip()
    json_text = json_text.replace("\ufeff", "")
    json_text = re.sub(r"[\u2018\u2019\u201C\u201D]", '"', json_text)
    json_text = re.sub(r",\s*([\}\]])", r"\1", json_text)
    json_text = re.sub(r"(?<=\{|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*:", r'"\1":', json_text)
    return json_text


def infer_chart_type(content: str, spec: dict[str, Any]) -> str | None:
    text = content.lower()
    if "histogram" in text or "distribution" in text:
        return "histogram"
    if "scatter" in text or "scatter plot" in text:
        return "scatter"
    if "pie" in text or "pie chart" in text:
        return "pie"
    if "line" in text or "trend" in text:
        return "line"
    if "box" in text or "box plot" in text:
        return "box"
    if "area" in text:
        return "area"
    if "bar" in text or ("compare" in text and spec.get("y")):
        return "bar"
    return None


@st.cache_data
def request_chart_spec(nl: str, columns_str: str) -> dict[str, Any]:
    """Request chart spec from Ollama with caching.
    
    Args:
        nl: Natural language request
        columns_str: Comma-separated column names (used for cache key)
    """
    columns = [col.strip() for col in columns_str.split(",")]
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE,
        },
        {
            "role": "user",
            "content": (
                f"Dataset columns(use EXACT names only): {', '.join(columns)}.\n"
                f"User request: {nl}\n"
                "Return ONLY valid JSON with keys chart_type, x, y, color, aggregation, title, orientation, description.\n"
                "Always include `x` and `y` in the JSON output. If a chart type does not require one of these fields, set it explicitly to null.\n"
                f"Use only these chart types: {', '.join(sorted(SUPPORTED_CHARTS))}.\n"
                "Return column names exactly as provided."
            ),
        },
    ]

    response = chat(
        model="llama3",
        messages=messages,
    )

    content = response["message"]["content"]

    try:
        json_text = extract_json_object(content)
        spec = json.loads(json_text)
    except json.JSONDecodeError:
        try:
            json_text = clean_json_text(json_text)
            spec = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse JSON from model response: {exc}\n\nResponse:\n{content}"
            ) from exc
    except ValueError as exc:
        raise ValueError(
            f"Could not parse JSON from model response: {exc}\n\nResponse:\n{content}"
        ) from exc

    spec["chart_type"] = spec.get("chart_type") or spec.get("type")
    if not spec.get("chart_type"):
        spec["chart_type"] = infer_chart_type(content, spec)

    if spec.get("chart_type") not in SUPPORTED_CHARTS:
        raise ValueError(
            f"Unsupported chart type: {spec.get('chart_type')}"
        )

    spec["x"] = normalize_column_name(spec.get("x"), columns)
    spec["y"] = normalize_column_name(spec.get("y"), columns)
    spec["color"] = normalize_column_name(spec.get("color"), columns)

    return spec


def get_chart_spec_with_progress(nl: str, columns: list[str]):
    """Wrapper that shows progress while fetching chart spec."""
    columns_str = ",".join(columns)
    
    with st.spinner("🔄 Generating chart specification..."):
        spec = request_chart_spec(nl, columns_str)
    
    return spec

def normalize_column_name(name, columns):
    if not name:
        return None
    
    # Handle case where name is a list (take first element)
    if isinstance(name, list):
        name = name[0] if name else None
    
    if not name or not isinstance(name, str):
        return None

    cleaned = (
        name.lower()
        .replace(" ", "_")
        .replace("(cm)", "")
        .strip("_ ")
    )

    for col in columns:
        if cleaned == col.lower():
            return col

    return None


def first_column_by_dtype(df: pd.DataFrame, include=None):
    if include is None:
        return df.columns[0] if len(df.columns) else None

    cols = df.select_dtypes(include=include).columns
    return cols[0] if len(cols) else None


def render_chart(df: pd.DataFrame, spec: dict[str, Any]):
    chart_type = spec.get("chart_type")
    x = spec.get("x")
    y = spec.get("y")
    color = spec.get("color")
    aggregation = spec.get("aggregation")
    title = spec.get("title") or "Generated Visualization"
    orientation = spec.get("orientation") or "v"
    
    # Remove invalid color values (empty lists, None, etc.)
    if not color or (isinstance(color, list) and len(color) == 0):
        color = None

    if aggregation and x and y and chart_type in {"bar", "line", "area"}:
        df = df.groupby(x, as_index=False).agg({y: "mean"})

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if chart_type == "histogram":
        if not x and not y:
            x = first_column_by_dtype(df, include=["number"]) or first_column_by_dtype(df, include=["object", "category"]) or df.columns[0]
        return px.histogram(df, x=x or y, color=color, title=title)

    if chart_type == "bar":
        if not x and not y and aggregation and str(aggregation).upper() == "COUNT":
            grouping_col = color or first_column_by_dtype(df, include=["object", "category"]) or df.columns[0]
            count_df = df.groupby(grouping_col, as_index=False).size().rename(columns={"size": "count"})
            return px.bar(count_df, x=grouping_col, y="count", title=title)
        if not x and y:
            x = first_column_by_dtype(df, include=["object", "category"]) or df.columns[0]
        if x and not y:
            y = y or first_column_by_dtype(df, include=["number"]) or df.columns[0]
        return px.bar(df, x=x, y=y, color=color, title=title, orientation=orientation)

    if chart_type == "line":
        if not x and not y:
            if len(numeric_cols) >= 2:
                x, y = numeric_cols[0], numeric_cols[1]
            elif len(numeric_cols) == 1:
                y = numeric_cols[0]
                x = df.index.name or "index"
                df = df.reset_index()
            else:
                x = first_column_by_dtype(df, include=["object", "category"]) or df.columns[0]
                y = df.columns[1] if len(df.columns) > 1 else x
        elif not x:
            x = first_column_by_dtype(df, include=["object", "category"]) or df.index.name or "index"
            if x == "index":
                df = df.reset_index()
        elif not y:
            y = first_column_by_dtype(df, include=["number"]) or df.columns[0]
        return px.line(df, x=x, y=y, color=color, title=title)

    if chart_type == "scatter":
        if not x or not y:
            if len(numeric_cols) >= 2:
                x = x or numeric_cols[0]
                y = y or numeric_cols[1]
            elif len(numeric_cols) == 1:
                y = y or numeric_cols[0]
                x = x or first_column_by_dtype(df, include=["object", "category"]) or df.columns[0]
        return px.scatter(df, x=x, y=y, color=color, title=title)

    if chart_type == "pie":
        return px.pie(df, names=x, values=y, title=title)
    if chart_type == "box":
        return px.box(df, x=x, y=y, color=color, title=title)
    if chart_type == "area":
        if not x and y:
            x = df.index.name or "index"
            df = df.reset_index()
        return px.area(df, x=x, y=y, color=color, title=title)

    raise ValueError(f"Cannot render chart type: {chart_type}")


def main():
    st.set_page_config(page_title="VizGPT", layout="wide")
    st.title("VizGPT — Natural Language Data Visualization")
    st.write("Describe the visualization you want and VizGPT will generate it.")

    samples = sample_datasets()
    dataset_name = st.sidebar.selectbox("Sample dataset", list(samples.keys()))
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = samples[dataset_name]

    st.sidebar.markdown("---")
    st.sidebar.write("**Columns in current dataset:**")
    st.sidebar.code(", ".join(df.columns), language="text")

    nl_request = st.text_area("Enter a natural language visualization request", "Show average total bill by day and time")
    generate_button = st.button("Generate Visualization")

    if generate_button:
        if df.empty:
            st.error("Dataset is empty. Upload a valid CSV or select a sample dataset.")
            return
        try:
            spec = get_chart_spec_with_progress(nl_request, list(df.columns))
            st.write("### Model chart spec")
            st.json(spec)
            
            with st.spinner("📊 Rendering chart..."):
                chart = render_chart(df, spec)
            
            st.plotly_chart(chart, use_container_width=True)
            description = spec.get("description")
            if description:
                st.info(description)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
