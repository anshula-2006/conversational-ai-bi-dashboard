import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CHART_COLORS = ["#0f4c81", "#1768ac", "#2e86c1", "#4ea5d9", "#79c2e8", "#b7def2", "#e6f4fb"]
GRID_COLOR = "rgba(15, 76, 129, 0.10)"
TEXT_COLOR = "#17324d"
ACCENT_COLOR = "#1768ac"
PAPER_BG = "rgba(0,0,0,0)"


def _sorted_plot_data(data, x, y, ascending=False, limit=None):
    plot_data = data.copy()
    if y in plot_data.columns:
        plot_data = plot_data.sort_values(y, ascending=ascending)
    if limit:
        plot_data = plot_data.head(limit)
    return plot_data.reset_index(drop=True)


def _top_n_with_other(data, x, y, limit=6):
    plot_data = _sorted_plot_data(data, x, y, ascending=False)
    if len(plot_data) <= limit:
        return plot_data
    top = plot_data.head(limit).copy()
    other_value = plot_data[y].iloc[limit:].sum()
    other_row = {x: "Other", y: other_value}
    return pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)


def format_chart_label(label):
    if str(label).strip().lower() == "duration":
        return "Duration (days)"
    return str(label)


def generate_chart(data, chart_type, x, y):
    safe_chart = str(chart_type).lower()
    display_x = format_chart_label(x)
    display_y = format_chart_label(y)
    plot_data = data.copy()
    unique_groups = plot_data[x].nunique(dropna=False) if x in plot_data.columns else 0
    title_text = f"{display_y} by {display_x}"

    if safe_chart == "line":
        if unique_groups > 12 and y in plot_data.columns:
            plot_data = plot_data.sort_values(y, ascending=False).head(12).reset_index(drop=True)
        try:
            plot_data = plot_data.sort_values(x).reset_index(drop=True)
        except Exception:
            plot_data = plot_data.reset_index(drop=True)
    elif safe_chart == "bar":
        plot_data = _sorted_plot_data(plot_data, x, y, ascending=True, limit=12)
    elif safe_chart == "pie":
        plot_data = _top_n_with_other(plot_data, x, y, limit=5)
    elif safe_chart == "treemap":
        plot_data = _top_n_with_other(plot_data, x, y, limit=10)
    elif safe_chart == "funnel":
        plot_data = _sorted_plot_data(plot_data, x, y, ascending=False, limit=8)
    elif safe_chart == "scatter" and unique_groups > 40:
        plot_data = plot_data.head(40).reset_index(drop=True)

    try:
        if safe_chart == "bar":
            fig = px.bar(
                plot_data,
                x=y,
                y=x,
                orientation="h",
                color=y,
                color_continuous_scale=["#dcecf8", CHART_COLORS[1], CHART_COLORS[0]],
            )
            title_text = f"Top {display_x} by {display_y}"
        elif safe_chart == "line":
            fig = px.line(plot_data, x=x, y=y, markers=True, color_discrete_sequence=[CHART_COLORS[0]])
        elif safe_chart == "pie":
            fig = px.pie(plot_data, names=x, values=y, color_discrete_sequence=CHART_COLORS)
        elif safe_chart == "scatter":
            fig = px.scatter(plot_data, x=x, y=y, color=y, color_continuous_scale="Blues", size=y if y in plot_data.columns else None)
        elif safe_chart == "histogram":
            fig = px.histogram(plot_data, x=y, color_discrete_sequence=[CHART_COLORS[0]])
        elif safe_chart == "treemap":
            fig = px.treemap(plot_data, path=[x], values=y, color=y, color_continuous_scale="Blues")
        elif safe_chart == "funnel":
            fig = go.Figure(
                go.Funnel(
                    y=plot_data[x],
                    x=plot_data[y],
                    marker={"color": CHART_COLORS[: len(plot_data)]},
                    textinfo="value+percent initial",
                )
            )
        elif safe_chart == "heatmap":
            heatmap_data = plot_data.copy()
            heatmap_data["_bucket"] = list(range(1, len(heatmap_data) + 1))
            fig = px.density_heatmap(heatmap_data, x=x, y="_bucket", z=y, histfunc="avg", color_continuous_scale="Blues")
            fig.update_yaxes(title_text="Rank Bucket")
        elif safe_chart == "waterfall":
            fig = go.Figure(
                go.Waterfall(
                    x=plot_data[x],
                    y=plot_data[y],
                    measure=["relative"] * len(plot_data),
                )
            )
        elif safe_chart == "gauge":
            gauge_value = float(plot_data[y].iloc[0]) if not plot_data.empty else 0.0
            gauge_max = float(max(plot_data[y].max(), gauge_value, 1))
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=gauge_value,
                    title={"text": display_y},
                    gauge={
                        "axis": {"range": [None, gauge_max]},
                        "bar": {"color": "#1f6aa5"},
                    },
                )
            )
        else:
            fig = px.bar(
                plot_data,
                x=y,
                y=x,
                orientation="h",
                color=y,
                color_continuous_scale=["#dcecf8", CHART_COLORS[1], CHART_COLORS[0]],
            )
            title_text = f"Top {display_x} by {display_y}"
    except Exception:
        fig = px.bar(
            plot_data,
            x=y,
            y=x,
            orientation="h",
            color_discrete_sequence=[CHART_COLORS[0]],
        )
        title_text = f"Top {display_x} by {display_y}"

    fig.update_layout(
        template="plotly_white",
        height=580,
        margin=dict(l=36, r=36, t=80, b=52),
        title=dict(text=title_text, x=0.02, xanchor="left", font=dict(size=22, color=TEXT_COLOR)),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PAPER_BG,
        legend_title_text=display_x,
        font=dict(color=TEXT_COLOR, size=13),
        hoverlabel=dict(bgcolor="white", font_size=13, font_color=TEXT_COLOR),
    )

    if safe_chart != "gauge":
        fig.update_xaxes(title_text=display_x, showgrid=False, zeroline=False)
        fig.update_yaxes(title_text=display_y, gridcolor=GRID_COLOR, zeroline=False)

    if safe_chart == "bar":
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        fig.update_traces(
            marker_line_color="rgba(255,255,255,0.85)",
            marker_line_width=1.4,
            hovertemplate=f"{display_x}: %{{y}}<br>{display_y}: %{{x:,.2f}}<extra></extra>",
        )
        fig.update_xaxes(title_text=display_y, showgrid=True, gridcolor=GRID_COLOR)
        fig.update_yaxes(title_text="")
    elif safe_chart == "line":
        fig.update_traces(
            line=dict(width=4, color=ACCENT_COLOR, shape="spline", smoothing=0.55),
            marker=dict(size=9, line=dict(width=2, color="white"), color="#4ea5d9"),
            fill="tozeroy",
            fillcolor="rgba(78, 165, 217, 0.10)",
            hovertemplate=f"{display_x}: %{{x}}<br>{display_y}: %{{y:,.2f}}<extra></extra>",
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-25, showgrid=False, automargin=True)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, automargin=True)
        fig.update_traces(cliponaxis=False)
    elif safe_chart == "scatter":
        fig.update_layout(coloraxis_colorbar=dict(title=display_y))
        fig.update_traces(
            marker=dict(size=12, line=dict(width=1.5, color="white"), opacity=0.82),
            hovertemplate=f"{display_x}: %{{x}}<br>{display_y}: %{{y:,.2f}}<extra></extra>",
        )
    elif safe_chart == "pie":
        fig.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(size=12, color=TEXT_COLOR),
            hole=0.42,
            marker=dict(line=dict(color="white", width=2)),
            pull=[0.04] + [0.01] * max(len(plot_data) - 1, 0),
            sort=False,
            hovertemplate=f"{display_x}: %{{label}}<br>{display_y}: %{{value:,.2f}}<br>Share: %{{percent}}<extra></extra>",
        )
        fig.update_layout(
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
            margin=dict(l=30, r=150, t=80, b=40),
            uniformtext_minsize=10,
            uniformtext_mode="hide",
            annotations=[
                dict(
                    text=display_y,
                    x=0.5,
                    y=0.5,
                    font=dict(size=15, color=TEXT_COLOR),
                    showarrow=False,
                )
            ],
        )
    elif safe_chart == "histogram":
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_layout(showlegend=False)
    elif safe_chart == "treemap":
        fig.update_traces(
            textinfo="label",
            marker=dict(line=dict(width=2, color="white")),
            hovertemplate=f"{display_x}: %{{label}}<br>{display_y}: %{{value:,.2f}}<extra></extra>",
        )
    elif safe_chart == "heatmap":
        fig.update_layout(coloraxis_colorbar=dict(title=display_y))
    elif safe_chart == "waterfall":
        fig.update_traces(
            connector={"line": {"color": "rgba(23, 50, 77, 0.25)"}},
            increasing={"marker": {"color": CHART_COLORS[1]}},
            decreasing={"marker": {"color": "#d96c6c"}},
            totals={"marker": {"color": CHART_COLORS[0]}},
        )
    elif safe_chart == "gauge":
        fig.update_layout(
            margin=dict(l=12, r=12, t=64, b=12),
            paper_bgcolor=PAPER_BG,
        )
        fig.update_traces(
            gauge={
                "axis": {"tickcolor": TEXT_COLOR, "tickwidth": 1},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 0.5 * float(plot_data[y].max() if not plot_data.empty else 1)], "color": "#dcecf8"},
                    {"range": [0.5 * float(plot_data[y].max() if not plot_data.empty else 1), float(max(plot_data[y].max() if not plot_data.empty else 1, 1))], "color": "#b7def2"},
                ],
            },
            number={"font": {"size": 36, "color": TEXT_COLOR}},
        )

    return fig
