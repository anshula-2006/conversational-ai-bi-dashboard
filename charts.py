import plotly.express as px
import plotly.graph_objects as go

CHART_COLORS = ["#1f6aa5", "#2f86c1", "#5aa7d6", "#87c1e3", "#b9d9ef", "#dcecf8"]


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
    if safe_chart == "line":
        if unique_groups > 12 and y in plot_data.columns:
            plot_data = plot_data.sort_values(y, ascending=False).head(12).reset_index(drop=True)
        try:
            plot_data = plot_data.sort_values(x).reset_index(drop=True)
        except Exception:
            plot_data = plot_data.reset_index(drop=True)
    try:
        if safe_chart == "bar":
            fig = px.bar(plot_data, x=x, y=y, color_discrete_sequence=[CHART_COLORS[0]])
        elif safe_chart == "line":
            fig = px.line(plot_data, x=x, y=y, markers=True, color_discrete_sequence=[CHART_COLORS[0]])
        elif safe_chart == "pie":
            fig = px.pie(plot_data, names=x, values=y, color_discrete_sequence=CHART_COLORS)
        elif safe_chart == "scatter":
            fig = px.scatter(plot_data, x=x, y=y, color_discrete_sequence=[CHART_COLORS[0]])
        elif safe_chart == "histogram":
            fig = px.histogram(plot_data, x=y, color_discrete_sequence=[CHART_COLORS[0]])
        elif safe_chart == "treemap":
            fig = px.treemap(plot_data, path=[x], values=y, color=y, color_continuous_scale="Blues")
        elif safe_chart == "funnel":
            fig = go.Figure(go.Funnel(y=plot_data[x], x=plot_data[y]))
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
            fig = px.bar(plot_data, x=x, y=y, color_discrete_sequence=[CHART_COLORS[0]])
    except Exception:
        fig = px.bar(plot_data, x=x, y=y, color_discrete_sequence=[CHART_COLORS[0]])
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=60, b=24),
        title=dict(text=f"{display_y} by {display_x}", x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text=display_x,
        font=dict(color="#17324d"),
    )
    if safe_chart != "gauge":
        fig.update_xaxes(title_text=display_x, showgrid=False)
        fig.update_yaxes(title_text=display_y, gridcolor="rgba(22, 62, 98, 0.08)")
    if safe_chart == "bar":
        fig.update_traces(marker_line_color="rgba(255,255,255,0.65)", marker_line_width=1.2)
    elif safe_chart == "line":
        fig.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1, color="white")))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(tickangle=-25)
    elif safe_chart == "scatter":
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    elif safe_chart == "pie":
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            insidetextorientation="radial",
            marker=dict(line=dict(color="white", width=2)),
            pull=[0.02] + [0] * max(len(data) - 1, 0),
        )
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=60, b=80),
        )
    return fig
