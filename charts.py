import plotly.express as px
import plotly.graph_objects as go

def generate_chart(data, chart_type, x, y):
    safe_chart = str(chart_type).lower()
    try:
        if safe_chart == "bar":
            fig = px.bar(data, x=x, y=y, color=x)
        elif safe_chart == "line":
            fig = px.line(data, x=x, y=y, markers=True)
        elif safe_chart == "pie":
            fig = px.pie(data, names=x, values=y)
        elif safe_chart == "scatter":
            fig = px.scatter(data, x=x, y=y)
        elif safe_chart == "histogram":
            fig = px.histogram(data, x=y)
        elif safe_chart == "treemap":
            fig = px.treemap(data, path=[x], values=y, color=y, color_continuous_scale="Blues")
        elif safe_chart == "funnel":
            fig = go.Figure(go.Funnel(y=data[x], x=data[y]))
        elif safe_chart == "heatmap":
            heatmap_data = data.copy()
            heatmap_data["_bucket"] = list(range(1, len(heatmap_data) + 1))
            fig = px.density_heatmap(heatmap_data, x=x, y="_bucket", z=y, histfunc="avg", color_continuous_scale="Blues")
            fig.update_yaxes(title_text="Rank Bucket")
        elif safe_chart == "waterfall":
            fig = go.Figure(
                go.Waterfall(
                    x=data[x],
                    y=data[y],
                    measure=["relative"] * len(data),
                )
            )
        elif safe_chart == "gauge":
            gauge_value = float(data[y].iloc[0]) if not data.empty else 0.0
            gauge_max = float(max(data[y].max(), gauge_value, 1))
            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=gauge_value,
                    title={"text": y},
                    gauge={
                        "axis": {"range": [None, gauge_max]},
                        "bar": {"color": "#1f6aa5"},
                    },
                )
            )
        else:
            fig = px.bar(data, x=x, y=y, color=x)
    except Exception:
        fig = px.bar(data, x=x, y=y)
    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        title=dict(text=f"{y} by {x}", x=0.02, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text=x,
    )
    if safe_chart != "gauge":
        fig.update_xaxes(title_text=x, showgrid=False)
        fig.update_yaxes(title_text=y, gridcolor="rgba(22, 62, 98, 0.08)")
    return fig
