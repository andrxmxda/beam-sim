import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from beam import Analysis, DistributedLoad, PointLoad, RectangularSection, analyze
from materials import MATERIALS, Material

LINE_COLOR = "#2a78d6"
FILL_COLOR = "rgba(42, 120, 214, 0.15)"

DEFAULT_POINT_LOADS = pd.DataFrame({"position": [2.0], "magnitude": [10.0]})
DEFAULT_UDLS = pd.DataFrame({"start": [0.0], "end": [6.0], "intensity": [2.0]})


def beam_inputs() -> tuple[float, Material, RectangularSection]:
    with st.sidebar:
        st.header("Beam")
        length = st.number_input("Span (m)", min_value=0.1, value=6.0, step=0.5)
        material = MATERIALS[st.selectbox("Material", list(MATERIALS))]

        st.header("Cross-section")
        width = st.number_input("Width (mm)", min_value=1.0, value=100.0, step=10.0)
        height = st.number_input("Height (mm)", min_value=1.0, value=300.0, step=10.0)

    return length, material, RectangularSection(width / 1000, height / 1000)


def load_inputs() -> list[PointLoad | DistributedLoad]:
    number = st.column_config.NumberColumn
    left, right = st.columns(2)
    with left:
        st.subheader("Point loads")
        points = st.data_editor(
            DEFAULT_POINT_LOADS,
            key="point_loads",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "position": number("Position (m)", min_value=0.0, required=True),
                "magnitude": number("Magnitude (kN)", required=True),
            },
        )
    with right:
        st.subheader("Distributed loads")
        udls = st.data_editor(
            DEFAULT_UDLS,
            key="udls",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "start": number("Start (m)", min_value=0.0, required=True),
                "end": number("End (m)", min_value=0.0, required=True),
                "intensity": number("Intensity (kN/m)", required=True),
            },
        )

    loads: list[PointLoad | DistributedLoad] = [
        PointLoad(row.position, row.magnitude * 1000) for row in points.dropna().itertuples()
    ]
    loads += [
        DistributedLoad(row.start, row.end, row.intensity * 1000)
        for row in udls.dropna().itertuples()
    ]
    return loads


def show_summary(result: Analysis, material: Material) -> None:
    r_a, r_b = result.reactions
    sf = "n/a" if math.isinf(result.safety_factor) else f"{result.safety_factor:.2f}"

    cols = st.columns(6)
    cols[0].metric("Left reaction", f"{r_a / 1000:.2f} kN")
    cols[1].metric("Right reaction", f"{r_b / 1000:.2f} kN")
    cols[2].metric(
        "Max moment",
        f"{result.max_moment / 1000:.2f} kN·m",
        help=f"At x = {result.max_moment_position:.2f} m",
    )
    cols[3].metric(
        "Max deflection",
        f"{abs(result.max_deflection) * 1000:.2f} mm",
        help=f"At x = {result.max_deflection_position:.2f} m",
    )
    cols[4].metric("Max stress", f"{result.max_stress / 1e6:.1f} MPa")
    cols[5].metric("Safety factor", sf, help=f"{material.name} strength: {material.strength / 1e6:.0f} MPa")


def diagrams(result: Analysis) -> go.Figure:
    series = [
        ("Shear force", "kN", result.shear / 1000),
        ("Bending moment", "kN·m", result.moment / 1000),
        ("Deflection", "mm", result.deflection * 1000),
    ]
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=[f"{name} ({unit})" for name, unit, _ in series],
    )
    for row, (name, unit, values) in enumerate(series, start=1):
        fig.add_trace(
            go.Scatter(
                x=result.x,
                y=values,
                mode="lines",
                line={"color": LINE_COLOR, "width": 2},
                fill="tozeroy",
                fillcolor=FILL_COLOR,
                name=name,
                hovertemplate=f"x = %{{x:.2f}} m<br>{name}: %{{y:.2f}} {unit}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_width=1, line_color="gray", row=row, col=1)

    peaks = [
        (2, result.max_moment_position, result.max_moment / 1000, "kN·m"),
        (3, result.max_deflection_position, result.max_deflection * 1000, "mm"),
    ]
    for row, x, y, unit in peaks:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker={"color": LINE_COLOR, "size": 9},
                text=[f"{y:.2f} {unit}"],
                # Label on the inside of the peak so the axis edge never clips it.
                textposition="bottom center" if y >= 0 else "top center",
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    fig.update_xaxes(title_text="Position along beam (m)", row=3, col=1)
    fig.update_layout(height=780, showlegend=False, hovermode="x", margin={"t": 40, "b": 40})
    return fig


st.set_page_config(page_title="Beam Simulator", layout="wide")
st.title("Simply supported beam")

length, material, section = beam_inputs()
loads = load_inputs()
result = analyze(length, loads, material, section)

show_summary(result, material)
st.plotly_chart(diagrams(result), width="stretch")
