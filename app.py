import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from beam import (
    Analysis,
    DistributedLoad,
    Load,
    PointLoad,
    RectangularSection,
    analyze,
    load_problem,
    safety_rating,
)
from materials import MATERIALS, Material

LINE_COLOR = "#d94c86"
FILL_COLOR = "rgba(217, 76, 134, 0.16)"
GRID_COLOR = "#f6d5e2"
TEXT_COLOR = "#4b2a3a"

PHYSICS_NOTES = r"""
The beam rests on a pin at the left end and a roller at the right, so each
support can only push up. Every result below comes from those two supports
and the loads you enter.

**Reactions.** The beam is not moving, so forces and turning effects balance.
Taking moments about the left support gives
$R_B = \sum F_i x_i \,/\, L$ and $R_A = \sum F_i - R_B$. A distributed load
acts like its total force placed at its midpoint.

**Shear force.** Cut the beam at $x$: $V(x)$ is $R_A$ minus every load to the
left of the cut. It jumps at point loads and slopes under distributed loads.

**Bending moment.** At the same cut, $M(x) = R_A x - \sum F_i (x - x_i)$ over
the loads to the left. It is zero at both supports and peaks where the shear
crosses zero.

**Deflection.** Curvature is proportional to moment: $E I \, y'' = M(x)$. The
app integrates this twice numerically and fixes the two constants so the beam
does not move at either support.

**Stress and safety.** For a rectangle $I = b h^3 / 12$, and the peak stress
sits at the top or bottom face: $\sigma = M c / I$ with $c = h / 2$. The safety
factor is material strength divided by peak stress. Green means 2 or more,
yellow means 1 to 2, red means below 1.

**Assumptions.** Linear elastic material, small deflections, the beam's own
weight ignored, and no shear deformation or buckling.
"""

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


def load_inputs() -> tuple[list[Load], int]:
    """Return the complete load rows and how many rows were left incomplete."""
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

    complete_points, complete_udls = points.dropna(), udls.dropna()
    incomplete = len(points) - len(complete_points) + len(udls) - len(complete_udls)

    loads: list[Load] = [
        PointLoad(row.position, row.magnitude * 1000) for row in complete_points.itertuples()
    ]
    loads += [
        DistributedLoad(row.start, row.end, row.intensity * 1000)
        for row in complete_udls.itertuples()
    ]
    return loads, incomplete


def usable_loads(loads: list[Load], length: float, incomplete: int) -> list[Load]:
    """Drop loads that don't fit on the beam, telling the user why."""
    if incomplete:
        st.info(f"{incomplete} incomplete row(s) skipped until every cell is filled in.")

    usable = []
    for load in loads:
        if problem := load_problem(load, length):
            st.warning(f"{problem} It is left out of the analysis.")
        else:
            usable.append(load)

    if not usable:
        st.info("No loads on the beam yet. Add a row to either table to see the diagrams change.")
    return usable


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
    cols[5].metric(
        "Safety factor",
        sf,
        help=f"{material.name} {material.strength_kind}: {material.strength / 1e6:.0f} MPa",
    )


def show_safety(result: Analysis, material: Material) -> None:
    factor = result.safety_factor
    if math.isinf(factor):
        return

    stress = f"Peak stress {result.max_stress / 1e6:.1f} MPa"
    strength = f"{material.strength / 1e6:.0f} MPa {material.strength_kind}"
    rating = safety_rating(factor)
    if rating == "safe":
        st.success(f"Safe. Safety factor {factor:.2f}. {stress} against a {strength}.")
    elif rating == "marginal":
        st.warning(
            f"Marginal. Safety factor {factor:.2f}, below the usual design target of 2. "
            f"{stress} against a {strength}."
        )
    else:
        st.error(
            f"Failing. Safety factor {factor:.2f}. {stress} exceeds the {strength}, "
            "so this beam would not hold."
        )


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
        fig.add_hline(y=0, line_width=1, line_color="#c9a3b4", row=row, col=1)

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
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    fig.update_annotations(font={"family": "Fredoka, sans-serif", "size": 16})
    fig.update_layout(
        height=780,
        showlegend=False,
        hovermode="x",
        margin={"t": 40, "b": 40},
        font={"family": "Nunito, sans-serif", "color": TEXT_COLOR},
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": LINE_COLOR},
    )
    return fig


st.set_page_config(page_title="Beam Simulator", layout="wide")
st.title("Simply supported beam")
st.caption("Set up a beam, add some loads, and watch it bend.")
with st.expander("How the physics works"):
    st.markdown(PHYSICS_NOTES)

length, material, section = beam_inputs()
loads, incomplete = load_inputs()
result = analyze(length, usable_loads(loads, length, incomplete), material, section)

show_summary(result, material)
show_safety(result, material)
st.plotly_chart(diagrams(result), width="stretch")
