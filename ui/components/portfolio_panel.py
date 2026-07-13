"""Portfolio panel — collapsible holdings pie chart for the selected client.

Reads directly from the same tools the Portfolio agent uses (get_allocation_by_asset_class)
so the chart is always real data, never something the LLM narrated. Refreshes
whenever the client changes or a new answer lands.
"""

import altair as alt
import pandas as pd
import streamlit as st


def render_portfolio_panel(client_id: str) -> None:
    with st.expander("📊 Portfolio snapshot", expanded=True):
        try:
            from app.tools.portfolio_tools import get_allocation_by_asset_class, get_portfolio_value

            alloc = get_allocation_by_asset_class(client_id)
            value = get_portfolio_value(client_id)
        except Exception as exc:  # noqa: BLE001 — a chart failure shouldn't break the chat
            st.warning(f"Couldn't load portfolio snapshot: {exc}")
            return

        st.metric("Total value", f"${value['total_value']:,.2f}")

        rows = [{"asset_class": k, "pct": v} for k, v in alloc["allocation_pct"].items()]
        df = pd.DataFrame(rows)
        chart = (
            alt.Chart(df)
            .mark_arc(innerRadius=50)
            .encode(
                theta="pct:Q",
                color=alt.Color("asset_class:N", legend=alt.Legend(title="Asset class")),
                tooltip=["asset_class:N", alt.Tooltip("pct:Q", format=".1f", title="% of portfolio")],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Grouped by asset class — Individual Stocks vs the ETF types vs Cash Equivalent.")
