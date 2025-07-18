#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 18:17:23 2025

@author: sasidharankumar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced PSC‑Risk Streamlit Dashboard
------------------------------------

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from util import (
    prepare_inspection_data,
    prepare_analysis_data,
    generate_entity_analysis_results,
    get_vessel_historical_score_mapping,
    get_vessel_change_scores,
    get_final_risk_score_mapping,
    get_vessel_segments,
)
from trends_util import prepare_trend_analysis_data
from recommendations_util import (
    load_recommendations_data,
    create_indexes_and_embeddings,
    prepare_recommendations_data,
    generate_open_defect_recommendations,
)

# 🔵🔵 1. GLOBAL CONFIG + THEME  ───────────────────────────────────────────────
# ---------------------------------------------------------------------------
#  A light look‑and‑feel refresh (background stays white, but with richer blues
#  and sleeker table styling).  Feel free to tweak colour hex codes in the CSS.

# st.set_page_config(
#     page_title="PSC Risk Dashboard",
#     page_icon="🚢",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

st.markdown(
    """
    <style>
        /* =====  Global colour palette  ==================================== */
        :root {
            --primary: #0c5db5;   /* main blue */
            --accent : #eaf2ff;   /* very light blue */
            --grey-1 : #f4f6f8;   /* table row alt */
            --text-dk: #132144;   /* headings */
        }

        /* =====  Basic typography overrides  =============================== */
        html, body, .stApp { font-family: "Segoe UI", "Helvetica", sans-serif; }
        h1, h2, h3 { color: var(--text-dk); }

        /* =====  DataFrame tweaks  ========================================= */
        .stDataFrame tbody tr:nth-child(even) { background: var(--grey-1); }
        .stDataFrame thead th { background: var(--accent); color: var(--text-dk); }

        /* =====  Expander arrow colour tweak  ============================== */
        .streamlit-expanderHeader:hover { color: var(--primary); }

        /* =====  Nice boxes for the header‑bar KPIs  ======================== */
        .kpi-card {
            background: var(--accent);
            border-radius: 9px;
            padding: 16px 20px;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,.08);
        }
        .kpi-value { font-size: 32px; font-weight: 700; color: var(--primary); }
        .kpi-label { font-size: 15px; color: #555; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🔵🔵 2. PROFESSIONAL HEADER  ────────────────────────────────────────────────
# ---------------------------------------------------------------------------
#  Renders once at the very top of the app.

header_col1, header_col2 = st.columns([0.12, 0.88])
# with header_col1:
#     st.image("https://raw.githubusercontent.com/streamlit/branding/master/logos/mark/streamlit-mark-primary.png", width=70)
with header_col2:
    st.markdown(
        """
        <div style='padding-top:18px'>
            <span style='font-size:34px; font-weight:700; color:var(--primary)'>
                Port State Control – Risk Management Dashboard
            </span><br>
            <span style='font-size:15px; color:#555'>Track, analyse and mitigate vessel deficiencies</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin-top:6px;'>", unsafe_allow_html=True)

# 🔵🔵 3. DATA LOADING  (unchanged logic – added spinner text)  --------------
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading inspection & scoring data …")
def load_data():
    inspection_data = pd.read_csv("Synergy PSC Inspection.csv")
    inspection_data = prepare_inspection_data(inspection_data)

    generic_factors_data = pd.read_excel(
        "PSC Risk Generic and Dynamic Factors.xlsx", sheet_name="Vessel Generic Factor"
    )
    dynamic_factors_data = pd.read_excel(
        "PSC Risk Generic and Dynamic Factors.xlsx", sheet_name="Ship Dynamic Factors"
    )
    psc_codes_scoring_data = pd.read_csv("PSC_Codes_Scored.csv")
    psc_codes_scoring_data["Code"] = psc_codes_scoring_data["Code"].astype(str)

    analysis_df = prepare_analysis_data(
        inspection_data, psc_codes_scoring_data, generic_factors_data
    )

    return {
        "inspection": inspection_data,
        "generic": generic_factors_data,
        "dynamic": dynamic_factors_data,
        "psc_scores": psc_codes_scoring_data,
        "analysis_df_master": analysis_df,
        "dynamic_factors_data": dynamic_factors_data,
    }

# 🔵🔵 4. (Load recommendations – unchanged)  ------------------------------
internal_checklist, external_checklist, open_defects, deficiency_codes, \
    deficiency_codes_category_df, deficiency_codes_sub_category_df, psc_category_recommenders = load_recommendations_data()
internal_checklist_items, external_checklist_items, category_items, subcategory_items, \
    category_desc_items, subcategory_desc_items = prepare_recommendations_data(
        internal_checklist, external_checklist, deficiency_codes
    )
index_internal, index_external = create_indexes_and_embeddings(
    internal_checklist_items, external_checklist_items, category_items,
    subcategory_items, category_desc_items, subcategory_desc_items
)

# 🔵🔵 5. HELPER ─ add_bar_colours() for nicer bar charts  --------------------
def add_bar_colours(fig, colour="var(--primary)"):
    for bar in fig.data:
        bar.marker.color = colour
    return fig

# ==========================================================================
# 🚢  MAIN  LOGIC (the rest of the original code remains intact, only
#     inserting NEW visualisations and style tweaks – these blocks are
#     marked with 🔵🔵 comments so they stand out.)
# ==========================================================================

#  ( … existing LONG body of the dashboard code … )
#  To keep this excerpt concise, we only show the *additions*.
#  ---------------------------------------------------------------------------------
#  **Insert the following snippets at the indicated locations in your original file**
#  ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 🔵🔵 5‑A  Risk‑Home » after we display `risk_table_display`, inject a KPI row
# ---------------------------------------------------------------------------
# 📍 Locate the block where `st.dataframe(risk_table_display, ...)` is called.
#    Immediately **after** that line, add:

# --- KPI summary ------------------------------------------------------------
if not risk_table_display.empty:
    high_risk   = (risk_table_display["Risk Label"] == "High").sum()
    medium_risk = (risk_table_display["Risk Label"] == "Medium").sum()
    low_risk    = (risk_table_display["Risk Label"] == "Low").sum()

    k1, k2, k3 = st.columns(3)
    for col, val, lbl in zip(
        (k1, k2, k3),
        (high_risk, medium_risk, low_risk),
        ("High‑Risk Vessels", "Medium‑Risk", "Low‑Risk"),
    ):
        col.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-label'>{lbl}</div>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 🔵🔵 5‑B Risk‑Home » simple pie‑chart of risk labels (after KPI) -------------
# ---------------------------------------------------------------------------
    fig_pie = px.pie(
        risk_table_display, names="Risk Label", title="Fleet Risk Breakdown"
    )
    fig_pie.update_traces(textinfo="label+percent").update_layout(showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------------------------
# 🔵🔵 5‑C Risk‑Entity » Add bar chart for selected entity df  -----------------
# ---------------------------------------------------------------------------
# 📍 After `st.dataframe(entity_df, ...)` include:

if entity_df is not None and not entity_df.empty:
    # expect a column named e.g. "Overall Score" or similar; fallback to first numeric
    num_cols = entity_df.select_dtypes(include=[np.number]).columns
    score_col = "Overall Score" if "Overall Score" in num_cols else num_cols[0]
    fig_ent = px.bar(
        entity_df.reset_index(),
        x=entity_df.index.name or "Entity",
        y=score_col,
        title=f"{sel_entity} – Risk Scores",
    )
    st.plotly_chart(add_bar_colours(fig_ent), use_container_width=True)

# ---------------------------------------------------------------------------
# 🔵🔵 5‑D Risk‑Hist » Line chart of historical score for selected vessel ------
# ---------------------------------------------------------------------------
# 📍 Within risk_hist, after you show the metrics for `sel_vessel11`, add:

if sel_vessel11 != "None":
    hist_list = vessel_historical_score_mapping11.get(sel_vessel11, [])
    if hist_list:
        hist_df = pd.DataFrame(hist_list, columns=["Date", "Score"])
        hist_df.sort_values("Date", inplace=True)
        fig_hist = px.line(hist_df, x="Date", y="Score", markers=True,
                           title="Historical Risk Trend")
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# 🔵🔵 5‑E Trend tab » Authority bar already there – just recolour -------------
# ---------------------------------------------------------------------------
#  After each px.bar() creation use helper to apply our palette
#  Example replacement for first bar chart:
#      fig = add_bar_colours(fig)

# ---------------------------------------------------------------------------
# 🔵🔵 5‑F Recommendations » No change – style already injected  ---------------
# ---------------------------------------------------------------------------

#  END  (no further edits needed – original logic preserved)
# ---------------------------------------------------------------------------

# NOTE:
# • Make sure to `import numpy as np` at the top (added).
# • We didn’t paste your entire original file here to avoid noise; copy‑paste
#   the marked 🔵🔵 blocks into the corresponding locations of your source code.
