#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:04:22 2025

@author: sasidharankumar
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
# from pathlib import Path
from util import prepare_inspection_data,prepare_analysis_data,generate_entity_analysis_results,get_vessel_historical_score_mapping
from util import get_vessel_change_scores,get_final_risk_score_mapping,get_vessel_segments,get_access_token,get_deficiency_df_snowflake
from trends_util import prepare_trend_analysis_data
from recommendations_util import load_recommendations_data,create_indexes_and_embeddings,prepare_recommendations_data,generate_open_defect_recommendations
from incidents_util import prepare_incidents_data
import plotly.express as px
import constants
import numpy as np
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="PSC Risk Dashboard",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)



# header_col1, header_col2 = st.columns([0.12, 0.88])
# with header_col1:
#     st.image("https://raw.githubusercontent.com/streamlit/branding/master/logos/mark/streamlit-mark-primary.png", width=70)
# with header_col2:
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


# ------------------------------------------------------------------
# 1.  DATA LOADING (cached so it runs once at start‑up)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    """Load all required CSV / Excel sheets into DataFrames."""
    # --- core inspection & scoring files --------------------------
    
    
    print("Loading Data......")
    #inspection_data = pd.read_csv("Synergy PSC Inspection.csv")
    access_token = get_access_token()
    inspection_data = get_deficiency_df_snowflake(access_token)
    inspection_data = prepare_inspection_data(inspection_data)
    
    incidents_imo_subset,incidents_owners_subset,incidents_flag_subset,incidents_manager_subset = prepare_incidents_data()
    generic_factors_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Vessel Generic Factor")
    dynamic_factors_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Ship Dynamic Factors")
    #owner_profile_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Registered Owner Profile")
    #manager_profile_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Ship Management Profile")
    #shipyard_profile_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Shipyard Profile")
    #engine_profile_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Engine Profile")
    #class_profile_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Vessel Class Profile")
    #flag_profile_data = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx",sheet_name="Vessel Flag Profile")
    psc_codes_scoring_data = pd.read_csv("PSC_Codes_Scored.csv")
    psc_codes_scoring_data["Code"] = psc_codes_scoring_data["Code"].astype(str)
    
    analysis_df = prepare_analysis_data(inspection_data,psc_codes_scoring_data,generic_factors_data)
    
    current_risk_df = pd.read_excel("Grounding Risk Scores.xlsx",sheet_name="Grounding Risk")
    
    # inspection = pd.read_csv("Synergy PSC Inspection.csv")
    # generic    = pd.read_excel(
    #     "PSC Risk Generic and Dynamic Factors.xlsx",
    #     sheet_name="Vessel Generic Factor",
    # )
    # dynamic    = pd.read_excel(
    #     "PSC Risk Generic and Dynamic Factors.xlsx",
    #     sheet_name="Ship Dynamic Factors",
    # )
    # psc_scores = pd.read_csv("PSC_Codes_Scored.csv")
    
    print("Running Load Data......")
    
    
    return {
        "inspection": inspection_data,
        "generic"   : generic_factors_data,
        "dynamic"   : dynamic_factors_data,
        "psc_scores": psc_codes_scoring_data,
        "analysis_df_master":analysis_df,
        "dynamic_factors_data":dynamic_factors_data,
        "current_risk_df":current_risk_df,
        "incidents_imo_subset":incidents_imo_subset,
        "incidents_owners_subset":incidents_owners_subset,
        "incidents_flag_subset":incidents_flag_subset,
        "incidents_manager_subset":incidents_manager_subset
    }


internal_checklist,external_checklist,open_defects,deficiency_codes,deficiency_codes_category_df,deficiency_codes_sub_category_df,psc_category_recommenders = load_recommendations_data()
internal_checklist_items,external_checklist_items,category_items,subcategory_items,category_desc_items,subcategory_desc_items = prepare_recommendations_data(internal_checklist,external_checklist,deficiency_codes)

# ------------------------------------------------------------------
# 2.  ANALYSIS PLACE‑HOLDERS  (fill in with your model logic)
# ------------------------------------------------------------------

def run_entity_analysis(df_filtered: pd.DataFrame):
    """Return a DF with entity‑level risk scores (placeholder)."""
    
    dataframe_mappings,score_mappings = generate_entity_analysis_results(df_filtered)
    # TODO: replace with real computation
    return dataframe_mappings,score_mappings

def run_historical_analysis(df_filtered: pd.DataFrame):
    """Return vessel x historical‑risk DataFrame (placeholder)."""
    vessel_historical_score_mapping = get_vessel_historical_score_mapping(df_filtered)
    return vessel_historical_score_mapping

def run_change_analysis(dynamic_factors_data):
    """Return vessel change‑risk DataFrame (placeholder)."""
    vessel_change_score_mapping = get_vessel_change_scores(dynamic_factors_data)
    return vessel_change_score_mapping

def aggregate_risk_scores(*dfs) -> pd.DataFrame:
    """Merge individual risk DataFrames into one master table.
    Expected index = IMO, columns = risk categories + Overall + Label.
    """
    # TODO: implement proper merge logic
    if not dfs:
        return pd.DataFrame()
    base = dfs[0].copy()
    for d in dfs[1:]:
        base = base.join(d, how="outer")
    return base
@st.cache_data(show_spinner=True)
def run_overall_analysis(analysis_df,unique_vessels,dynamic_factors_data):
    dataframe_mappings,score_mappings = run_entity_analysis(analysis_df)
    vessel_historical_score_mapping,vessel_summary_statistics = run_historical_analysis(analysis_df)
    vessel_change_score_mapping = run_change_analysis(dynamic_factors_data)
    st.session_state["vessel_change_score_mapping"] = vessel_change_score_mapping
    score_mappings["vessel_historical_score_mapping"] = vessel_historical_score_mapping
    score_mappings["vessel_change_score_mapping"] = vessel_change_score_mapping
    st.session_state["vessel_summary_statistics"] = vessel_summary_statistics
    vessel_risk_scores_final = get_final_risk_score_mapping(unique_vessels,analysis_df,score_mappings)
    segmented = get_vessel_segments(vessel_risk_scores_final)
    vessel_risk_scores_final["Risk Label"] = segmented["Risk Label"]
    print("Running Overall Analysis.......")
    return score_mappings,vessel_risk_scores_final,segmented



# ------------------------------------------------------------------
# 3.  SMALL UI HELPERS
# ------------------------------------------------------------------

def radar_plot(row: pd.Series):
    """Return a Plotly radar chart for one vessel row."""
    categories = [c for c in row.index if c.endswith("Score")]
    values     = [row[c] for c in categories]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]],
                                  theta=categories + [categories[0]],
                                  fill="toself",
                                  name="Risk"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                      showlegend=False,
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Map IMO → "Name (IMO)" for display
@st.cache_data
def build_vessel_name_map(dynamic_df: pd.DataFrame):
    return dict(zip(dynamic_df["IMO No."], dynamic_df["Vessel Name "]))


def format_vessel_option(imo_number):                                                                                                                                                                         
    """Format vessel option for selectbox display"""    
    #print(imo_number)
    #print(st.session_state.vessel_name_mapping)
    #print("--"*50)
    if(str(imo_number)=="None"):
        return "None"
    
    if str(imo_number) in st.session_state.vessel_name_mapping:                                                                                                                                                    
        vessel_name = st.session_state.vessel_name_mapping[str(imo_number)]
        return f"{vessel_name} (IMO: {imo_number})"
    elif int(imo_number) in st.session_state.vessel_name_mapping:
        vessel_name = st.session_state.vessel_name_mapping[int(imo_number)]
        return f"{vessel_name} (IMO: {imo_number})"
    else:                                                                                                                                                                                                     
        return f"IMO: {imo_number}"  
    

def load_vessel_name_mapping(dynamic_factors_df):                                                                                                                                                                               
    """Load vessel name mapping from PSC Risk Generic and Dynamic Factors.xlsx"""                                                                                                                             
    try:                                                                                                                                                                                                      
        # dynamic_factors_df = pd.read_excel("PSC Risk Generic and Dynamic Factors.xlsx", sheet_name="Ship Dynamic Factors")   
        # print(dynamic_factors_df.head())
        # print(dynamic_factors_df.columns)
        if 'IMO No.' in dynamic_factors_df.columns and 'Vessel Name ' in dynamic_factors_df.columns:                                                                                                            
            # print("Creating the Name mapping.....")
            # Create mapping from IMO to vessel name                                                                                                                                                          
            mapping = dict(zip(dynamic_factors_df['IMO No.'], dynamic_factors_df['Vessel Name ']))                                                                                                              
            st.session_state.vessel_name_mapping = mapping    
            # print(mapping)
            return True                                                                                                                                                                                       
    except Exception as e:                                                                                                                                                                                    
        st.error(f"Error loading vessel name mapping: {e}")                                                                                                                                                   
        return False                                                                                                                                                                                          
    return False 

@st.cache_resource(show_spinner=True)
def get_checklist_indexes():
    return create_indexes_and_embeddings(
        internal_checklist_items, external_checklist_items,
        category_items, subcategory_items,
        category_desc_items, subcategory_desc_items,
    )

index_internal, index_external = get_checklist_indexes()



# ------------------------------------------------------------------
# 4.  MAIN APP  -----------------------------------------------------
# ------------------------------------------------------------------

data = load_data()

inspection_data = data["inspection"]
psc_codes_scoring_data = data["psc_scores"]
current_risk_df = data["current_risk_df"]
analysis_df = data["analysis_df_master"].copy()
st.session_state["analysis_df"] = analysis_df
dynamic_factors_data = data["dynamic_factors_data"].copy()
load_vessel_name_mapping(dynamic_factors_data)
unique_vessels = pd.unique(analysis_df["IMO_NO"])

incidents_imo_subset = data["incidents_imo_subset"]
incidents_owners_subset = data["incidents_owners_subset"]
incidents_flag_subset = data["incidents_flag_subset"]
incidents_manager_subset = data["incidents_manager_subset"]



# 👉  if you have a pre‑cleaned `analysis_df` in your module, import it
authorities = ["None"]+sorted(analysis_df["AUTHORITY"].dropna().unique())
entities = ["Owners","Yard","Flag","Class","Main Engine Make","Main Engine Model","Manager"]

# # Vessel selector
# vessel_map = build_vessel_name_map(dynamic_factors_data)
# vessel_ids = list(dynamic_factors_data["IMO No."].values)
# fmt = lambda imo: f"{vessel_map.get(imo, 'Vessel')} (IMO {imo})"


score_mappings,vessel_risk_scores_final,segmented = run_overall_analysis(analysis_df,unique_vessels,dynamic_factors_data)

st.session_state["risk_table"] = vessel_risk_scores_final
vessel_ids = vessel_risk_scores_final.index.tolist()                                                                                                                                                          
# selected_vessel = st.selectbox(                                                                                                                                                                       
#     "Select Vessel",                                                                                                                                                                                  
#     vessel_ids,                                                                                                                                                                                       
#     format_func=format_vessel_option                                                                                                                                                                  
# )

category_distribution,sub_category_distribution,text_analysis_df,gram_result_df = prepare_trend_analysis_data(inspection_data,psc_codes_scoring_data)
vessel_category_distribution = category_distribution["Vessel"]
authority_category_distribution = category_distribution["Authority"]

vessel_subcategory_distribution = sub_category_distribution["Vessel"]
authority_subcategory_distribution = sub_category_distribution["Authority"]

vessel_subcategory_distribution_time = sub_category_distribution["Vessel_Time"]
authority_subcategory_distribution_time = sub_category_distribution["Authority_Time"]

authorities_tr = ["None"]+list(authority_category_distribution.keys())
vessel_tr = ["None"]+list(vessel_category_distribution.keys())




# --------------------  TOP‑LEVEL TABS ------------------------------
tab_risk, tab_trend, tab_reco,tab_settings = st.tabs(["🛡 Risk", "📈 Trend", "💡 Recommendations", "⚙ Settings"])

# ====================  RISK  =======================================
with tab_risk:
    risk_home, risk_entity, risk_hist, risk_change, current_risk,incident_risk = st.tabs([
        "🏠 Home", "🏢 Entity Risk", "📜 Historical Risk", "🔀 Change Risk","🛡 Current Risk","🛠️ Incident Risk"])

    # --------------------------------------------------------------
    # 4‑A  Home sub‑tab
    # --------------------------------------------------------------
    with risk_home:
        st.header("Risk Overview")
        col1, col2, = st.columns(2)                                                                                                                                                                
                                                                                                                                                                                                              
        with col1:                                                                                                                                                                                            

            sel_auth = st.selectbox("Select Port Authority", authorities, index=0)
            
        with col2:
            # run_btn  = st.button("🔄 Run Analysis", type="primary")
            sel_vessel = st.selectbox(                                                                                                                                                                       
                "Select Vessel",                                                                                                                                                                                  
                vessel_ids,                                                                                                                                                                                       
                format_func=format_vessel_option                                                                                                                                                                  
            )

        # if run_btn or "analysis_df" in st.session_state:
        if(sel_auth):
            # ----------------  Filter once ------------------------

            if(sel_auth=="None"):
                df_auth = analysis_df
            else:
                df_auth = analysis_df[analysis_df["AUTHORITY"] == sel_auth]

            dataframe_mappings,score_mappings = run_entity_analysis(df_auth)
            vessel_historical_score_mapping,vessel_summary_statistics = run_historical_analysis(df_auth)
            st.session_state["vessel_summary_statistics"] = vessel_summary_statistics
            
            vessel_change_score_mapping = run_change_analysis(dynamic_factors_data)
            score_mappings["vessel_historical_score_mapping"] = vessel_historical_score_mapping
            score_mappings["vessel_change_score_mapping"] = vessel_change_score_mapping
            vessel_risk_scores_final = get_final_risk_score_mapping(unique_vessels,df_auth,score_mappings)
            segmented = get_vessel_segments(vessel_risk_scores_final)
            vessel_risk_scores_final["Risk Label"] = segmented["Risk Label"]
            st.session_state["risk_table"] = vessel_risk_scores_final
            risk_table = vessel_risk_scores_final#st.session_state["risk_table"]
            #else:
            #    risk_table = st.session_state["risk_table"]

          
            risk_table_display = risk_table.copy()
            #risk_table_display.index = risk_table_display.index.astype(str)
            # risk_table_display.index = risk_table_display.index.map(lambda imo: f"{imo_to_name.get(imo, '???')} ({imo})")
            risk_table_display["Vessel Name"] = risk_table_display.index.map(lambda imo: format_vessel_option(imo))
            # risk_table_display.reset_index(drop=True,inplace=True)
            risk_table_display.set_index("Vessel Name", inplace=True)
            with col1:
                st.header("Fleet Risks")
                st.dataframe(risk_table_display, use_container_width=True)                

        if(sel_vessel):

            # Table (filtered or full)
            if sel_vessel:
                view_df = risk_table.loc[[sel_vessel]]
            else:
                view_df = risk_table
                
            styled_view_df = view_df.copy()
            # styled_view_df.index = styled_view_df.index.astype(str)
            styled_view_df["Vessel Name"] = styled_view_df.index.map(lambda imo: format_vessel_option(imo))
            # styled_view_df.reset_index(drop=True,inplace=True)
            styled_view_df.set_index("Vessel Name", inplace=True)
            
            with col2:
                st.header("Vessel Risks")
                st.dataframe(styled_view_df.T, use_container_width=True,column_config={"IMO": st.column_config.TextColumn("IMO")})
            # Details for one vessel
            if sel_vessel:
                row = view_df.iloc[0]
                try:
                    baseline_risk = row.get('Overall Risk')
                    current_risk_vessel = current_risk_df[current_risk_df["IMO"]==sel_vessel]["Scorecalc"].values[0]
                    overall_risk_vessel = round((baseline_risk*0.4) + (current_risk_vessel*0.6),2)
                except:
                    current_risk_vessel = "NA"
                    overall_risk_vessel = "NA"
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.plotly_chart(radar_plot(row), use_container_width=True)
                with col2:
                    st.metric("Baseline Risk", value=f"{row.get('Overall Risk', 'N/A'):.1f}")
                    st.metric("Current Risk", value=f"{current_risk_vessel}")
                    st.metric("Overall Risk", value=f"{overall_risk_vessel}")
                    st.write("**Labels:**", row.get("Risk Label", "-"))
                st.header("Vessel Summary:")
                if(sel_auth=="None"):
                    sub_df = analysis_df[analysis_df["IMO_NO"]==sel_vessel].copy()
                else:
                    sub_df = analysis_df[(analysis_df["IMO_NO"]==sel_vessel) & (analysis_df["AUTHORITY"]==sel_auth)].copy()
                sub_df.sort_values(by="INSPECTION_FROM_DATE",inplace=True)
                sub_df.reset_index(drop=True,inplace=True)
                # "Issue Count":len(sub_df),
                # "Action Codes":np.unique(sub_df["ActionCodes"]),
                # "Issue Weights(History Based)":np.mean(sub_df["ISSUE_BASESCORE"]),
                # "Overall Severity":np.mean(sub_df["FINAL_SEVERITY"])
                vessel_summary_statistics = st.session_state["vessel_summary_statistics"]
                if(sel_vessel in vessel_summary_statistics):
                    # print(vessel_summary_statistics[sel_vessel]["Action Codes"])
                    vessel = vessel_summary_statistics[sel_vessel]

                    st.markdown(
                        """
                        <style>
                            .detail-card {
                                background-color: #f7fafd;
                                border-radius: 10px;
                                padding: 32px 24px 20px 24px;
                                margin-bottom: 22px;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.06);
                                border-left: 6px solid #1a7ef7;
                            }
                            .detail-header {
                                font-size: 28px;
                                font-weight: 700;
                                color: #1867c0;
                                margin-bottom: 8px;
                            }
                            .detail-label {
                                color: #555;
                                font-weight: 600;
                            }
                            .detail-value {
                                color: #111;
                                font-weight: 400;
                            }
                        </style>
                        """, unsafe_allow_html=True
                    )
                    
                    # st.markdown('<div class="detail-card">', unsafe_allow_html=True)
                    st.markdown('<div class="detail-header">Vessel Details</div>', unsafe_allow_html=True)
                    
                    if str(sel_vessel) in st.session_state.vessel_name_mapping:                                                                                                                                                    
                        vessel_name = st.session_state.vessel_name_mapping[str(sel_vessel)]
                    elif int(sel_vessel) in st.session_state.vessel_name_mapping:
                        vessel_name = st.session_state.vessel_name_mapping[int(sel_vessel)]
                    else:
                        vessel_name = "NA"
                    
                    if(len(sub_df)==0):
                        sub_df1 = analysis_df[analysis_df["IMO_NO"]==sel_vessel].copy()
                    else:
                        sub_df1 = sub_df
                    vessel_owner = sub_df1["R_OWNERS"].values[0]
                    vessel_yard = sub_df1["YARD"].values[0]
                    vessel_flag = sub_df1["FLAG_STATE"].values[0]
                    vessel_class = sub_df1["VESSEL_CLASS"].values[0]
                    vessel_age = sub_df1["AGE_OF_VESSEL"].values[0]

                        
                    
                    
                    
                    # You can also use columns to display two items per row for compactness
                    fields = [
                        ("Vessel Name", vessel_name),
                        ("Vessel Owner", vessel_owner),
                        ("Vessel Yard", vessel_yard),
                        ("Vessel Flag", vessel_flag),
                        ("Vessel Class", vessel_class),
                        # ("Vessel Manager", vessel.get("Vessel Manager", "")),
                        # ("Vessel ME Make", vessel.get("Vessel ME Make", "")),
                        # ("Vessel ME Model", vessel.get("Vessel ME Model", "")),
                        ("Age(Years)", vessel_age),
                        ("Issue Count", vessel.get("Issue Count", "")),
                        ("Action Codes", ", ".join(vessel.get("Action Codes", []))),
                        ("Overall Severity", vessel.get("Overall Severity", "")),
                        ("Issue Weights (History Based)", vessel.get("Issue Weights(History Based)", "")),

                    ]
                    
                    for i in range(0, len(fields), 2):
                        cols = st.columns(2)
                        for j, col in enumerate(cols):
                            if i + j < len(fields):
                                label, value = fields[i + j]
                                col.markdown(
                                    f"""<span class="detail-label">{label}:</span> 
                                        <span class="detail-value">{value}</span>""",
                                    unsafe_allow_html=True
                                )
                    st.markdown('</div>', unsafe_allow_html=True)
                    sub_df_display = sub_df.copy()
                    sub_df_display["IMO_NO"] = sub_df_display["IMO_NO"].astype(str)
                    st.dataframe(sub_df_display, use_container_width=True)
                    
                    

    # --------------------------------------------------------------
    # 4‑B  Empty placeholders for other sub‑tabs
    # --------------------------------------------------------------
    with risk_entity:
        col11, col12, col13, col14 = st.columns(4) 
        with col11:
            sel_entity = st.selectbox("Select Entity", entities, index=0)
        with col12:
            sel_auth1 = st.selectbox("Select Port Authority", authorities, index=0,key="entityauthority")
        

        if(sel_auth1=="None"):
            df_auth = analysis_df
        else:
            df_auth = analysis_df[analysis_df["AUTHORITY"] == sel_auth1]
            
        dataframe_mappings_sub,score_mappings_sub = run_entity_analysis(df_auth)
        # "Owners","Yard","Flag","Class","Main Engine Make","Main Engine Model","Manager"
        
        
        if(sel_entity=="Owners"):
            entity_df = dataframe_mappings_sub["owners_result"]
        elif(sel_entity=="Yard"):
            entity_df = dataframe_mappings_sub["yard_result"]
        elif(sel_entity=="Flag"):
            entity_df = dataframe_mappings_sub["flag_result"]
        elif(sel_entity=="Class"):
            entity_df = dataframe_mappings_sub["class_result"]
        elif(sel_entity=="Main Engine Make"):
            entity_df = dataframe_mappings_sub["make_result"]
        elif(sel_entity=="Main Engine Model"):
            entity_df = dataframe_mappings_sub["model_result"]
        elif(sel_entity=="Manager"):
            entity_df = dataframe_mappings_sub["manager_result"]
        else:
            entity_df=None
            
        st.dataframe(entity_df, use_container_width=True)
        # st.info("Entity‑level risk analytics will appear here.")

    with risk_hist:
        col21, col22= st.columns(2) 
        
        with col21:
            sel_auth11 =  st.selectbox("Select Port Authority", authorities, index=0,key="auth_hist11")
        
        if(sel_auth11):
            
            if(sel_auth11=="None"):
                df_auth11 = analysis_df
            else:
                df_auth11 = analysis_df[analysis_df["AUTHORITY"] == sel_auth11]
            
            vessel_historical_score_mapping11,vessel_summary_statistics11 = run_historical_analysis(df_auth11)
           
            # vessel_historical_score_mapping11_df = pd.DataFrame(vessel_historical_score_mapping11)
            vessel_historical_score_mapping11_df = pd.DataFrame(list(vessel_historical_score_mapping11.items()), columns=['Vessel', 'Historical Score'])
            vessel_historical_score_mapping11_df["Vessel"] = vessel_historical_score_mapping11_df["Vessel"].astype(str)
            unique_vessels11 = list(vessel_summary_statistics11.keys())
            with col22:
                sel_vessel11 = st.selectbox(                                                                                                                                                                       
                    "Select Vessel",                                                                                                                                                                                  
                    unique_vessels11,                                                                                                                                                                                       
                    format_func=format_vessel_option ,key="selves11"                                                                                                                                                                 
                )
            with col21:
                st.dataframe(vessel_historical_score_mapping11_df,use_container_width=True,column_config={"IMO": st.column_config.TextColumn("Vessel")})
            with col22:
                st.write("Issue Count:  "+str(vessel_summary_statistics11[sel_vessel11]["Issue Count"]))
                st.write("Action Codes:  "+str(",".join(vessel_summary_statistics11[sel_vessel11]["Action Codes"])))
                st.write("Issue Weights (History Based) :  "+str(vessel_summary_statistics11[sel_vessel11]["Issue Weights(History Based)"]))
                st.write("Overall Severity:  "+str(vessel_summary_statistics11[sel_vessel11]["Overall Severity"]))

    with risk_change:
        
        # vessel_change_score_mapping = st.session_state["vessel_change_score_mapping"]
        vessel_change_score_mapping = run_change_analysis(dynamic_factors_data)
        vessel_change_score_mapping_df = pd.DataFrame(list(vessel_change_score_mapping.items()), columns=['Vessel', 'Change Score'])
        unique_vessels31 = ["None"]+list(vessel_change_score_mapping.keys())
        # vessel_change_score_mapping_df["Vessel"] = vessel_change_score_mapping_df["Vessel"].astype(str)
        
        col301, col302= st.columns(2)
        col31, col32= st.columns(2) 
        with col301:
            st.header("Change Scores")
            sel_vessel31 = st.selectbox(                                                                                                                                                                       
                "Select Vessel",                                                                                                                                                                                  
                unique_vessels31,                                                                                                                                                                                       
                format_func=format_vessel_option ,key="selves31"                                                                                                                                                                 
            )
        with col302:
            st.header("Change History")
        if(sel_vessel31 and sel_vessel31!="None"):
            # print(sel_vessel31)
            with col31:
                vessel_change_score_mapping_df31 = vessel_change_score_mapping_df[vessel_change_score_mapping_df["Vessel"]==sel_vessel31]
                vessel_change_score_mapping_df31["Vessel"] = vessel_change_score_mapping_df31["Vessel"].astype(str)
                vessel_change_score_mapping_df31 = vessel_change_score_mapping_df31.T
                st.dataframe(vessel_change_score_mapping_df31,use_container_width=True,column_config={"IMO": st.column_config.TextColumn("Vessel")})
            with col32:
                dynamic_factors_data31 = dynamic_factors_data[dynamic_factors_data["IMO No."]==sel_vessel31]
                dynamic_factors_data31["IMO No."] = dynamic_factors_data31["IMO No."].astype(str)
                dynamic_factors_data31 = dynamic_factors_data31.T
                st.dataframe(dynamic_factors_data31,use_container_width=True)
        else:
            with col31:
                vessel_change_score_mapping_df_disp = vessel_change_score_mapping_df.copy()
                vessel_change_score_mapping_df_disp["Vessel"] = vessel_change_score_mapping_df_disp["Vessel"].astype(str)
                st.dataframe(vessel_change_score_mapping_df_disp,use_container_width=True,hide_index=True)
            with col32:
                dynamic_factors_data_display = dynamic_factors_data.copy().reset_index(drop=True)
                dynamic_factors_data_display["IMO No."] = dynamic_factors_data_display["IMO No."].astype(str)
                st.dataframe(dynamic_factors_data_display,use_container_width=True,hide_index=True)
            
    with current_risk:
        unique_vessels81 = list(pd.unique(current_risk_df["IMO"]))
        unique_risks81 = list(pd.unique(current_risk_df["kpi_item"]))
        
        col81, col82= st.columns(2)
        with col81:
            sel_vessel81 = st.selectbox(                                                                                                                                                                       
                "Select Vessel",                                                                                                                                                                                  
                unique_vessels81,                                                                                                                                                                                       
                format_func=format_vessel_option ,key="selves81"                                                                                                                                                                 
            )
            
        with col82:
            sel_risk81 = st.selectbox(                                                                                                                                                                       
                "Select Risk Category",                                                                                                                                                                                  
                unique_risks81,                                                                                                                                                                                       
                key="selrisk81"                                                                                                                                                                 
            )
            
        if(sel_vessel81 and sel_risk81):
            curr_risk_df_disp = current_risk_df[(current_risk_df["IMO"]==sel_vessel81) & (current_risk_df["kpi_item"]==sel_risk81)]
            curr_risk_df_disp["IMO"] = curr_risk_df_disp["IMO"].astype(str)
            st.dataframe(curr_risk_df_disp)
    
    with incident_risk:
        unique_vessel91 = ["None"]+list(pd.unique(incidents_imo_subset["IMO_NO"]))
        unique_owners91 = ["None"]+list(pd.unique(incidents_owners_subset["ACTUAL_OWNERS"]))
        unique_managers91 = ["None"]+list(pd.unique(incidents_manager_subset["MANAGER_GROUP"]))
        unique_flags91 = ["None"]+list(pd.unique(incidents_flag_subset["FLAG_STATE"]))
        col911, col912 = st.columns(2)
        col913,col914= st.columns(2)
        with col911:
            sel_vessel91 = st.selectbox(                                                                                                                                                                       
                "Select Vessel",                                                                                                                                                                                  
                unique_vessel91,                                                                                                                                                                                       
                format_func=format_vessel_option ,key="selves91"                                                                                                                                                                 
            )
            
            if(sel_vessel91=="None"):
                st.dataframe(incidents_imo_subset, hide_index=True)
            else:
                vessel_incident_df = incidents_imo_subset[incidents_imo_subset["IMO_NO"]==sel_vessel91]
                # vessel_incident_df = vessel_incident_df.groupby("IMO_NO").mean()
                st.dataframe(vessel_incident_df.groupby("IMO_NO").mean().T)
                st.dataframe(vessel_incident_df, hide_index=True)

                
        with col912:
            sel_owner91 = st.selectbox(                                                                                                                                                                       
                "Select Owner",                                                                                                                                                                                  
                unique_owners91,key="selowner91"                                                                                                                                                                                    
            )
            
            if(sel_owner91=="None"):
                st.dataframe(incidents_owners_subset, hide_index=True)
            else:
                vessel_owner_df = incidents_owners_subset[incidents_owners_subset["ACTUAL_OWNERS"]==sel_owner91]
                # vessel_incident_df = vessel_incident_df.groupby("IMO_NO").mean()
                st.dataframe(vessel_owner_df.groupby("ACTUAL_OWNERS").mean().T)
                st.dataframe(vessel_owner_df, hide_index=True)

        with col913:
            sel_manager91 = st.selectbox(                                                                                                                                                                       
                "Select Manager",                                                                                                                                                                                  
                unique_managers91,                                                                                                                                                                                       
                key="selmanager91"                                                                                                                                                                 
            )
            
            if(sel_manager91=="None"):
                st.dataframe(incidents_manager_subset, hide_index=True)
            else:
                vessel_manager_df = incidents_manager_subset[incidents_manager_subset["MANAGER_GROUP"]==sel_manager91]
                # vessel_incident_df = vessel_incident_df.groupby("IMO_NO").mean()
                st.dataframe(vessel_manager_df.groupby("MANAGER_GROUP").mean().T)
                st.dataframe(vessel_manager_df, hide_index=True)

        with col914:
            sel_flag91 = st.selectbox(                                                                                                                                                                       
                "Select Flag",                                                                                                                                                                                  
                unique_flags91 ,key="selflag91"                                                                                                                                                                 
            )
            
            if(sel_flag91=="None"):
                st.dataframe(incidents_flag_subset, hide_index=True)
            else:
                vessel_flag_df = incidents_flag_subset[incidents_flag_subset["FLAG_STATE"]==sel_flag91]
                # vessel_incident_df = vessel_incident_df.groupby("IMO_NO").mean()
                st.dataframe(vessel_flag_df.groupby("FLAG_STATE").mean().T)
                st.dataframe(vessel_flag_df, hide_index=True)

        

        
# ====================  TREND  ======================================
with tab_trend:
    
    col51, col52= st.columns(2)
    col521, col522= st.columns(2)
    col511, col512= st.columns(2)

    with col51:
        st.header("Authority Analysis")
        sel_auth51 =  st.selectbox("Select Port Authority", authorities_tr, index=0,key="auth_hist51")
    
    
    with col52:
        st.header("Vessel Analysis")
        sel_vessel51 = st.selectbox(                                                                                                                                                                       
            "Select Vessel",                                                                                                                                                                                  
            vessel_tr,                                                                                                                                                                                       
            format_func=format_vessel_option ,key="selves51"                                                                                                                                                                 
        )
    
    if(sel_auth51 and sel_auth51 !="None"):
        category_percentages_df = authority_category_distribution[sel_auth51]
        subcategory_percentages_df = authority_subcategory_distribution[sel_auth51]
        with col51:
            st.header("Category Distribution")
            fig = px.bar(
                category_percentages_df,
                x='Category',
                y='Percentage',
                title="Category Percentage Distribution",
                text='Percentage'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(category_percentages_df, use_container_width=True)
        with col521:
            st.header("Sub Category Distribution")
            fig = px.bar(
                subcategory_percentages_df,
                x='Category',
                y='Percentage',
                title="Sub Category Percentage Distribution",
                text='Percentage'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(subcategory_percentages_df, use_container_width=True)


        
    if(sel_vessel51 and sel_vessel51!="None"):
        vessel_category_percentages_df = vessel_category_distribution[sel_vessel51]
        vessel_subcategory_percentages_df = vessel_subcategory_distribution[sel_vessel51]
        with col52:
            st.header("Category Distribution")
            fig = px.bar(
                vessel_category_percentages_df,
                x='Category',
                y='Percentage',
                title="Category Percentage Distribution",
                text='Percentage'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(vessel_category_percentages_df, use_container_width=True)
        with col522:
            st.header("Sub Category Distribution")
            fig = px.bar(
                vessel_subcategory_percentages_df,
                x='Category',
                y='Percentage',
                title="Sub Category Percentage Distribution",
                text='Percentage'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(vessel_subcategory_percentages_df, use_container_width=True)
            
    
    # col511, col512= st.columns(2)
    
    if(sel_vessel51 and sel_vessel51!="None" and sel_auth51 and sel_auth51 !="None"):
        df_mix_category = (category_percentages_df.merge(vessel_category_percentages_df, on="Category", how="outer").fillna(0))
        df_mix_category.columns = ["Category", "Authority Trends", "Vessel Trends"]
        df_mix_category["Score"] = (df_mix_category["Authority Trends"] + df_mix_category["Vessel Trends"]) /2 # simple average
        # sort by descending score
        df_mix_category = df_mix_category.sort_values("Score", ascending=False)
        top_cats = df_mix_category.head(5)#["Category"]
        
        
        df_mix_sub_category = (subcategory_percentages_df.merge(vessel_subcategory_percentages_df, on="Category", how="outer").fillna(0))
        df_mix_sub_category.columns = ["Sub Category", "Authority Trends", "Vessel Trends"]
        df_mix_sub_category["Score"] = (df_mix_sub_category["Authority Trends"] + df_mix_sub_category["Vessel Trends"]) /2 # simple average
        # sort by descending score
        df_mix_sub_category = df_mix_sub_category.sort_values("Score", ascending=False)
        top_sub_cats = df_mix_sub_category.head(10)#["Category"]
        
        with col511:
            st.header(f"Top Category Trends for {sel_vessel51} - {sel_auth51} Combo:")
            st.dataframe(top_cats)
        
        with col512:
            st.header(f"Top Sub Category Trends for {sel_vessel51} - {sel_auth51} Combo:")
            st.dataframe(top_sub_cats)
    else:
        st.info("Select a Authority and Vessel to analyze their trends")
        
    
    # st.info("Trend analysis coming soon.")

# ====================  RECOMMENDATIONS  ============================
with tab_reco:
    trends_based, open_defect  = st.tabs([
        "Vessel/Authority Trend", "Open Defects"])
    
    vessel_ids_rec = ["None"] + list(pd.unique(open_defects["IMO_NO"].values))
    with trends_based:
        
        col711, col712= st.columns(2)
        
        with col711:
            # st.header("Authority Analysis")
            sel_auth712 =  st.selectbox("Select Port Authority", authorities_tr, index=0,key="auth_hist712")
        
        
        with col712:
            # st.header("Vessel Analysis")
            sel_vessel712 = st.selectbox(                                                                                                                                                                       
                "Select Vessel",                                                                                                                                                                                  
                vessel_tr,                                                                                                                                                                                       
                format_func=format_vessel_option,key="sel_vessel712"                                                                                                                                                               
            )

        
        
        if(sel_vessel712 and sel_vessel712!="None" and sel_auth712 and sel_auth712 !="None"):
            
            subcategory_percentages_df_rec = authority_subcategory_distribution_time[sel_auth712]
            vessel_subcategory_percentages_df_rec = vessel_subcategory_distribution_time[sel_vessel712]
            
            subcategory_percentages_df_rec = authority_subcategory_distribution_time[sel_auth712]
            vessel_subcategory_percentages_df_rec = vessel_subcategory_distribution_time[sel_vessel712]
            
            df_mix_sub_category_rec = (subcategory_percentages_df_rec.merge(vessel_subcategory_percentages_df_rec, on="Category", how="outer").fillna(0))
            df_mix_sub_category_rec.columns = ["Sub Category", "Authority Trends", "Vessel Trends"]
            df_mix_sub_category_rec["Score"] = ((df_mix_sub_category_rec["Authority Trends"] *0.7) + (df_mix_sub_category_rec["Vessel Trends"])*0.3) # simple average
            # sort by descending score
           # ── Sort, show the ranking table ──────────────────────────────────────────────
            df_mix_sub_category_rec = df_mix_sub_category_rec.sort_values("Score",
                                                                          ascending=False)
            top_sub_cats_rec = df_mix_sub_category_rec.head(10)

            
            # ── Global look & feel tweaks  (inject once, early in your script) ───────────
            st.markdown(
                """
                <style>
                    /* page background (Streamlit default is already white) */
                    body, .stApp { background: #ffffff; }
            
                    /* simple card widget */
                    .psc-card        { border: 1px solid #e6e6e6;
                                       border-radius: 6px;
                                       margin: 1rem 0;
                                       box-shadow: 0 2px 4px rgba(0,0,0,.05); }
            
                    /* blue header strip */
                    .psc-card-header { background: #0c5db5;      /* adjust shade here */
                                       color: #ffffff;
                                       padding: 8px 14px;
                                       font-weight: 600;
                                       border-radius: 6px 6px 0 0;
                                       font-size: 16px; }
            
                    /* recommendation body */
                    .psc-card-body   { padding: 12px 14px;
                                       color: #333333;
                                       line-height: 1.45; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            # ── Render each recommendation as a card ─────────────────────────────────────
            for subcat in top_sub_cats_rec["Sub Category"].values:
                
                # print(subcat)
                # print(psc_codes_scoring_data)
                
                # subcattxt = psc_codes_scoring_data[psc_codes_scoring_data["Code"]==subcat].values[0]
                # print(subcattxt)
                
                rec_series = psc_category_recommenders.loc[
                    psc_category_recommenders["PSC Item Title"] == subcat, "Recommendation"
                ]
            
                # print(rec_series)
                if not rec_series.empty:
                    try:
                        # print(rec_series.iloc[0])
                        # print("---"*50)
                        rec_text = rec_series.iloc[0].encode('latin1').decode('utf-8')
                    except:
                        continue
            
                    st.markdown(
                        f"""
                        <div class="psc-card">
                            <div class="psc-card-header">{subcat}</div>
                            <div class="psc-card-body">{rec_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("Click Here View Trends"):
                st.dataframe(top_sub_cats_rec, use_container_width=True)
            
            
            # st.header("Recommendations")
    with open_defect:
        col611, col612= st.columns(2)
        
        with col611:
            sel_vessel61 = st.selectbox(                                                                                                                                                                       
                "Select Vessel",                                                                                                                                                                                  
                vessel_ids_rec,                                                                                                                                                                                       
                format_func=format_vessel_option,key="sel_vessel61"                                                                                                                                                               
            )
            
        if(sel_vessel61):
            # index_internal,index_external = create_indexes_and_embeddings(internal_checklist_items,external_checklist_items,category_items,subcategory_items,category_desc_items,subcategory_desc_items)
            deficiency_recommendations = generate_open_defect_recommendations(internal_checklist,external_checklist,open_defects,index_internal,index_external,sel_vessel61)
            
        # col621, col622= st.columns(2)
        st.markdown("<h2 style='margin-bottom: 18px;'>Live Defects & Recommendations basis Checklist</h2>", unsafe_allow_html=True)
    
        for defect, chks in deficiency_recommendations.items():
            
            internal_chks = deficiency_recommendations[defect]["internal"]
            ichkarr = []
            echkarr = []
            for ichk in internal_chks:
                ichkarr.append(ichk[0])
            
            external_chks = deficiency_recommendations[defect]["external"]
            for echk in external_chks:
                echkarr.append(echk[0])
            
            # for section, items in recommendations.items():
            st.markdown("""<style>
                    .custom-bullet {
                        margin-bottom: 10px;
                        padding: 12px 18px;
                        background: #f7fafd;
                        border-radius: 7px;
                        border-left: 5px solid #1a7ef7;
                        font-size: 1.0em;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                    }
                    .custom-bullet1 {
                        margin-bottom: 10px;
                        padding: 12px 18px;
                        background: #f7fafd;
                        border-radius: 7px;
                        border-left: 5px solid green;
                        font-size: 1.0em;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
                    }
                </style>
            """, unsafe_allow_html=True)
            
            with st.expander(f"🛠️ {defect}", expanded=False):  # Add emoji for a modern touch
                for item in ichkarr[:1]:
                    st.markdown(
                        f"<div class='custom-bullet'>{item}</div>",
                        unsafe_allow_html=True
                    )
                for item in echkarr[:1]:
                    st.markdown(
                        f"<div class='custom-bullet1'>{item}</div>",
                        unsafe_allow_html=True
                    )
            
# ====================  SETTINGS  =====================================
with tab_settings:
    st.markdown("## Global Settings & Weights")

    # ---------- helper funcs -----------------------------------------
    def _to_df(d, k1="Key", k2="Value"):
        return pd.DataFrame(list(d.items()), columns=[k1, k2])

    def _from_df(df):
        return {str(r[0]).strip(): r[1] for r in df.itertuples(index=False)
                if pd.notna(r[0])}

    # ---------- editable tables --------------------------------------
    st.subheader("PSC weight map")
    weight_df = st.session_state.get(
        "weight_map_df", _to_df(constants.weight_map, "PSC Code", "Weight")
    )
    weight_df = st.data_editor(weight_df,
                               num_rows="dynamic",
                               use_container_width=True,
                               hide_index=True,
                               key="weight_editor")

    st.subheader("Entity → Issue‑Type map")
    ent_flat = {k: ", ".join(v) for k, v in constants.entity_issuetype_mapping.items()}
    ent_df = st.session_state.get("ent_map_df",
                                  _to_df(ent_flat, "Entity", "Issue Types"))
    ent_df = st.data_editor(ent_df,
                            num_rows="dynamic",
                            use_container_width=True,
                            hide_index=True,
                            key="entity_editor")

    st.subheader("Change‑event weights")
    change_df = st.session_state.get("change_weight_df",
                                     _to_df(constants.change_weight_mapping,
                                            "Change Event", "Weight"))
    change_df = st.data_editor(change_df,
                               num_rows="dynamic",
                               use_container_width=True,
                               hide_index=True,
                               key="change_editor")

    st.subheader("Top‑level risk‑weights")
    risk_df = st.session_state.get("risk_weight_df",
                                   _to_df(constants.RISK_WEIGHTS,
                                          "Component", "Weight"))
    risk_df = st.data_editor(risk_df,
                             num_rows="dynamic",
                             use_container_width=True,
                             hide_index=True,
                             key="risk_editor")

    # ---------- sliders for scalar parameters ------------------------
    st.markdown("### Tuning constants")

    WACTION_val = st.slider(
        "⚖ WACTION (weight assigned to Action Codes)",
        min_value=0.0, max_value=1.0, step=0.01,
        value=float(constants.WACTION)
    )

    ISSUE_HALF_LIFE_val = st.slider(
        "Half‑life for historical issues (days)",
        50, 1000, int(constants.ISSUE_HALF_LIFE), step=10
    )

    HALF_LIFE_CHANGES_val = st.slider(
        "Half‑life for dynamic‑factor changes (days)",
        50, 1000, int(constants.HALF_LIFE_CHANGES), step=10
    )

    st.markdown("#### Baseline / Deviation / Severity split (must sum to 100)")
    col_b, col_d, col_s = st.columns(3)
    with col_b:
        BASELINE_val = st.number_input("Baseline %", 0, 100,
                                       int(constants.BASELINE_RISK_WEIGHT), 1)
    with col_d:
        DEVIATION_val = st.number_input("Deviation %", 0, 100,
                                        int(constants.DEVIATION_RISK_WEIGHT), 1)
    with col_s:
        SEVERITY_val = st.number_input("Severity %", 0, 100,
                                       int(constants.SEVERITY_RISK_WEIGHT), 1)

    total_100 = BASELINE_val + DEVIATION_val + SEVERITY_val

    if total_100 != 100:
        st.warning(f"⚠ The three percentages must add to 100 (current: {total_100}).")

    # ---------- buttons ----------------------------------------------
    c1, c2 = st.columns(2)
    with c1:
        save_disabled = total_100 != 100
        if st.button("💾 Save & Recalculate", type="primary",
                     disabled=save_disabled):
            # 1. Update scalar constants
            constants.WACTION              = WACTION_val
            constants.ISSUE_HALF_LIFE      = ISSUE_HALF_LIFE_val
            constants.lambda_val           = np.log(2) / constants.ISSUE_HALF_LIFE
            constants.HALF_LIFE_CHANGES    = HALF_LIFE_CHANGES_val
            constants.lambda_val_dynamic_factors = (
                np.log(2) / constants.HALF_LIFE_CHANGES
            )
            constants.BASELINE_RISK_WEIGHT  = BASELINE_val
            constants.DEVIATION_RISK_WEIGHT = DEVIATION_val
            constants.SEVERITY_RISK_WEIGHT  = SEVERITY_val

            # 2. Push table edits back into constants
            constants.weight_map               = _from_df(weight_df)
            constants.entity_issuetype_mapping = {
                k: [s.strip() for s in v.split(",") if s.strip()]
                for k, v in _from_df(ent_df).items()
            }
            constants.change_weight_mapping    = {
                k: int(v) for k, v in _from_df(change_df).items()
            }
            constants.RISK_WEIGHTS             = {
                k: float(v) for k, v in _from_df(risk_df).items()
            }

            # 3. Remember DataFrames for next visit
            st.session_state["weight_map_df"]    = weight_df
            st.session_state["ent_map_df"]       = ent_df
            st.session_state["change_weight_df"] = change_df
            st.session_state["risk_weight_df"]   = risk_df

            # 4. Clear caches & rebuild
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Settings saved – rebuilding dashboard …")
            st.rerun()

    with c2:
        if st.button("↩ Discard changes", type="secondary"):
            for k in ("weight_map_df", "ent_map_df",
                      "change_weight_df", "risk_weight_df"):
                st.session_state.pop(k, None)
            st.rerun()


            # st.markdown(
            #     f"""
            #     <div style="background-color:#f7fafd; border-left: 4px solid #1a7ef7; border-radius:7px; padding:16px 24px; margin-bottom:12px;">
            #         <strong style="font-size:1.1em;">{defect}</strong>
            #         <ul style="margin-top:7px;">
            #             {''.join([f"<li style='margin-bottom:5px;'>{item}</li>" for item in echkarr])}
            #         </ul>
            #     </div>
            #     """,
            #     unsafe_allow_html=True
            # )

# ------------------------------------------------------------------
# End of file
# ------------------------------------------------------------------
