#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 14:41:44 2025

@author: sasidharankumar
"""

from util import get_access_token,normalize_strings
from databricks import sql
import pandas as pd
from constants import external_incidents_sub_categories,external_incidents_sub2_categories,standardization_mapping,risk_score_mapping,lambda_val_incidents
import numpy as np
from datetime import datetime

today = datetime.now()
def get_incident_df_snowflake(access_token):
    connection = sql.connect(
                        server_hostname = "adb-4626041107022307.7.azuredatabricks.net",
                        http_path = "/sql/1.0/warehouses/de2adf4943b29216",
                        access_token = access_token)
    cursor = connection.cursor()
    #cursor.execute("select * from REPORTING_LAYER.QHSE.SYNERGYPOOL_VW_PSC_PERFORMANCE")#
    cursor.execute("select * from REPORTING_LAYER.QHSE.SYNERGYPOOL_VW_INCIDENT")
    results = cursor.fetchall()
    rows = results  # list of tuples
    cols = [c[0] for c in cursor.description]
    # 2. Build the DataFrame
    incidents_df_latest = pd.DataFrame(rows, columns=cols)
    return incidents_df_latest

def standardize_category(category, mapping_dict):
    category = category.strip().lower()
    return mapping_dict.get(category, category)
# Function to calculate severity score based on other columns if SMM_CH_15_CATEGORY is None
def infer_severity_from_columns(row):
    if pd.notna(row['SEVERITY_SCORE']):
        return row['SEVERITY_SCORE']
    if row['INCIDENT_CATEGORY'] in ['Significant_Incident', 'Other_Major_Incident']:
        return 80
    if row['SUB_CATEGORY'] in ['Fire', 'Structural_failure', 'Main engine failure', 'Loss of power - Blackout']:
        return 70
    if row['SUB_2_CATEGORY'] in ['Engine breakdown', 'Hull damage', 'Cargo pump failure']:
        return 60
    if row['DEFECT_TYPE'] == 'Hull & Machinery V3':
        return 50
    return 30
def get_incident_weight(row):
    occurrence_date = row['DATE_AND_TIME_OF_OCCURRENCE']

    # Attempt to convert to pandas Timestamp and handle errors
    try:
        # This handles None, NaT, and various date/datetime formats
        occurrence_date = pd.to_datetime(occurrence_date)
    except (ValueError, TypeError):
        # If conversion fails, treat as invalid and return 0
        return 0.25

    # Handle NaT values after conversion
    if pd.isna(occurrence_date):
        return 0.25

    age_d = (today - occurrence_date).days
    # Ensure age is not negative
    age_d = max(0, age_d)
    w = np.exp(-lambda_val_incidents * age_d)
    return w

def process_incidents_data(df):
    # Filter out external incidents
    vessel_related_df = df[~(
        df['SUB_CATEGORY'].isin(external_incidents_sub_categories) |
        df['SUB_2_CATEGORY'].isin(external_incidents_sub2_categories)
    )].copy()
    
    # Apply standardization
    vessel_related_df['STANDARDIZED_CATEGORY'] = vessel_related_df['SMM_CH_15_CATEGORY'].apply(
        lambda x: standardize_category(x, standardization_mapping) if pd.notnull(x) else None
    )

    # Assign numeric risk scores
    vessel_related_df['SEVERITY_SCORE'] = vessel_related_df['STANDARDIZED_CATEGORY'].map(risk_score_mapping)

    # Apply risk inference for missing values
    vessel_related_df['SEVERITY_SCORE'] = vessel_related_df.apply(infer_severity_from_columns, axis=1)
    vessel_related_df["ISSUE_WEIGHT"] = vessel_related_df.apply(get_incident_weight,axis=1)
    # We recompute the Severity of each and every incident based on the Weight calculated in the previous Step
    vessel_related_df["ISSUE_BASESCORE"] = vessel_related_df["SEVERITY_SCORE"]*vessel_related_df["ISSUE_WEIGHT"]
    return vessel_related_df
    


def prepare_incidents_data():
    access_token = get_access_token()
    incidents_df_latest = get_incident_df_snowflake(access_token)
    owners_group_mapping = normalize_strings(incidents_df_latest['OWNER_GROUP'], threshold=90)
    flag_state_mapping = normalize_strings(incidents_df_latest['FLAG_STATE'], threshold=90)
    owners_actual_mapping = normalize_strings(incidents_df_latest['ACTUAL_OWNERS'], threshold=90)
    manager_group_mapping = normalize_strings(incidents_df_latest['MANAGER_GROUP'], threshold=90)
    vessel_name_mapping = normalize_strings(incidents_df_latest['VESSEL_NAME'], threshold=90)
    vessel_type_mapping = normalize_strings(incidents_df_latest['VESSEL_TYPE'], threshold=90)
    dry_wet_mapping = normalize_strings(incidents_df_latest['"Dry/Wet"'], threshold=90)
    
    
    
    incidents_df_latest["OWNER_GROUP"] = incidents_df_latest["OWNER_GROUP"].map(owners_group_mapping)
    incidents_df_latest["FLAG_STATE"] = incidents_df_latest["FLAG_STATE"].map(flag_state_mapping)
    incidents_df_latest["ACTUAL_OWNERS"] = incidents_df_latest["ACTUAL_OWNERS"].map(owners_actual_mapping)
    incidents_df_latest["MANAGER_GROUP"] = incidents_df_latest["MANAGER_GROUP"].map(manager_group_mapping)
    incidents_df_latest["VESSEL_NAME"] = incidents_df_latest["VESSEL_NAME"].map(vessel_name_mapping)
    incidents_df_latest["VESSEL_TYPE"] = incidents_df_latest["VESSEL_TYPE"].map(vessel_type_mapping)
    incidents_df_latest['"Dry/Wet"'] = incidents_df_latest['"Dry/Wet"'].map(dry_wet_mapping)
    incidents_df_latest["IMO_NO"] = incidents_df_latest["IMO_NO"].astype(str).apply(lambda row:row.replace("\n","").replace("\t",""))
    
    vessel_related_incidents_df = process_incidents_data(incidents_df_latest)
    incidents_imo_subset = vessel_related_incidents_df[["IMO_NO","SEVERITY_SCORE","ISSUE_WEIGHT","ISSUE_BASESCORE"]]
    incidents_imo_subset = incidents_imo_subset[incidents_imo_subset["IMO_NO"]!=""]
    incidents_owners_subset = vessel_related_incidents_df[["ACTUAL_OWNERS","SEVERITY_SCORE","ISSUE_WEIGHT","ISSUE_BASESCORE"]]
    incidents_owners_subset = incidents_owners_subset[incidents_owners_subset["ACTUAL_OWNERS"]!=""]
    incidents_flag_subset = vessel_related_incidents_df[["FLAG_STATE","SEVERITY_SCORE","ISSUE_WEIGHT","ISSUE_BASESCORE"]]
    incidents_flag_subset = incidents_flag_subset[incidents_flag_subset["FLAG_STATE"]!=""]
    incidents_manager_subset = vessel_related_incidents_df[["MANAGER_GROUP","SEVERITY_SCORE","ISSUE_WEIGHT","ISSUE_BASESCORE"]]
    incidents_manager_subset = incidents_manager_subset[incidents_manager_subset["MANAGER_GROUP"]!=""]
    
    return incidents_imo_subset,incidents_owners_subset,incidents_flag_subset,incidents_manager_subset

    