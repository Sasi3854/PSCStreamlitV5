#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:12:31 2025

@author: sasidharankumar
"""

import pandas as pd
from rapidfuzz import process, fuzz
import re
from constants import DELIM_RE,NUM_RE, weight_map, entity_issuetype_mapping,lambda_val,WACTION,BUFFER_ZONE,max_severity,RISK_WEIGHTS
from constants import BASELINE_RISK_WEIGHT,DEVIATION_RISK_WEIGHT,SEVERITY_RISK_WEIGHT,lambda_val_dynamic_factors,change_weight_mapping
from constants import HIST_COLS,TECH_COLS,CHANGE_COL, RISK_COLS
import numpy as np
from datetime import datetime
from typing import Dict, Iterable, Tuple
import itertools
import warnings
import requests
import base64
from databricks import sql
import os

warnings.filterwarnings("ignore")
today = datetime.now()
def normalize_text(value):
    if pd.isna(value):
        return ''
    value = str(value).lower().strip()
    value = re.sub(r'[^a-z0-9\s]', ' ', value)  # Remove special characters (keep alphanum & spaces)
    value = re.sub(r'\s+', ' ', value)  # Normalize multiple spaces to single space
    return value

def normalize_strings(series, threshold=90):
    # Normalize casing & spaces
    series_norm = series.apply(normalize_text)
    #print(series_norm)
    # Get unique normalized strings
    unique_norms = series_norm.unique()

    # Fuzzy matching to group similar strings
    mapping_norm_to_canon = {}
    for val in unique_norms:
        if val in mapping_norm_to_canon or val.strip() == '':
            continue
        matches = process.extract(val, unique_norms, scorer=fuzz.token_sort_ratio, limit=None)
        for match_str, score, _ in matches:
            if score >= threshold:
                mapping_norm_to_canon[match_str] = val

    # Final mapping: original -> canonical normalized
    original_to_canonical = {}
    for original, norm in zip(series, series_norm):
        canonical_norm = mapping_norm_to_canon.get(norm, norm)
        original_to_canonical[original] = canonical_norm
    #print(original_to_canonical)
    return original_to_canonical

def parse_action_codes(raw):
    """
    Convert one raw PSC Action-Code cell (string or NaN) into
    a **sorted list of unique numeric codes (as strings)**.

    Examples
    --------
    '17/70'       → ['17','70']
    '40 AC'       → ['40']
    '10C'         → ['10']
    '17,50,70'    → ['17','50','70']
    np.nan / 'NA' → []
    """
    if pd.isna(raw):
        return []                     # keep NaNs empty

    # step 1 – unify delimiters to commas, strip spaces
    cleaned = DELIM_RE.sub(",", str(raw).upper()).replace(" ", "")

    # step 2 – pull out every numeric run
    codes = NUM_RE.findall(cleaned)

    # step 3 – drop leading zeros, deduplicate, sort
    codes = sorted({str(int(c)) for c in codes})   # int()→remove 00 padding
    return codes

def max_weight(code_list):
    return max((weight_map.get(c, 50) for c in code_list), default=np.nan)


def get_issue_weight(row):
    age_d = (today - row['INSPECTION_FROM_DATE']).days
    w= np.exp(-lambda_val * age_d)
    return w


def get_severity(row,psc_codes_scoring_data):
    psc_code = row["PSC_CODE"]
    rowmatch = psc_codes_scoring_data[psc_codes_scoring_data["Code"] == psc_code]

    if(rowmatch.empty):
      return 0
    return rowmatch["Severity"].values[0]

def get_classification(row,psc_codes_scoring_data):
    psc_code = row["PSC_CODE"]
    rowmatch = psc_codes_scoring_data[psc_codes_scoring_data["Code"] == psc_code]
    if(rowmatch.empty):
      return 0
    return rowmatch["Classification"].values[0]

def get_vessel_class(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
      return 0
    return rowmatch["Vessel Class"].values[0]

def get_me_make(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
      return 0

    me_make_model = rowmatch["Main Engine Spec (Make And Model)"].values[0]
    me_make = me_make_model.split(" x ")[0]
    return me_make

def get_me_model(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
      return 0
    me_make_model = rowmatch["Main Engine Spec (Make And Model)"].values[0]
    return me_make_model

def get_yard_name(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
        return row["YARD"]
    return rowmatch["Ship Yard Name"].values[0]

def get_yard_country(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
        return ""
    return rowmatch["Ship Yard Location"].values[0]

def get_vesseltype(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
        return row["VESSEL_TYPE"]
    return rowmatch["Vessel Type"].values[0]

def get_vessel_subtype(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
        return ""
    return rowmatch["Vessel Subtype"].values[0]

def get_vessel_age(row,generic_factors_data):
    imo_no = row["IMO_NO"]
    rowmatch = generic_factors_data[generic_factors_data["IMO No."] == imo_no]
    if(rowmatch.empty):
        return row["AGE_OF_VESSEL"]
    return rowmatch["Vessel Age"].values[0]

# Correct the matching function to handle the output correctly
def match_code(owner_name, owner_list):
    # Get the best match using fuzzywuzzy (only the match without unpacking score)
    matched_string = process.extractOne(owner_name, owner_list)
    if matched_string is None:
        return owner_name  # Return None if no match is found
    return matched_string[0]  # Return only the matched string (Owner name)

def calculate_risk_score(actual_issue_count,expected_issue_count,avg_severity,entity,entity_analysis_results):
    #STEP 1:
    #Max Expected Count (MEC): The worst-case expected issues (e.g., 20).
    #Max Severity (MS): The highest severity score (e.g., 5).
    max_issue_count_per_owner_weighted = entity_analysis_results["max_issue_count_per_owner_weighted"]
    max_issue_count_per_yard_weighted = entity_analysis_results["max_issue_count_per_yard_weighted"]
    max_issue_count_per_flag_weighted = entity_analysis_results["max_issue_count_per_flag_weighted"]
    max_issue_count_per_manager_weighted = entity_analysis_results["max_issue_count_per_manager_weighted"]
    max_issue_count_per_crew_weighted = entity_analysis_results["max_issue_count_per_crew_weighted"]
    max_issue_count_per_inspector_weighted = entity_analysis_results["max_issue_count_per_inspector_weighted"]
    max_issue_count_per_marine_manager_weighted = entity_analysis_results["max_issue_count_per_marine_manager_weighted"]
    max_issue_count_per_marine_superintendent_weighted = entity_analysis_results["max_issue_count_per_marine_superintendent_weighted"]
    max_issue_count_per_technical_manager_weighted = entity_analysis_results["max_issue_count_per_technical_manager_weighted"]
    max_issue_count_per_make_weighted = entity_analysis_results["max_issue_count_per_make_weighted"]
    max_issue_count_per_model_weighted = entity_analysis_results["max_issue_count_per_model_weighted"]
    max_issue_count_per_class_weighted = entity_analysis_results["max_issue_count_per_class_weighted"]
    if(entity=="R_OWNERS"):
        mec = max_issue_count_per_owner_weighted
    elif(entity=="YARD"):
        mec = max_issue_count_per_yard_weighted
    elif(entity=="FLAG_STATE"):
        mec = max_issue_count_per_flag_weighted
    elif(entity=="MANAGER_GROUP"):
        mec = max_issue_count_per_manager_weighted
    elif(entity=="NATIONALITY_OF_THE_CREW"):
        mec = max_issue_count_per_crew_weighted
    elif(entity=="INSPECTOR"):
        mec = max_issue_count_per_inspector_weighted
    elif(entity=="MARINE_MANAGER"):
        mec = max_issue_count_per_marine_manager_weighted
    elif(entity=="MARINE_SUPERINTENDENT"):
        mec = max_issue_count_per_marine_superintendent_weighted
    elif(entity=="TECHNICAL_MANAGER"):
        mec = max_issue_count_per_technical_manager_weighted
    elif(entity=="ME_MAKE"):
        mec = max_issue_count_per_make_weighted
    elif(entity=="ME_MODEL"):
        mec = max_issue_count_per_model_weighted
    elif(entity=="VESSEL_CLASS"):
        mec = max_issue_count_per_class_weighted
    else:
        return {"Error":"Please select a valid Entity Type for Analysis"}

    # STEP 2: Calculate a Baseine risk in terms of number of issues incurred by the owner
    # We assign 25% weightage purely in terms of issue count
    #baseline_risk = (expected_issue_count/mec) * 25
    baseline_risk = round(min((actual_issue_count/mec) * BASELINE_RISK_WEIGHT ,BASELINE_RISK_WEIGHT),2)

    # STEP 3: If a Vessel exceeds expected number of issues we amplify the underlying risk
    deviation_factor=(actual_issue_count/expected_issue_count)/2
    # if(actual_issue_count<=expected_issue_count):
    #     deviation_risk = deviation_factor*25
    # else:
    #     deviation_risk = 25 + ((actual_issue_count-expected_issue_count)/expected_issue_count)*25
    deviation_risk = round(min(deviation_factor*DEVIATION_RISK_WEIGHT,DEVIATION_RISK_WEIGHT),2)



    # STEP 4:
    severity_risk = round(min((avg_severity/max_severity)*SEVERITY_RISK_WEIGHT,SEVERITY_RISK_WEIGHT),2)
    if(np.isnan(severity_risk)):
        severity_risk = 15

    #print(f"Baseline Risk: {baseline_risk}, Deviation Risk: {deviation_risk}, Severity Risk: {severity_risk}")

    # STEP 5:
    total_risk = round(baseline_risk + deviation_risk + severity_risk,2)
    total_risk = min(total_risk,100)
    return total_risk,baseline_risk, deviation_risk, severity_risk
def prepare_analysis_data(inspection_data,psc_codes_scoring_data,generic_factors_data):
    analysis_df = inspection_data[["IMO_NO","AUTHORITY","NATURE_OF_DEFICIENCY","PSC_CODE","R_OWNERS","VESSEL_TYPE","YARD","AGE_OF_VESSEL","FLAG_STATE","DETENTION","MANAGER_GROUP","REFERENCE_CODE_1","NATIONALITY_OF_THE_CREW","ISSUE_WEIGHT","TECHNICAL_MANAGER","MARINE_SUPERINTENDENT","MARINE_MANAGER","INSPECTOR","INSPECTION_FROM_DATE"]].copy() #"SEVERITY"
    # analysis_df["SEVERITY"] = analysis_df.apply(get_severity,axis=1)
    
    analysis_df["SEVERITY"] = analysis_df.apply(lambda row:get_severity(row,psc_codes_scoring_data),axis=1)
    
    # analysis_df["ISSUE_CLASSIFICATION"] = analysis_df.apply(get_classification,axis=1)
    analysis_df["ISSUE_CLASSIFICATION"] = analysis_df.apply(
        lambda row: get_classification(row, psc_codes_scoring_data),  # Using a lambda
        axis=1
    )
    
    
    # analysis_df["VESSEL_CLASS"] = analysis_df.apply(get_vessel_class,axis=1)
    analysis_df["VESSEL_CLASS"] = analysis_df.apply(lambda row: get_vessel_class(row, generic_factors_data),axis=1)
    analysis_df["ME_MAKE"] = analysis_df.apply(lambda row: get_me_make(row,generic_factors_data),axis=1)
    analysis_df["ME_MODEL"] = analysis_df.apply(lambda row: get_me_model(row,generic_factors_data),axis=1)
    analysis_df["YARD"] = analysis_df.apply(lambda row: get_yard_name(row,generic_factors_data),axis=1)
    analysis_df["YARD_COUNTRY"] = analysis_df.apply(lambda row: get_yard_country(row,generic_factors_data),axis=1)
    analysis_df["VESSEL_TYPE"] = analysis_df.apply(lambda row: get_vesseltype(row,generic_factors_data),axis=1)
    analysis_df["VESSEL_SUBTYPE"] = analysis_df.apply(lambda row: get_vessel_subtype(row,generic_factors_data),axis=1)
    analysis_df["AGE_OF_VESSEL"] = analysis_df.apply(lambda row: get_vessel_age(row,generic_factors_data),axis=1)
    # analysis_df['INSPECTION_FROM_DATE'] = pd.to_datetime(analysis_df['INSPECTION_FROM_DATE'], format='%d-%m-%y')
    # analysis_df["SEVERITY"].replace(np.nan,4,inplace=True)
    
    r_owners_mapping = normalize_strings(analysis_df['R_OWNERS'], threshold=90)
    authority_mapping = normalize_strings(analysis_df["AUTHORITY"], threshold=90)
    vessel_type_mapping = normalize_strings(analysis_df["VESSEL_TYPE"], threshold=90)
    yard_mapping = normalize_strings(analysis_df["YARD"], threshold=90)
    yard_country_mapping = normalize_strings(analysis_df["YARD_COUNTRY"], threshold=90)
    flag_state_mapping = normalize_strings(analysis_df["FLAG_STATE"], threshold=90)
    manager_group_mapping = normalize_strings(analysis_df["MANAGER_GROUP"], threshold=90)
    crew_nationality_mapping = normalize_strings(analysis_df["NATIONALITY_OF_THE_CREW"], threshold=90)
    technical_manager_mapping = normalize_strings(analysis_df["TECHNICAL_MANAGER"], threshold=90)
    marine_superintendent_mapping = normalize_strings(analysis_df["MARINE_SUPERINTENDENT"], threshold=90)
    marine_manager_mapping = normalize_strings(analysis_df["MARINE_MANAGER"], threshold=90)
    inspector_mapping = normalize_strings(analysis_df["INSPECTOR"], threshold=90)
    make_mapping = normalize_strings(analysis_df["ME_MAKE"], threshold=90)
    model_mapping = normalize_strings(analysis_df["ME_MODEL"], threshold=90)
    vessel_type_mapping = normalize_strings(analysis_df["VESSEL_TYPE"], threshold=90)
    vessel_subtype_mapping = normalize_strings(analysis_df["VESSEL_SUBTYPE"], threshold=90)
    vessel_class_mapping = normalize_strings(analysis_df["VESSEL_CLASS"], threshold=90)
    
    
    analysis_df["R_OWNERS"] = analysis_df["R_OWNERS"].map(r_owners_mapping)
    analysis_df["AUTHORITY"] = analysis_df["AUTHORITY"].map(authority_mapping)
    analysis_df["VESSEL_TYPE"] = analysis_df["VESSEL_TYPE"].map(vessel_type_mapping)
    analysis_df["YARD"] = analysis_df["YARD"].map(yard_mapping)
    analysis_df["YARD_COUNTRY"] = analysis_df["YARD_COUNTRY"].map(yard_country_mapping)
    analysis_df["FLAG_STATE"] = analysis_df["FLAG_STATE"].map(flag_state_mapping)
    analysis_df["MANAGER_GROUP"] = analysis_df["MANAGER_GROUP"].map(manager_group_mapping)
    analysis_df["NATIONALITY_OF_THE_CREW"] = analysis_df["NATIONALITY_OF_THE_CREW"].map(crew_nationality_mapping)
    analysis_df["TECHNICAL_MANAGER"] = analysis_df["TECHNICAL_MANAGER"].map(technical_manager_mapping)
    analysis_df["MARINE_SUPERINTENDENT"] = analysis_df["MARINE_SUPERINTENDENT"].map(marine_superintendent_mapping)
    analysis_df["MARINE_MANAGER"] = analysis_df["MARINE_MANAGER"].map(marine_manager_mapping)
    analysis_df["INSPECTOR"] = analysis_df["INSPECTOR"].map(inspector_mapping)
    analysis_df["ME_MAKE"] = analysis_df["ME_MAKE"].map(make_mapping)
    analysis_df["ME_MODEL"] = analysis_df["ME_MODEL"].map(model_mapping)
    analysis_df["VESSEL_TYPE"] = analysis_df["VESSEL_TYPE"].map(vessel_type_mapping)
    analysis_df["VESSEL_SUBTYPE"] = analysis_df["VESSEL_SUBTYPE"].map(vessel_subtype_mapping)
    analysis_df["VESSEL_CLASS"] = analysis_df["VESSEL_CLASS"].map(vessel_class_mapping)
    
    # add a new column with the *list* of codes
    analysis_df["ActionCodes"] = analysis_df["REFERENCE_CODE_1"].apply(parse_action_codes)
    analysis_df["ACTIONCODE_SEVERITY"] = analysis_df["ActionCodes"].apply(max_weight)
    analysis_df['FINAL_SEVERITY'] = WACTION * analysis_df["ACTIONCODE_SEVERITY"] + ((1 - WACTION) * analysis_df['SEVERITY']*10)
    analysis_df["ISSUE_BASESCORE"] = analysis_df["FINAL_SEVERITY"]*analysis_df["ISSUE_WEIGHT"]
    return analysis_df




def prepare_inspection_data(inspection_data):
    inspection_data = inspection_data[~inspection_data["NATURE_OF_DEFICIENCY"].str.lower().isin( ["nil","nil findings","nil finding", "nil deficiency","nil observations"])]
    inspection_data = inspection_data[~inspection_data["PSC_CODE"].str.lower().isin( ["0","17","null","NA","Not mentioned in the report","CG005"])]
    inspection_data = inspection_data.dropna(subset=["PSC_CODE"])
    inspection_data['INSPECTION_FROM_DATE'] = pd.to_datetime(inspection_data['INSPECTION_FROM_DATE'], format='%d-%m-%y')
    inspection_data["ISSUE_WEIGHT"] = inspection_data.apply(get_issue_weight,axis=1)
    return inspection_data

def prepare_entity_analysis_data(analysis_df):
    owners_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","R_OWNERS","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    owners_analysis_df = owners_analysis_df[owners_analysis_df['R_OWNERS'] != '']
    
    yard_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","YARD","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    yard_analysis_df = yard_analysis_df[yard_analysis_df['YARD'] != '']
    
    flag_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","FLAG_STATE","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    flag_analysis_df = flag_analysis_df[flag_analysis_df['FLAG_STATE'] != '']
    
    manager_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","MANAGER_GROUP","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    manager_analysis_df = manager_analysis_df[manager_analysis_df['MANAGER_GROUP'] != '']
    
    crew_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","NATIONALITY_OF_THE_CREW","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    crew_analysis_df = crew_analysis_df[crew_analysis_df['NATIONALITY_OF_THE_CREW'] != '']
    
    inspector_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","INSPECTOR","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    inspector_analysis_df = inspector_analysis_df[inspector_analysis_df['INSPECTOR'] != '']
    
    marine_manager_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","MARINE_MANAGER","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    marine_manager_analysis_df = marine_manager_analysis_df[marine_manager_analysis_df['MARINE_MANAGER'] != '']
    
    marine_superintendent_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","MARINE_SUPERINTENDENT","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    marine_superintendent_analysis_df = marine_superintendent_analysis_df[marine_superintendent_analysis_df['MARINE_SUPERINTENDENT'] != '']
    
    technical_manager_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","TECHNICAL_MANAGER","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    technical_manager_analysis_df = technical_manager_analysis_df[technical_manager_analysis_df['TECHNICAL_MANAGER'] != '']
    
    make_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","ME_MAKE","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    make_analysis_df = make_analysis_df[make_analysis_df['ME_MAKE'] != '']
    
    model_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","ME_MODEL","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    model_analysis_df = model_analysis_df[model_analysis_df['ME_MODEL'] != '']
    
    class_analysis_df = analysis_df[["IMO_NO","AUTHORITY","FINAL_SEVERITY","VESSEL_CLASS","ISSUE_CLASSIFICATION","ISSUE_WEIGHT","ISSUE_BASESCORE"]].copy()
    class_analysis_df = class_analysis_df[class_analysis_df['VESSEL_CLASS'] != '']
    
    total_no_vessels = len(pd.unique(analysis_df["IMO_NO"]))
    total_no_owners = len(pd.unique(analysis_df["R_OWNERS"]))
    total_no_yards = len(pd.unique(analysis_df["YARD"]))
    total_no_flags = len(pd.unique(analysis_df["FLAG_STATE"]))
    total_no_managers = len(pd.unique(analysis_df["MANAGER_GROUP"]))
    total_no_crews = len(pd.unique(analysis_df["NATIONALITY_OF_THE_CREW"]))
    total_no_inspectors = len(pd.unique(analysis_df["INSPECTOR"]))
    total_no_marine_managers = len(pd.unique(analysis_df["MARINE_MANAGER"]))
    total_no_marine_superintendent = len(pd.unique(analysis_df["MARINE_SUPERINTENDENT"]))
    total_no_technical_managers = len(pd.unique(analysis_df["TECHNICAL_MANAGER"]))
    total_no_makes = len(pd.unique(analysis_df["ME_MAKE"]))
    total_no_models = len(pd.unique(analysis_df["ME_MODEL"]))
    total_no_classes = len(pd.unique(analysis_df["VESSEL_CLASS"]))
    total_no_authorities = len(pd.unique(analysis_df["AUTHORITY"]))
    total_defect_count = len(analysis_df)
    expected_issues_per_vessel = round(total_defect_count / total_no_vessels,2)
    total_weighted_issue_count = analysis_df["ISSUE_WEIGHT"].sum()
    

    max_issue_count_per_owner,max_issue_count_per_owner_weighted = round((total_defect_count/total_no_owners)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_owners)*BUFFER_ZONE)
    max_issue_count_per_yard,max_issue_count_per_yard_weighted = round((total_defect_count/total_no_yards)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_yards)*BUFFER_ZONE)
    max_issue_count_per_flag,max_issue_count_per_flag_weighted = round((total_defect_count/total_no_flags)*BUFFER_ZONE), round((total_weighted_issue_count/total_no_flags)*BUFFER_ZONE)
    max_issue_count_per_manager,max_issue_count_per_manager_weighted = round((total_defect_count/total_no_managers)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_managers)*BUFFER_ZONE)
    max_issue_count_per_crew,max_issue_count_per_crew_weighted = round((total_defect_count/total_no_crews)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_crews)*BUFFER_ZONE)
    max_issue_count_per_inspector,max_issue_count_per_inspector_weighted = round((total_defect_count/total_no_inspectors)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_inspectors)*BUFFER_ZONE)
    max_issue_count_per_marine_manager,max_issue_count_per_marine_manager_weighted = round((total_defect_count/total_no_marine_managers)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_marine_managers)*BUFFER_ZONE)
    max_issue_count_per_marine_superintendent,max_issue_count_per_marine_superintendent_weighted = round((total_defect_count/total_no_marine_superintendent)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_marine_superintendent)*BUFFER_ZONE)
    max_issue_count_per_technical_manager,max_issue_count_per_technical_manager_weighted = round((total_defect_count/total_no_technical_managers)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_technical_managers)*BUFFER_ZONE)
    max_issue_count_per_make,max_issue_count_per_make_weighted = round((total_defect_count/total_no_makes)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_makes)*BUFFER_ZONE)
    max_issue_count_per_model,max_issue_count_per_model_weighted = round((total_defect_count/total_no_models)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_models)*BUFFER_ZONE)
    max_issue_count_per_class,max_issue_count_per_class_weighted = round((total_defect_count/total_no_classes)*BUFFER_ZONE),round((total_weighted_issue_count/total_no_classes)*BUFFER_ZONE)
    
    return {
        # DataFrames
        "owners_analysis_df": owners_analysis_df,
        "yard_analysis_df": yard_analysis_df,
        "flag_analysis_df": flag_analysis_df,
        "manager_analysis_df": manager_analysis_df,
        "crew_analysis_df": crew_analysis_df,
        "inspector_analysis_df": inspector_analysis_df,
        "marine_manager_analysis_df": marine_manager_analysis_df,
        "marine_superintendent_analysis_df": marine_superintendent_analysis_df,
        "technical_manager_analysis_df": technical_manager_analysis_df,
        "make_analysis_df": make_analysis_df,
        "model_analysis_df": model_analysis_df,
        "class_analysis_df": class_analysis_df,

        # Summary counts/statistics
        "total_no_vessels": total_no_vessels,
        "total_no_owners": total_no_owners,
        "total_no_yards": total_no_yards,
        "total_no_flags": total_no_flags,
        "total_no_managers": total_no_managers,
        "total_no_crews": total_no_crews,
        "total_no_inspectors": total_no_inspectors,
        "total_no_marine_managers": total_no_marine_managers,
        "total_no_marine_superintendent": total_no_marine_superintendent,
        "total_no_technical_managers": total_no_technical_managers,
        "total_no_makes": total_no_makes,
        "total_no_models": total_no_models,
        "total_no_classes": total_no_classes,
        "total_no_authorities": total_no_authorities,
        "total_defect_count": total_defect_count,
        "expected_issues_per_vessel": expected_issues_per_vessel,
        "total_weighted_issue_count": total_weighted_issue_count,

        # Max per entity
        "max_issue_count_per_owner": max_issue_count_per_owner,
        "max_issue_count_per_owner_weighted": max_issue_count_per_owner_weighted,
        "max_issue_count_per_yard": max_issue_count_per_yard,
        "max_issue_count_per_yard_weighted": max_issue_count_per_yard_weighted,
        "max_issue_count_per_flag": max_issue_count_per_flag,
        "max_issue_count_per_flag_weighted": max_issue_count_per_flag_weighted,
        "max_issue_count_per_manager": max_issue_count_per_manager,
        "max_issue_count_per_manager_weighted": max_issue_count_per_manager_weighted,
        "max_issue_count_per_crew": max_issue_count_per_crew,
        "max_issue_count_per_crew_weighted": max_issue_count_per_crew_weighted,
        "max_issue_count_per_inspector": max_issue_count_per_inspector,
        "max_issue_count_per_inspector_weighted": max_issue_count_per_inspector_weighted,
        "max_issue_count_per_marine_manager": max_issue_count_per_marine_manager,
        "max_issue_count_per_marine_manager_weighted": max_issue_count_per_marine_manager_weighted,
        "max_issue_count_per_marine_superintendent": max_issue_count_per_marine_superintendent,
        "max_issue_count_per_marine_superintendent_weighted": max_issue_count_per_marine_superintendent_weighted,
        "max_issue_count_per_technical_manager": max_issue_count_per_technical_manager,
        "max_issue_count_per_technical_manager_weighted": max_issue_count_per_technical_manager_weighted,
        "max_issue_count_per_make": max_issue_count_per_make,
        "max_issue_count_per_make_weighted": max_issue_count_per_make_weighted,
        "max_issue_count_per_model": max_issue_count_per_model,
        "max_issue_count_per_model_weighted": max_issue_count_per_model_weighted,
        "max_issue_count_per_class": max_issue_count_per_class,
        "max_issue_count_per_class_weighted": max_issue_count_per_class_weighted,
    }

    
def analyze_entity(entity_df, entity_col,entity_analysis_results):
    analysis_results = []
    unique_entities = pd.unique(entity_df[entity_col])
    
    total_no_vessels,total_weighted_issue_count = entity_analysis_results["total_no_vessels"],entity_analysis_results["total_weighted_issue_count"]

    for entity in unique_entities:
        entity_risk_mapping = {}
        entity_issuetype = entity_issuetype_mapping[entity_col]
        entity_subset = entity_df[(entity_df[entity_col]==entity) & (entity_df["ISSUE_CLASSIFICATION"].isin(entity_issuetype))]
        total_weighted_issue_count_under_entity = entity_subset["ISSUE_WEIGHT"].sum()
        # total_weighted_issue_count
        total_vessel_under_entity = len(pd.unique(entity_subset["IMO_NO"]))
        if(total_vessel_under_entity==0):
            continue

        percentage_vessel_under_entity = total_vessel_under_entity / total_no_vessels
        expected_weighted_issue_count = round((total_weighted_issue_count * percentage_vessel_under_entity)*1.2,2)
        actual_weighted_issue_count = total_weighted_issue_count_under_entity
        #expected_issue_count = round((total_defect_count * percentage_vessel_under_entity)*1.2,2)
        #actual_issue_count = len(entity_subset)

        #original_severity = entity_subset["FINAL_SEVERITY"].values
        weighted_subset_severity = entity_subset["ISSUE_BASESCORE"].values
        #weighted_subset_severity = np.array(weighted_subset_severity)
        #avg_weighted_severity = round(np.mean(weighted_subset_severity[~np.isnan(weighted_subset_severity)]),2)
        avg_weighted_severity = round(np.mean(weighted_subset_severity[~np.isnan(weighted_subset_severity)]),2)
        #ISSUE_BASESCORE

        risk_score,baseline_risk, deviation_risk, severity_risk = calculate_risk_score(actual_weighted_issue_count,expected_weighted_issue_count,avg_weighted_severity,entity_col,entity_analysis_results)
        entity_risk_mapping["Risk"] = risk_score
        entity_risk_mapping["Baseline Risk"] = baseline_risk
        entity_risk_mapping["Severity Risk"] = severity_risk
        entity_risk_mapping["Deviation Risk"] = deviation_risk
        entity_risk_mapping["Entity Name"] = entity
        entity_risk_mapping["Entity Type"] = entity_col
        entity_risk_mapping["Actual Issue Count"] = actual_weighted_issue_count
        entity_risk_mapping["Expected Issue Count"] = expected_weighted_issue_count
        entity_risk_mapping["Average Weighted Severity"] = avg_weighted_severity
        entity_risk_mapping["Original Severity"] = round(np.mean(weighted_subset_severity[~np.isnan(weighted_subset_severity)]),2)#np.mean(original_severity)
        #print(f"Entity {entity_col}: {entity}, Actual Issue: {actual_weighted_issue_count}, Expected issue Count: {expected_weighted_issue_count}, Average Severity: {avg_weighted_severity}, Risk Score: {risk_score}")
        #print("--"*50)
        analysis_results.append(entity_risk_mapping)
    # analysis_results["Entity Name"] = entity
    # analysis_results["Entity Risk Mapping"] = entity_risk_mapping
    # analysis_results["Entity Type"] = entity_col
    # analysis_results["Actual Issue Count"] = actual_issue_count
    # analysis_results["Expected issue Count"] = expected_issue_count
    # analysis_results["Average Severity"] = avg_severity

    return analysis_results
    
def convert_entity_analysis_to_dataframe(analysis_results):
    analysis_results_df = pd.DataFrame(analysis_results)
    analysis_results_df.sort_values(by="Risk",inplace=True,ascending=False)
    analysis_results_df.reset_index(drop=True,inplace=True)
    return analysis_results_df

def generate_entity_analysis_results(analysis_df):
    
    
    entity_analysis_results = prepare_entity_analysis_data(analysis_df)
    
    
    owners_analysis_df,yard_analysis_df = entity_analysis_results["owners_analysis_df"],entity_analysis_results["yard_analysis_df"]
    flag_analysis_df,manager_analysis_df = entity_analysis_results["flag_analysis_df"],entity_analysis_results["manager_analysis_df"]
    make_analysis_df,model_analysis_df,class_analysis_df = entity_analysis_results["make_analysis_df"],entity_analysis_results["model_analysis_df"],entity_analysis_results["class_analysis_df"]
    
    
    # ENTITIES AVAILABLE= OWNERS, YARD, FLAG, MANAGER, MAKE, MODEL, CLASS, MARINE MANAGER, MARINE SUPERINTENDENT, TECHNICAL MANAGER, INSPECTOR
    analysis_results_owners = analyze_entity(owners_analysis_df, "R_OWNERS",entity_analysis_results)
    analysis_results_yard = analyze_entity(yard_analysis_df, "YARD",entity_analysis_results)
    analysis_results_flag = analyze_entity(flag_analysis_df, "FLAG_STATE",entity_analysis_results)
    analysis_results_manager = analyze_entity(manager_analysis_df, "MANAGER_GROUP",entity_analysis_results)
    # analysis_results_crew = analyze_entity(crew_analysis_df, "NATIONALITY_OF_THE_CREW")
    # analysis_results_inspector = analyze_entity(inspector_analysis_df, "INSPECTOR")
    # analysis_results_marine_manager = analyze_entity(marine_manager_analysis_df, "MARINE_MANAGER")
    # analysis_results_marine_superintendent = analyze_entity(marine_superintendent_analysis_df, "MARINE_SUPERINTENDENT")
    # analysis_results_technical_manager = analyze_entity(technical_manager_analysis_df, "TECHNICAL_MANAGER")
    analysis_results_make = analyze_entity(make_analysis_df, "ME_MAKE",entity_analysis_results)
    analysis_results_model = analyze_entity(model_analysis_df, "ME_MODEL",entity_analysis_results)
    analysis_results_class = analyze_entity(class_analysis_df, "VESSEL_CLASS",entity_analysis_results)
    owners_result = convert_entity_analysis_to_dataframe(analysis_results_owners)
    yard_result = convert_entity_analysis_to_dataframe(analysis_results_yard)
    flag_result = convert_entity_analysis_to_dataframe(analysis_results_flag)
    manager_result = convert_entity_analysis_to_dataframe(analysis_results_manager)
    # crew_result = convert_entity_analysis_to_dataframe(analysis_results_crew)
    # inspector_result = convert_entity_analysis_to_dataframe(analysis_results_inspector)
    # marine_manager_result = convert_entity_analysis_to_dataframe(analysis_results_marine_manager)
    # marine_superintendent_result = convert_entity_analysis_to_dataframe(analysis_results_marine_superintendent)
    # technical_manager_result = convert_entity_analysis_to_dataframe(analysis_results_technical_manager)
    make_result = convert_entity_analysis_to_dataframe(analysis_results_make)
    model_result = convert_entity_analysis_to_dataframe(analysis_results_model)
    class_result = convert_entity_analysis_to_dataframe(analysis_results_class)
    
    
    owners_risk_score_mapping = owners_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    yard_risk_score_mapping = yard_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    flag_risk_score_mapping = flag_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    manager_risk_score_mapping = manager_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    # crew_risk_score_mapping = crew_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    # inspector_risk_score_mapping = inspector_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    # marine_manager_risk_score_mapping = marine_manager_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    # marine_superintendent_risk_score_mapping = marine_superintendent_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    # technical_manager_risk_score_mapping = technical_manager_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    make_risk_score_mapping = make_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    model_risk_score_mapping = model_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    class_risk_score_mapping = class_result[["Entity Name","Risk"]].set_index('Entity Name')["Risk"].to_dict()
    score_mappings = {
        "owners_risk_score_mapping":owners_risk_score_mapping,
        "yard_risk_score_mapping":yard_risk_score_mapping,
        "flag_risk_score_mapping":flag_risk_score_mapping,
        "manager_risk_score_mapping":manager_risk_score_mapping,
        "make_risk_score_mapping":make_risk_score_mapping,
        "model_risk_score_mapping":model_risk_score_mapping,
        "class_risk_score_mapping":class_risk_score_mapping
    }
    
    dataframe_mappings = {
        "owners_result":owners_result,
        "yard_result":yard_result,
        "flag_result":flag_result,
        "manager_result":manager_result,
        "make_result":make_result,
        "model_result":model_result,
        "class_result":class_result
    }
    return dataframe_mappings,score_mappings


########################################################################################################################################################################
##########################################################################################################################################################
########################################################################################################################################################################




def vessel_risk(sub, E, w_cnt=0.6):
    λ=np.log(2)/90
    today = pd.Timestamp.today()
    age_d = (today - sub['ISSUE_DATE']).dt.days
    w= np.exp(-λ * age_d)

    Cw = w.sum()
    if Cw == 0:
        return 1

    Sw = (w * sub['SEVERITY']).sum()
    avg_sev = Sw / Cw

    A = 100 * min(1, Cw / E)
    B = 100 * (avg_sev / 10)

    score = w_cnt*A + (1-w_cnt)*B
    return float(np.clip(score, 1, 100))

def get_vessel_historical_score_mapping(analysis_df):
    
    unique_vessels = pd.unique(analysis_df["IMO_NO"])
    COUNT_WEIGHT = 0.6
    vessel_historical_score_mapping = {}
    total_defect_count = len(analysis_df)
    total_no_vessels = len(pd.unique(analysis_df["IMO_NO"]))
    expected_issues_per_vessel = round(total_defect_count / total_no_vessels,2)
    vessel_summary_statistics ={}
    for vessel in unique_vessels:
        sub_df = analysis_df[analysis_df["IMO_NO"]==vessel].copy()
        sub_df.sort_values(by="INSPECTION_FROM_DATE",inplace=True)
        sub_df.reset_index(drop=True,inplace=True)
    
        # We get the Weight of Each and Every Issue based on the age.
        # For Example Issue Raised today will have weight of 1 whereas issue raised 90 days(Half Life) ago will have a weight of 0.5
        sub_df["ISSUE_WEIGHT"] = sub_df.apply(get_issue_weight,axis=1)
        # We recompute the Severity of each and every issue based on the Weight calculated in the previous Step
        sub_df["ISSUE_BASESCORE"] = sub_df["FINAL_SEVERITY"]*sub_df["ISSUE_WEIGHT"]
        # We sum the Weights of each issue
        issue_weight_sum = round(sub_df["ISSUE_WEIGHT"].sum(),5)
        # We sum the recomputed severity of each issue
        weighted_sev_score_sum = sub_df["ISSUE_BASESCORE"].sum()
        # We compute the Severity score of the vessel by dividing the Weighted Severity sum by Weights Sum
        # This is going to give me average Weighted Severity
        avg_weighted_sev_score = round(weighted_sev_score_sum / issue_weight_sum,2)
    
        # issue_weight_sum represents the actual count of Issues Weighted by their recency -
        # If all issues were raised today this sum would be equal to actual count of issues
        # expected_issues_per_vessel - Represents total number of Issues expected per vessel based on historical context
        
        
        count_metric = round(100 * min(1, issue_weight_sum / expected_issues_per_vessel),2)
    
    
        #actual_count_of_issues_per_vessel = len(sub_df)
        #deviation = (actual_count_of_issues_per_vessel - expected_issues_per_vessel) / expected_issues_per_vessel
    
        historical_risk_score = round(COUNT_WEIGHT*count_metric + (1-COUNT_WEIGHT)*avg_weighted_sev_score,2)
        vessel_historical_score_mapping[vessel] = float(historical_risk_score)
        all_codes = list(itertools.chain.from_iterable(
            x if isinstance(x, list) else [x] for x in sub_df["ActionCodes"]
        ))
        
        # 2. Remove NaN or None if you want (optional)
        all_codes = [code for code in all_codes if pd.notnull(code)]
        
        # 3. Get unique elements (order-preserving)
        unique_action_codes = list(pd.unique(all_codes))
        vessel_summary_statistics[vessel] = {
            "Issue Count":len(sub_df),
            "Action Codes":unique_action_codes,
            "Issue Weights(History Based)":round(np.nanmean(sub_df["ISSUE_WEIGHT"].values),2),
            "Overall Severity":round(np.nanmean(sub_df["FINAL_SEVERITY"]),2)
        }
        #print("IMO:"+str(vessel)+", Weighted Severity Score:"+str(avg_weighted_sev_score)+", Weighted Issue Count:"+str(issue_weight_sum)+", Weighted Count Score:"+str(count_metric)+" Risk Score For Vessel:"+str(historical_risk_score))
        #display(sub_df[["INSPECTION_FROM_DATE","FINAL_SEVERITY","ISSUE_WEIGHT","ISSUE_BASESCORE"]])
        #break
    return vessel_historical_score_mapping,vessel_summary_statistics

    
    
    
    
    
############################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def decay(days):
    return np.exp(-lambda_val_dynamic_factors * days) if days > 0 else 1.0


def row_score(row):
    score = 20
    #print(datetime.datetime.strptime(row['Vessle Name Change Date'],"%Y-%m-%d"))

    # 1. Name / call-sign change
    if row['Changes In Vessel Name And Call Sign'] == 1:
        days_since_change = (today-row['Vessle Name Change Date']).days
        score += change_weight_mapping['Changes In Vessel Name And Call Sign'] * decay(row['Months Since Takeover'])
        #print("Vessel:"+str(row["IMO No."])+", Name Change Score:"+str(score)+", Days Since Change:"+str(days_since_change))
    # 2. Flag change
    if row['Changes In Vessel Flag'] == 1:
        days_since_change = (today-row['Flag Change date']).days
        score += change_weight_mapping["Changes In Vessel Flag"] * decay(days_since_change)
        #print("Vessel:"+str(row["IMO No."])+", Flag Change Score:"+str(score)+", Days Since Change:"+str(days_since_change))

    # 3. Class change
    if row['Changes In Vessel Class'] == 1:
        days_since_change = (today-row['Class Change Date']).days
        score += change_weight_mapping["Changes In Vessel Class"] * decay(days_since_change)
        #print("Vessel:"+str(row["IMO No."])+", Class Change Score:"+str(score)+", Days Since Change:"+str(days_since_change))

    # 4. Ownership
    if row['Change In Vessel Ownership'] == 1:
        days_since_change = (today-row['Ownership Change Date']).days
        score += change_weight_mapping["Change In Vessel Ownership"] * decay(days_since_change)
        #print("Vessel:"+str(row["IMO No."])+", Ownership Change Score:"+str(score)+", Days Since Change:"+str(days_since_change))

    # 5. Ship management
    if row['Change In Ship Management'] == 1:
        # days_since_change = (today-row['Ship Management Change Date']).days
        score += change_weight_mapping["Change In Ship Management"] #* decay(days_since_change)
        #print("Vessel:"+str(row["IMO No."])+", Ship Management Change Score:"+str(score))#+", Days Since Change:"+str(days_since_change))

    # # 6. Critical systems
    # if row['Changes In Vessel Critical Systems'] == 1:
    #     days_since_change =
    #     impact = row['Vessel Critical Systems Change Impact Category']
    #     score += change_weight_mapping["Changes In Vessel Critical Systems"] * impact * decay(days_since_change)

    return min(round(score, 1), 100)


def get_vessel_change_scores(dynamic_factors_data):
    dynamic_factors_data['Change_Risk_Score'] = dynamic_factors_data.apply(row_score, axis=1)
    vessel_change_score_mapping = dynamic_factors_data[["IMO No.","Change_Risk_Score"]].set_index('IMO No.')["Change_Risk_Score"].to_dict()
    # dynamic_factors_data[["IMO No.","Change_Risk_Score"]]
    # print(vessel_change_score_mapping)
    return vessel_change_score_mapping



############################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################



def compute_overall_risk_data(unique_vessels,analysis_df,score_mappings):
    
    vessel_change_score_mapping = score_mappings["vessel_change_score_mapping"]
    vessel_historical_score_mapping = score_mappings["vessel_historical_score_mapping"]
    owners_risk_score_mapping = score_mappings["owners_risk_score_mapping"]
    yard_risk_score_mapping = score_mappings["yard_risk_score_mapping"]
    flag_risk_score_mapping = score_mappings["flag_risk_score_mapping"]
    manager_risk_score_mapping = score_mappings["manager_risk_score_mapping"]
    make_risk_score_mapping = score_mappings["make_risk_score_mapping"]
    model_risk_score_mapping = score_mappings["model_risk_score_mapping"]
    class_risk_score_mapping = score_mappings["class_risk_score_mapping"]
    
    
    vessel_profile = {}
    for vessel in unique_vessels:
        temprow = analysis_df[analysis_df["IMO_NO"]==vessel]
        if(len(temprow)==0):
            continue
        row = temprow.iloc[0]
        vessel_profile[vessel] = {}
        vessel_profile[vessel]["Owners"] = row["R_OWNERS"]
        vessel_profile[vessel]["Yard"] = row["YARD"]
        vessel_profile[vessel]["Flag"] = row["FLAG_STATE"]
        vessel_profile[vessel]["Manager"] = row["MANAGER_GROUP"]
        vessel_profile[vessel]["Crew"] = row["NATIONALITY_OF_THE_CREW"]
        vessel_profile[vessel]["Inspector"] = row["INSPECTOR"]
        vessel_profile[vessel]["Marine Manager"] = row["MARINE_MANAGER"]
        vessel_profile[vessel]["Marine Superintendent"] = row["MARINE_SUPERINTENDENT"]
        vessel_profile[vessel]["Technical Manager"] = row["TECHNICAL_MANAGER"]
        vessel_profile[vessel]["Make"] = row["ME_MAKE"]
        vessel_profile[vessel]["Model"] = row["ME_MODEL"]
        vessel_profile[vessel]["Class"] = row["VESSEL_CLASS"]
    
    vessel_scores = {}
    for vessel in unique_vessels:
        vessel_scores[vessel] = {}
        if(vessel in vessel_change_score_mapping or int(vessel) in vessel_change_score_mapping):
            try:
                vessel_scores[vessel]["Change Score"] = vessel_change_score_mapping[int(vessel)]
            except:
                vessel_scores[vessel]["Change Score"] = vessel_change_score_mapping[vessel]
        else:
            vessel_scores[vessel]["Change Score"] = 0
        if(vessel in vessel_historical_score_mapping):
            try:
                vessel_scores[vessel]["Historical Score"] = vessel_historical_score_mapping[int(vessel)]
            except:
                vessel_scores[vessel]["Historical Score"] = vessel_historical_score_mapping[vessel]
        else:
            vessel_scores[vessel]["Historical Score"] = 0
        
        if(vessel in vessel_profile):
            vesselowners = vessel_profile[vessel]["Owners"]
            if(vesselowners in owners_risk_score_mapping):
                vessel_scores[vessel]["Owners Risk Score"] = owners_risk_score_mapping[vesselowners]
            else:
                vessel_scores[vessel]["Owners Risk Score"] = 0
        
    
            vesselyard = vessel_profile[vessel]["Yard"]
            if(vesselyard in yard_risk_score_mapping):
                vessel_scores[vessel]["Yard Risk Score"] = yard_risk_score_mapping[vesselyard]
            else:
                vessel_scores[vessel]["Yard Risk Score"] = 0
            vesselflag = vessel_profile[vessel]["Flag"]
            if(vesselflag in flag_risk_score_mapping):
                vessel_scores[vessel]["Flag Risk Score"] = flag_risk_score_mapping[vesselflag]
            else:
                vessel_scores[vessel]["Flag Risk Score"] = 0
            vesselmanager = vessel_profile[vessel]["Manager"]
            if(vesselmanager in manager_risk_score_mapping):
                vessel_scores[vessel]["Manager Risk Score"] = manager_risk_score_mapping[vesselmanager]
            else:
                vessel_scores[vessel]["Manager Risk Score"] = 0
            # vesselcrew = vessel_profile[vessel]["Crew"]
            # if(vesselcrew in crew_risk_score_mapping):
            #     vessel_scores[vessel]["Crew Risk Score"] = crew_risk_score_mapping[vesselcrew]
            # else:
            #     vessel_scores[vessel]["Crew Risk Score"] = 0
        
            # vesselinspector = vessel_profile[vessel]["Inspector"]
            # if(vesselinspector in inspector_risk_score_mapping):
            #     vessel_scores[vessel]["Inspector Risk Score"] = inspector_risk_score_mapping[vesselinspector]
            # else:
            #     vessel_scores[vessel]["Inspector Risk Score"] = 0
            # vesselmarine_manager = vessel_profile[vessel]["Marine Manager"]
            # if(vesselmarine_manager in marine_manager_risk_score_mapping):
            #     vessel_scores[vessel]["Marine Manager Risk Score"] = marine_manager_risk_score_mapping[vesselmarine_manager]
            # else:
            #     vessel_scores[vessel]["Marine Manager Risk Score"] = 0
            # vesselmarine_superintendent = vessel_profile[vessel]["Marine Superintendent"]
            # if(vesselmarine_superintendent in marine_superintendent_risk_score_mapping):
            #     vessel_scores[vessel]["Marine Superintendent Score"] = marine_superintendent_risk_score_mapping[vesselmarine_superintendent]
            # else:
            #     vessel_scores[vessel]["Marine Superintendent Score"] = 0
            # vesseltechnical_manager = vessel_profile[vessel]["Technical Manager"]
            # if(vesseltechnical_manager in technical_manager_risk_score_mapping):
            #     vessel_scores[vessel]["Technical Manager Risk Score"] = technical_manager_risk_score_mapping[vesseltechnical_manager]
            # else:
            #     vessel_scores[vessel]["Technical Manager Risk Score"] = 0
            vesselmake = vessel_profile[vessel]["Make"]
            if(vesselmake in make_risk_score_mapping):
                vessel_scores[vessel]["ME Make Risk Score"] = make_risk_score_mapping[vesselmake]
            else:
                vessel_scores[vessel]["ME Make Risk Score"] = 0
            vesselmodel = vessel_profile[vessel]["Model"]
            if(vesselmodel in model_risk_score_mapping):
                vessel_scores[vessel]["ME Model Risk Score"] = model_risk_score_mapping[vesselmodel]
            else:
                vessel_scores[vessel]["ME Model Risk Score"] = 0
            vesselclass = vessel_profile[vessel]["Class"]
            if(vesselclass in class_risk_score_mapping):
                vessel_scores[vessel]["Class Risk Score"] = class_risk_score_mapping[vesselclass]
            else:
                vessel_scores[vessel]["Class Risk Score"] = 0
        else:
            vessel_scores[vessel]["Class Risk Score"] = 0
            vessel_scores[vessel]["ME Model Risk Score"] = 0
            vessel_scores[vessel]["ME Make Risk Score"] = 0
            vessel_scores[vessel]["Manager Risk Score"] = 0
            vessel_scores[vessel]["Flag Risk Score"] = 0
            vessel_scores[vessel]["Yard Risk Score"] = 0
            vessel_scores[vessel]["Owners Risk Score"] = 0
            
    return vessel_scores


def get_final_risk_score_mapping(unique_vessels,analysis_df,score_mappings):
    
    vessel_scores = compute_overall_risk_data(unique_vessels,analysis_df,score_mappings)
    vessel_risk_scores_final = pd.DataFrame(vessel_scores).T
    overall_risk = compute_overall_risk(vessel_risk_scores_final,RISK_WEIGHTS)
    vessel_risk_scores_final["Overall Risk"] = overall_risk
    return vessel_risk_scores_final



############################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################







def compute_overall_risk(
    df: pd.DataFrame,
    weight_map: Dict[str, float] = RISK_WEIGHTS
) -> pd.Series:
    """Return weighted 0‑100 average risk per row."""
    weights = np.array([weight_map.get(c, 1.0) for c in df.columns])
    wsum    = weights.sum()
    return (df * weights).sum(axis=1, skipna=True) / wsum


def build_percentile_thresholds(df, cols, low_q=0.3, high_q=0.8):
    present = [c for c in cols if c in df.columns]
    if len(present) < len(cols):
        missing = set(cols) - set(present)
        print("Warning: missing columns ignored:", missing)

    q = df[present].quantile([low_q, high_q])
    return {c: (q.at[low_q, c], q.at[high_q, c]) for c in present}



def risk_level(value: float, p25: float, p75: float) -> str:
    """Return 'L', 'M', or 'H'. Missing values → 'M'."""
    if np.isnan(value) or value==0:
        return "UA"
    if value >= p75:
        return "H"
    if value < p25:
        return "L"
    return "M"


# ──────────────────────────────────────────────────────────────────────────
# 4.  BUSINESS BUCKET LABELLING
# ──────────────────────────────────────────────────────────────────────────
def label_vessel(
    row: pd.Series,
    thresholds: Dict[str, Tuple[float, float]]
) -> str:
    """Return comma‑separated bucket labels for one vessel."""
    # H / M / L per column  -----------------------------
    levels = {
        c: risk_level(row[c], *thresholds[c])
        for c in thresholds
    }
    # print("Labelling....")
    # print(row)
    # print(levels)
    cnt_H = sum(v == "H" for v in levels.values())
    cnt_M = sum(v == "M" for v in levels.values())
    cnt_UA = sum(v == "UA" for v in levels.values())
    if(cnt_UA>5):
        return "Not Enough Data"

    hist_H = sum(levels[c] == "H" for c in HIST_COLS)
    tech_H = sum(levels[c] == "H" for c in TECH_COLS)
    change_H = levels.get(CHANGE_COL, "M") == "H"

    labels: list[str] = []

    # ----- Champions ----------------------------------
    if all(v == "L" for v in levels.values()):
        labels.append("Champions")

    # ----- Red Alert Fleet ----------------------------
    if cnt_H >= 7:
        labels.append("Red Alert Fleet")

    # ----- Systemic Drift -----------------------------
    if cnt_H + cnt_M >= 5 >= cnt_H >= 3:   # 5+ risks medium/high (not all high)
        labels.append("Systemic Drift")

    # ----- Transition Watch ---------------------------
    if change_H and cnt_H < 5:
        labels.append("Transition Watch")

    # ----- Technically Vulnerable ---------------------
    if tech_H >= 2 and levels.get("Crew Risk Score", "M") != "H" \
                   and levels.get("Manager Risk Score", "M") != "H":
        labels.append("Technically Vulnerable")

    # ----- Heritage Concerns --------------------------
    if hist_H >= 2 and levels.get("Historical Score", "M") in ("L", "M"):
        labels.append("Heritage Concerns")

    # ----- Operational Urgency ------------------------
    if levels.get("Overall Risk", "L") == "H" and hist_H <= 2:
        labels.append("Operational Urgency")

    # ----- Steady Voyagers (fallback) -----------------
    if not labels:
        labels.append("Steady Voyagers")

    return ", ".join(labels)


# ──────────────────────────────────────────────────────────────────────────
# 5.  MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────
def segment_fleet(
    df_raw: pd.DataFrame,
    *,
    add_cluster: bool = True,
    n_clusters: int = 8,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df_raw : DataFrame
        Index = IMO, columns = raw risk scores (0‑100).
    add_cluster : bool
        If True, appends a K‑Means cluster id.
    Returns
    -------
    DataFrame  – original scores + Overall Risk + labels (+ cluster id)
    """
    df = df_raw.copy()

    # 1.  Ensure all score columns exist
    for col in RISK_COLS:
        if col not in df:
            df[col] = np.nan

    # 2.  Overall Risk
    df["Overall Risk"] = compute_overall_risk(df, RISK_WEIGHTS)

    # 3.  Percentiles
    thresholds = build_percentile_thresholds(df, RISK_COLS)

    # 4.  Business label
    df["Risk Label"] = df.apply(label_vessel, axis=1, thresholds=thresholds)

    # 5.  Optional K‑Means cluster id
    # if add_cluster:
    #     imputer = SimpleImputer(strategy="median")
    #     scaler  = MinMaxScaler()
    #     X = scaler.fit_transform(imputer.fit_transform(df[RISK_COLS]))
    #     km = KMeans(n_clusters=n_clusters, random_state=random_state,
    #                 n_init="auto")
    #     df["Cluster ID"] = km.fit_predict(X)

    return df


def get_vessel_segments(vessel_risk_scores_final):
    segmented = segment_fleet(vessel_risk_scores_final, add_cluster=True)
    return segmented
    
    
####################################################################################################################################
####################################################################################################################################

def get_vessel_details(vessel,generic_factors,dynamic_factors):
    return {}
    
###################################################



def get_access_token():
    # 1) The token string you passed as "user" in curl
    token = os.environ["DB_TOKEN"]

    # 2) Build the Basic auth header value exactly as curl -u xyz does:
    #    curl will base64‑encode "xyz" (no trailing colon) when you omit a secret.
    b64 = base64.b64encode(token.encode("utf-8")).decode("ascii")
    auth_header = f"Basic {b64}"

    # 3) Your token endpoint and form payload
    url = "https://adb-4626041107022307.7.azuredatabricks.net/oidc/v1/token"
    payload = {
        "grant_type": "client_credentials",
        "scope":      "all-apis"
    }

    # 4) Send the POST with the handcrafted header
    resp = requests.post(
        url,
        headers={
          "Authorization": auth_header,
          "Content-Type":  "application/x-www-form-urlencoded"
        },
        data=payload,
        timeout=10
    )

    # 5) Check for errors
    if not resp.ok:
        print("Request failed:", resp.status_code, resp.text)
        return None
    else:
        data = resp.json()
        print("Success:", data)
        print("Access token:", data.get("access_token"))
        return data.get("access_token")

def get_deficiency_df_snowflake(access_token):
    connection = sql.connect(
                        server_hostname = "adb-4626041107022307.7.azuredatabricks.net",
                        http_path = "/sql/1.0/warehouses/de2adf4943b29216",
                        access_token = access_token)
    cursor = connection.cursor()
    cursor.execute("select * from reporting_layer.qhse.synergypool_vw_psc_performance")
    results = cursor.fetchall()
    rows = results  # list of tuples
    cols = [c[0] for c in cursor.description]
    # 2. Build the DataFrame
    deficiencies_df_latest = pd.DataFrame(rows, columns=cols)
    return deficiencies_df_latest
    
    
    
    