#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 15:13:42 2025

@author: sasidharankumar
"""
import re
import numpy as np
from typing import Dict, Iterable, Tuple
from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer
from sentence_transformers import CrossEncoder
from databricks import sql
# ── core regex patterns ────────────────────────────────────────────
NUM_RE   = re.compile(r"\d+")        # grabs every run of digits
DELIM_RE = re.compile(r"[\/,;]")     # split on / , or ;  (add more if needed)
# if you only care about the “worst” code according to a weight-map:
weight_map = {"10":10, "14":25, "15":35, "16":45, "17":60, "18":40, "19":80,
              "21":50, "26":45, "30":100, "35":85, "36":85, "40":55, "45":75,
              "46":65, "47":50, "48":50, "49":45, "50":40, "55":30, "60":45,
              "65":80, "70":35, "80":45, "81":45, "85":70, "95":20, "96":10,
              "99":70}

entity_issuetype_mapping = {
    "R_OWNERS":["Process","Equipment"],
    "YARD": ["Equipment"],
    "YARD_COUNTRY": ["Equipment"],
    "FLAG_STATE":["Process"],
    "MANAGER_GROUP":["Process"],
    "ME_MAKE": ["Equipment"],
    "ME_MODEL": ["Equipment"],
    "VESSEL_CLASS": ["Process"],
    "NATIONALITY_OF_THE_CREW": ["Process"],
    "INSPECTOR": ["Process"],
    "MARINE_MANAGER": ["Process"],
    "MARINE_SUPERINTENDENT": ["Process"],
    "TECHNICAL_MANAGER": ["Process"]
}

change_weight_mapping = {
    "Changes In Vessel Name And Call Sign": 5,
    "Changes In Vessel Flag": 10,
    "Changes In Vessel Class": 15,
    "Change In Vessel Ownership": 20,
    "Change In Ship Management": 15,
    "Changes In Vessel Critical Systems": 25
}

RISK_WEIGHTS = {
    # Historical / organisation
    "Owners Risk Score"        : 0.05,
    "Yard Risk Score"          : 0.1,
    "Flag Risk Score"          : 0.05,
    "Manager Risk Score"       : 0.05,
    "Class Risk Score"         : 0.05,

    # Technical
    "ME Make Risk Score"       : 0.05,
    "ME Model Risk Score"      : 0.05,

    # Change & operational
    "Change Score"             : 0.4,
    "Historical Score"         : 0.2,

    # If you add new columns later simply extend the dict
}

ISSUE_HALF_LIFE=365
lambda_val=np.log(2)/ISSUE_HALF_LIFE
HALF_LIFE_CHANGES = 365 # 6 Months to reduce the score by 0.5
lambda_val_dynamic_factors = np.log(2) / HALF_LIFE_CHANGES

WACTION = 0.7
BUFFER_ZONE = 2
max_severity = 100

BASELINE_RISK_WEIGHT = 20
DEVIATION_RISK_WEIGHT = 40
SEVERITY_RISK_WEIGHT = 40
max_severity=100






RISK_COLS: Tuple[str, ...] = (
    "Change Score", "Historical Score",
    "Owners Risk Score", "Yard Risk Score", "Flag Risk Score",
    "Manager Risk Score",
    "ME Make Risk Score", "ME Model Risk Score",
    "Class Risk Score", "Overall Risk"       # Overall will be added later
)

HIST_COLS  = ("Owners Risk Score", "Yard Risk Score",
              "Flag Risk Score", "Manager Risk Score", "Class Risk Score")

TECH_COLS  = ("ME Make Risk Score", "ME Model Risk Score",
              "Yard Risk Score", "Class Risk Score")

CHANGE_COL = "Change Score"





##################################################################################################
##Trends Settings#########
###################
code_category_map = {}
category_dict = {
    "01":"Certificates & Documentation",
    "02":"Structural condition",
    "03":"Water/Weathertight condition",
    "04":"Emergency Systems",
    "05":"Radio communication",
    "06":"Cargo operations including equipment",
    "07":"Fire safety",
    "08":"Alarms",
    "09":"Working and Living Conditions ",
    "10": "Safety of Navigation",
    "11": "Life saving appliances",
    "12": "Dangerous Goods",
    "13": "Propulsion and auxiliary machinery",
    "14": "Pollution Prevention ",
    "15": "ISM",
    "16": "ISPS",
    "18": "MLC, 2006",
    "99": "Other"
}
# A small stop-list of generic bigrams/trigrams to drop
GENERIC_PHRASES = {
    "go to", "properly fill", "fill up", "and the", "the the",
    # … add more as you discover them
}

TERMS_NOT_REQUIRED = ["this","the","in","also","upon","and","to","of","at","by","for","with","on","from","is","are","was","were","be","been",
                      "being","was","were","meanwhile", "throughout","compare", "every",
                      "can","could","may","might","will","shall","should","do","does","did","have","had","having","be","been","being","am"
                      ]

MARINE_TERMS = {
    "engine", "leak", "leaking", "fire", "damper", "hydraulic","certificate","crew","port","starboard","stbd","stern","vessel","sewage","cover","door",
    "ventilation", "lifeboat", "hatch", "bilge", "pump", "cargo","rust","hull","navigation","gauge","pressure","rescue","boat","oil","aft","fwd","forward",
    "hatchway", "hose", "valve", "pipe", "fuel", "oily","ballast","record","procedure","bridge","generator","main","panel","steering","unit","room","galley",
    "alarm","draft","raft","bouys","lifebouy","fan","feeder","diesel","rail","corrode","toilet","log","compass","garbage","report","deck","spanner","water",
    # "security","labour","medical","ladder","safety","meter","plug","boiler","accomodation","gangway","gear","steer","signal","incinerator","detector",
    # "sounder","scrubber","system","flag","equipment","ecdis","antenna","store","plan","radar","moor","rope","air","manifold","turbo","charger","sensor",
    # "device","ppe","ism","sprinkler","nozzle","incenarator","crane","piracy","light","gear","vdr","screen","corridor","medicine","battery","laundry",
    # "detector","gas","isps","route","training","manual","overtime","shipboard","lifebouys","booklet","electrical","damage","unsafe","refrigerator","printer"
    # … add the key domain tokens you care about
}
# ── 2. Sentence-BERT embeddings ────────────────────────────────────
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim, small & fast
model_rerank = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
lemmatizer = WordNetLemmatizer()
# embeddings = model.encode(text_analysis_df["ISSUE_DETAILS"], show_progress_bar=False)

DEFICIENCIES_TABLE = "REPORTING_LAYER.QHSE.SYNERGYPOOL_VW_PSC_PERFORMANCE"
INCIDENTS_TABLE = "REPORTING_LAYER.QHSE.VW_INCIDENT"
OPEN_DEFECTS_TABLE = "REPORTING_LAYER.CRP.DFT_DEFECT_LIST"





###################################################################################
# Define external/crew-related incident categories to exclude
external_incidents_sub_categories = [
    'Security_incident', 'Injury', 'Medvac', 'Death due to illness ',
    'Fatality', 'Fatality due to illness ', 'Missing person', 'Man overboard'
]

external_incidents_sub2_categories = [
    'Sabotage', 'First aid', 'Fatality', 'Theft', 'LTI',
    'Disability', 'Stowaways'
]

# Mapping for standardization
standardization_mapping = {
    'minor': 'minor',
    'miinor': 'minor',
    'slight': 'minor',
    'substantial': 'substantial',
    'substantial harm': 'substantial',
    'level 3': 'substantial',
    'level 3 - substantial': 'substantial',
    'level 2': 'marginal',
    'level 2 (marginal)': 'marginal',
    'marginal': 'marginal',
    'critical': 'critical',
    'severe': 'critical',
    'catastrophic': 'catastrophic',
    'catestrophic': 'catastrophic'
}

# Define risk mapping to scores from 0 to 100
risk_score_mapping = {
    'catastrophic': 100,
    'critical': 80,
    'substantial': 60,
    'marginal': 40,
    'minor': 20
}

HALF_LIFE_INCIDENTS = 360 # 1 year to reduce the score by 0.5
lambda_val_incidents = np.log(2) / HALF_LIFE_INCIDENTS
