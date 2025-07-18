#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:12:32 2025

@author: sasidharankumar
"""
import pandas as pd
import numpy as np
from datetime import datetime
from constants import lambda_val,weight_map, GENERIC_PHRASES, MARINE_TERMS,TERMS_NOT_REQUIRED,category_dict,lemmatizer,WACTION
from util import parse_action_codes,max_weight,get_severity
import re, networkx as nx
from pyvis.network import Network
import streamlit as st
# import string
from rapidfuzz import process, fuzz
import warnings
warnings.filterwarnings("ignore")
# from nltk.util import ngrams
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.tag import pos_tag
# from nltk.corpus import wordnet
# import nltk
# # Download necessary NLTK data
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')

import math

today = datetime.now()
def get_issue_weight(row):
    age_d = (today - row['INSPECTION_FROM_DATE']).days
    w= np.exp(-lambda_val * age_d)
    return w
def max_weight(code_list):
    return max((weight_map.get(c, 50) for c in code_list), default=np.nan)

# def get_top_topics_details(top_topics_for_authority):
#     top_topic_ids = list(top_topics_for_authority.index)# meanwhile, throughout,compare, 
#     col = 'Topic'                 # the column you want to filter/sort
#     # 1. Build a small DataFrame just to capture the desired order
#     order_df = pd.DataFrame({
#         col: top_topic_ids,
#         'order': range(len(top_topic_ids))
#     })

#     # 2. Inner‐join your data to that order (this both filters and tags each row)
#     df_ordered = (
#         topic_mapping
#         .merge(order_df, on=col, how='inner')
#         .sort_values('order')   # put them in the list’s order
#         .drop(columns='order')  # drop the helper column if you like
#         .reset_index(drop=True)
#     )
#     return df_ordered

def group_words(words):
    # First pass: lemmatize everything
    lemmas = [lemma(w) for w in words]

    # 4. Build clusters of similar lemmas
    threshold = 80  # similarity % cutoff
    clusters = {}   # rep_lemma -> set of lemmas in cluster

    for w in set(lemmas):
        # try to match to an existing cluster
        best = None
        best_score = 0
        for rep in clusters:
            score = fuzz.ratio(w, rep)
            if score > best_score:
                best_score, best = score, rep

        if best_score >= threshold:
            clusters[best].add(w)
        else:
            # no good match → start new cluster
            clusters[w] = {w}

    # 5. Decide a canonical for each cluster (e.g. shortest string)
    canonical = {}
    for rep, members in clusters.items():
        canon = min(members, key=len)
        for m in members:
            canonical[m] = canon

    # 6. Map your original list
    normalized = [canonical[l] for l in lemmas]

    return normalized

def get_keyword_frequency(authority_df,column="TOPIC_NAMES"):
    # Flatten the list of lists in the specified column
    keywords = [item for sublist in authority_df[column].values for item in sublist]
    cleaned = [
        x for x in keywords
        if x not in ("", "nil", None)
          and not (isinstance(x, float) and math.isnan(x))
    ]
    final_keywords = group_words(cleaned)
    top_keywords = pd.Series(final_keywords).value_counts().head(10)
    return top_keywords

def get_topic_frequency_weighted(authority_df):
    df = authority_df[["TOPIC", "ISSUE_WEIGHT"]].copy()
    df = df[df["TOPIC"]!=-1]

    top_topics = (
        df.groupby("TOPIC")["ISSUE_WEIGHT"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return top_topics

def get_keyword_frequency_weighted(authority_df,column="TOPIC_NAMES"):
    # 1. Start with your DF of shape (n_rows, …)
    #    Assume `column` contains a Python list of phrases for each row
    #    and FINAL_SEVERITY is a numeric weight.
    df = authority_df[[column, "ISSUE_WEIGHT"]].copy()

    # 2. Explode the list-of-phrases into one row per phrase
    df = df.explode(column)

    # 3. Drop bad values
    df = df[df[column].notna() & (df[column] != "") & (df[column] != "nil")]

    # 4. Now apply your group_words to each phrase,
    #    which returns a list of “final keywords” per phrase.
    df["keywords"] = df[column].apply(lambda lst: group_words([lst])[0] if isinstance(lst, str) else [])

    # 5. Explode again so each keyword is its own row
    df = df.explode("keywords")

    # 6. Finally: group by keyword & sum the severities
    top_keywords = (
        df
        .groupby("keywords")["ISSUE_WEIGHT"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    return top_keywords



# def extract_useful_ngrams(tokens, n):
#     """Return list of n-grams (joined with space) that pass our filters."""
#     out = []
#     for gram in ngrams(tokens, n):
#         phrase = " ".join(gram)
#         # 1) drop if in our generic list
#         if phrase in GENERIC_PHRASES:
#             continue
#         # 2) keep only if any token is in MARINE_TERMS
#         # if any(tok in MARINE_TERMS for tok in gram):
#         if any(any(marine_term in tok for marine_term in MARINE_TERMS) for tok in gram):
#             out.append(phrase)
#     return out

# def clean_issue_text_grams(issue_text):
#     if pd.isna(issue_text):
#         return "", [], []

#     # (A) Basic cleanup & tokenize
#     text = issue_text.lower()
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = re.sub(r'\d+', "", text)
#     tokens = nltk.word_tokenize(text)

#     # (B) Remove stopwords
#     stops = set(stopwords.words("english"))
#     tokens = [t for t in tokens if t not in stops]

#     # (C) POS & Lemmatize
#     lemmatizer = WordNetLemmatizer()
#     tagged = pos_tag(tokens)
#     lemmas = []
#     for w, tag in tagged:
#         pos = get_wordnet_pos(tag) or "n"
#         lemmas.append(lemmatizer.lemmatize(w, pos=pos))

#     # (D) Extract useful 2-grams and 3-grams
#     bigrams = extract_useful_ngrams(lemmas, 2)
#     trigrams = extract_useful_ngrams(lemmas, 3)

#     # (E) Return cleaned text + n-grams for further analysis
#     cleaned_text = " ".join(lemmas)
#     return cleaned_text, bigrams, trigrams

def merge_ngrams(bigrams, trigrams):
    """
    Merge bigrams and trigrams, removing any bigram whose two words
    both appear in any one trigram.
    """
    # Split trigrams into sets of words for quick subset testing
    trigram_sets = [set(ng.split()) for ng in trigrams]

    cleaned_bigrams = []
    for bg in bigrams:
        bg_set = set(bg.split())
        # If bg_set is NOT a subset of any trigram_set, we keep it
        if not any(bg_set.issubset(tg_set) for tg_set in trigram_sets):
            cleaned_bigrams.append(bg)

    # Now combine cleaned bigrams + trigrams
    merged = list(dict.fromkeys(cleaned_bigrams + trigrams))
    # dict.fromkeys(...) preserves order and deduplicates

    return merged

# def get_wordnet_pos(tag):
#     """Map POS tag to first character used by WordNetLemmatizer"""
#     if tag.startswith('J'):
#         return wordnet.ADJ
#     elif tag.startswith('V'):
#         return wordnet.VERB
#     elif tag.startswith('N'):
#         return wordnet.NOUN
#     elif tag.startswith('R'):
#         return wordnet.ADV
#     else:
#         return None

# def clean_issue_text(issue_text):
#     """
#     Cleans the issue text by removing stopwords, adjectives, pronouns, etc.,
#     and performs lemmatization.
#     """
#     if pd.isna(issue_text):
#         return ""

#     # Convert to lowercase
#     text = issue_text.lower()
#     # Remove punctuation
#     text = text.translate(str.maketrans('', '', string.punctuation))
#     # Remove digits
#     text = re.sub(r'\d+', '', text)
#     # Tokenize
#     tokens = nltk.word_tokenize(text)
#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     tokens = [word for word in tokens if word not in stop_words and word not in TERMS_NOT_REQUIRED]

#     # POS tagging
#     tagged_tokens = pos_tag(tokens)

#     # Lemmatization and filtering based on POS
#     lemmatizer = WordNetLemmatizer()
#     cleaned_tokens = []
#     for word, tag in tagged_tokens:
#         w_pos = get_wordnet_pos(tag)
#         if w_pos is not None:
#             # Lemmatize based on WordNet POS
#             lemma = lemmatizer.lemmatize(word, pos=w_pos)
#         else:
#             # Default to noun lemmatization if POS not recognized by WordNet
#             lemma = lemmatizer.lemmatize(word)

#         # Filter out pronouns (PRP, PRP$) and adjectives (JJ, JJR, JJS)
#         # if not (tag.startswith('PRP') or tag.startswith('JJ')):
#         # #if not tag.startswith('JJ'):
#         #     cleaned_tokens.append(lemma)
#         cleaned_tokens.append(lemma)

#     # Join tokens back into a string
#     cleaned_text = ' '.join(cleaned_tokens)

#     return cleaned_text.strip()

def visualize_text_graph(text_analysis_df):
    defects_description_list = list(text_analysis_df["MERGED_NGRAMS"].values)
    authorities_list = list(text_analysis_df["AUTHORITY"].values)
    # sentences = defects_description_list
    authority_description_combo = zip(authorities_list,defects_description_list)
    # ── 2. Build co-occurrence edges (same sentence) ─────────────
    def tokenize(text):
        return re.findall(r"[a-zA-Z']+", text.lower())

    G = nx.Graph()
    # for sent in sentences:
    for combo in authority_description_combo:
        sent = combo[1]
        auth = combo[0]
        if(not G.has_node(auth)):
            G.add_node(auth,category="authority",color="red")
        # words = tokenize(sent)
        for w in sent:
            if(not G.has_node(w)):
                G.add_node(w,category="issue",color="green")
            if G.has_edge(auth,w):
                G[auth][w]['weight'] += 1
            else:
                G.add_edge(auth, w, weight=1)

        
    # ── 4. Send to PyVis for an interactive view ─────────────────
    net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black",notebook=True,cdn_resources='remote')

    DEGREE_THRESHOLD = 3

    if G.is_directed():                                     # handle both kinds
        deg = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}
    else:
        deg = dict(G.degree())

    active_nodes = {n for n, d in deg.items() if d >= DEGREE_THRESHOLD}

    # ------------------------------------------------------------------
    # ❷  Add nodes (only the active ones)
    # ------------------------------------------------------------------
    for node in active_nodes:
        col = G.nodes[node].get("color")          # returns '#6ECE6E' (or None if unset)
        # level 0 for red, level 1 for green
        level = 0 if col.lower().startswith("red") else 1

        net.add_node(node,
                    label=node,
                    color=col,#"lightgreen",
                    size=8,
                    level=level)

    # ------------------------------------------------------------------
    # ❸  Add edges:
    #     • both endpoints must be “active”
    #     • edge weight must exceed the visual threshold (> 5 here)
    # ------------------------------------------------------------------
    for u, v, data in G.edges(data=True):
        if u in active_nodes and v in active_nodes:
            w = data.get("weight", 1)          # default weight 1 if missing
            if w is not None:        #  and w > 5 - visual filter
                net.add_edge(u, v, value=w, color="blue")


    net.set_options("""
    var options = {
      "edges": { "color": {"color":"#AAAAAA"}, "smooth": false },
      "nodes": {
        "shape": "dot",
        "scaling": { "min": 4, "max": 20 }
      },

      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -800,
          "centralGravity": 0.1,
          "springLength": 200,
          "springConstant": 0.01,
          "damping": 0.8,
          "avoidOverlap": 0.2
        },
        "minVelocity": 0.005,
        "maxVelocity": 0.2,
        "solver": "barnesHut",
        "timestep": 0.5
      }
    }
    """)
    # "layout":{
    #     "hierarchical": {
    #       "enabled": true,
    #       "direction": "LR",
    #       "levelSeparation": 500
    #     }
    #   },

    # net.show("word_network.html")        # opens in default browser
    net.show("word_network.html")
    # net.show("mygraph.html")
    with open("word_network.html", 'r') as f:
        html_string = f.read()
    return html_string
    


def summarize_text(row):
    text = " ".join([str(row[col]) for col in ["NATURE_OF_DEFICIENCY","CAUSE_ANALYSIS","IMMEDIATE_ACTION_TAKEN"] if pd.notna(row[col])])
    return text

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

def lemma(word: str) -> str:
    """Lowercase + noun-then-verb lemmatize, take shortest."""
    w = word.lower()
    noun = lemmatizer.lemmatize(w, pos='n')
    verb = lemmatizer.lemmatize(w, pos='v')
    return noun if len(noun) <= len(verb) else verb



def compute_deficiency_sub_category_to_category_mapping(deficiency_codes,category_dict):
    deficiency_sub_category_to_category_mapping = {}
    code_subcategory_map = {}

    for item,row in deficiency_codes.iterrows():
        sub_category_str = str(row["Title"])
        code_str = str(row["Code"])
        if(len(code_str)>3):
            if(len(code_str)==4):
                code_str = "0"+code_str
            category_code_str = code_str[:2]
            sub_category_code_str = code_str[:3]
            category_str = category_dict[category_code_str]
            #sub_category_str = sub_category_dict[sub_category_code_str]
            deficiency_sub_category_to_category_mapping[sub_category_str] = category_str
            code_subcategory_map[code_str] = sub_category_str
    return deficiency_sub_category_to_category_mapping,code_subcategory_map

def apply_code_correction(row):
    category = row["PSC_CODE"]
    if(len(category)==4):
        return "0"+category
    return category



def extract_sub_category(row,code_subcategory_map):
    subcategory_str = np.nan
    category = row["PSC_CODE"]
    if(category in code_subcategory_map):
        subcategory_str = code_subcategory_map[category]
    return subcategory_str

def extract_category(row):
    category_str = np.nan
    category = row["PSC_CODE"]
    cat = category[:2]
    if(cat in category_dict):
        category_str = category_dict[cat]
    return category_str


def get_entity_cat_subcat_distribution(entity,text_analysis_df,code_subcategory_map, aggtype="count"):
    unique_entities = text_analysis_df[entity].unique()
    entity_category_distribution = {}
    entity_sub_category_distribution = {}
    for entity_item in unique_entities:
        entity_df = text_analysis_df[text_analysis_df[entity]==entity_item]
        subcategories_str = entity_df.apply(lambda row: extract_sub_category(row,code_subcategory_map),axis=1)
        categories_str = entity_df.apply(lambda row: extract_category(row),axis=1)
        entity_df["SUBCATEGORY"] = subcategories_str
        entity_df["CATEGORY"] = categories_str
        # authority_df[["SUBCATEGORY", "CATEGORY"]] = authority_df.apply(extract_category_sub_category,axis=1)
        entity_df.dropna(subset=["SUBCATEGORY","CATEGORY"],inplace=True)
        if(aggtype=="count"):
            category_counts = entity_df["CATEGORY"].value_counts()
            category_percentages = (category_counts / category_counts.sum()) * 100
            category_percentages_df = pd.DataFrame({
                    'Category': category_percentages.index,
                    'Percentage': category_percentages.values
                })
        else:
            # category_counts = entity_df["CATEGORY"].value_counts()
            adf = entity_df.groupby("SUBCATEGORY").sum().sort_values(by="ISSUE_WEIGHT")
            # adf = (entity_df
            #        .groupby("PSC_CODE", as_index=False)        # keep PSC_CODE as a column
            #        .agg(ISSUE_WEIGHT = ("ISSUE_WEIGHT", "sum"),
            #             CATEGORY     = ("CATEGORY",     "first"))   # or "max"/"min"
            #        .sort_values("ISSUE_WEIGHT", ascending=False))
            category_percentages_df = pd.DataFrame({
                    'Category': adf.index.values,
                    'Percentage': adf["ISSUE_WEIGHT"].values
                })
        
        if(aggtype=="count"):
            sub_category_counts = entity_df["SUBCATEGORY"].value_counts()
            sub_category_percentages = (sub_category_counts / sub_category_counts.sum()) * 100
            sub_category_percentages_df = pd.DataFrame({
                    'Category': sub_category_percentages.index,
                    'Percentage': sub_category_percentages.values
                })
        else:
            # sub_category_counts = entity_df["SUBCATEGORY"].value_counts()
            adf = entity_df.groupby("SUBCATEGORY").sum().sort_values(by="ISSUE_WEIGHT")
            # adf = (entity_df
            #        .groupby("PSC_CODE", as_index=False)        # keep PSC_CODE as a column
            #        .agg(ISSUE_WEIGHT = ("ISSUE_WEIGHT", "sum"),
            #             CATEGORY     = ("CATEGORY",     "first"))   # or "max"/"min"
            #        .sort_values("ISSUE_WEIGHT", ascending=False))
            sub_category_percentages_df = pd.DataFrame({
                    'Category': adf.index.values,
                    'Percentage': adf["ISSUE_WEIGHT"].values
                })
        


        entity_category_distribution[entity_item] = category_percentages_df[:5]
        entity_sub_category_distribution[entity_item] = sub_category_percentages_df[:15]
       
    return entity_category_distribution,entity_sub_category_distribution



@st.cache_data(show_spinner=True)
def prepare_trend_analysis_data(inspection_data,psc_codes_scoring_data):
    print("Preparing Trend Analysis Data.......")
    deficiency_codes = psc_codes_scoring_data.copy()
    text_analysis_df = inspection_data[["IMO_NO","AUTHORITY","INSPECTION_FROM_DATE","PSC_CODE","NATURE_OF_DEFICIENCY","IMMEDIATE_ACTION_TAKEN","CAUSE_ANALYSIS","ISSUE_WEIGHT",'PSC_CATEGORY','REFERENCE_CODE_1']]#.groupby("NATURE_OF_DEFICIENCY").sum().sort_values("ISSUE_WEIGHT",ascending=False)
    text_analysis_df["ISSUE_DETAILS"] = text_analysis_df.apply(summarize_text,axis=1)
    authority_mapping = normalize_strings(text_analysis_df["AUTHORITY"], threshold=90)
    text_analysis_df["AUTHORITY"] = text_analysis_df["AUTHORITY"].map(authority_mapping)
    # text_analysis_df['CLEANED_ISSUE_DETAILS'] = text_analysis_df['ISSUE_DETAILS'].apply(clean_issue_text)
    # Apply to your DataFrame
    # text_analysis_df[["CLEANED_TEXT", "USEFUL_BIGRAMS", "USEFUL_TRIGRAMS"]] = (text_analysis_df["ISSUE_DETAILS"]
    #       .apply(lambda t: pd.Series(clean_issue_text_grams(t)))
    # )
    # Apply to your DataFrame
    # text_analysis_df['MERGED_NGRAMS'] = text_analysis_df.apply(
    #     lambda row: merge_ngrams(row['USEFUL_BIGRAMS'] or [], row['USEFUL_TRIGRAMS'] or []),
    #     axis=1
    # )
    # gram_result_df = text_analysis_df[["CLEANED_TEXT", "USEFUL_BIGRAMS", "USEFUL_TRIGRAMS", "MERGED_NGRAMS"]]
    text_analysis_df = text_analysis_df.reset_index(drop=True)
    
    deficiency_sub_category_to_category_mapping,code_subcategory_map = compute_deficiency_sub_category_to_category_mapping(deficiency_codes,category_dict)
    
    text_analysis_df["SEVERITY"] = text_analysis_df.apply(lambda row:get_severity(row,psc_codes_scoring_data),axis=1)
    # analysis_df.apply(lambda row:get_severity(row,psc_codes_scoring_data),axis=1)
    text_analysis_df["PSC_CODE"] = text_analysis_df.apply(apply_code_correction,axis=1)


    text_analysis_df["ActionCodes"] = text_analysis_df["REFERENCE_CODE_1"].apply(parse_action_codes)
    text_analysis_df["ACTIONCODE_SEVERITY"] = text_analysis_df["ActionCodes"].apply(max_weight)
    text_analysis_df['FINAL_SEVERITY'] = WACTION * text_analysis_df["ACTIONCODE_SEVERITY"] + ((1 - WACTION) * text_analysis_df['SEVERITY']*10)
    text_analysis_df["ISSUE_BASESCORE"] = text_analysis_df["FINAL_SEVERITY"]*text_analysis_df["ISSUE_WEIGHT"]


    

    vessel_category_distribution,vessel_sub_category_distribution = get_entity_cat_subcat_distribution("IMO_NO",text_analysis_df,code_subcategory_map)
    authority_category_distribution,authority_sub_category_distribution = get_entity_cat_subcat_distribution("AUTHORITY",text_analysis_df,code_subcategory_map)
    
    
    vessel_category_distribution_time,vessel_sub_category_distribution_time = get_entity_cat_subcat_distribution("IMO_NO",text_analysis_df,code_subcategory_map,aggtype="time")
    authority_category_distribution_time,authority_sub_category_distribution_time = get_entity_cat_subcat_distribution("AUTHORITY",text_analysis_df,code_subcategory_map,aggtype="time")
    
    category_distribution = {
        "Vessel":vessel_category_distribution,
        "Authority":authority_category_distribution,
        "Vessel_Time":vessel_category_distribution_time,
        "Authority_Time":authority_category_distribution_time,
    }
    sub_category_distribution = {
        "Vessel":vessel_sub_category_distribution,
        "Authority":authority_sub_category_distribution,
        "Vessel_Time":vessel_sub_category_distribution_time,
        "Authority_Time":authority_sub_category_distribution_time
    }
    return category_distribution,sub_category_distribution,text_analysis_df,None


