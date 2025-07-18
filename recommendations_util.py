#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 20:12:07 2025

@author: sasidharankumar
"""
import pandas as pd
import re
from constants import model, model_rerank,category_dict
import faiss
import chardet
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
def clean_internal_checklist(df):
    # ───────────────────────────────────────────────────────────────
    # 1. Load the CSV  (first row contains WEIGHTAGE/RATING header cells)
    # ───────────────────────────────────────────────────────────────
    # SRC = Path("/mnt/data/PSC Internal Checklist.csv")   # adjust as needed
    # df  = pd.read_csv('PSC Internal Checklist.csv', encoding='latin-1')
    # df = pd.read_csv(path, encoding='latin-1')
    df.dropna(axis="rows", how="all",inplace=True)
    df = df.iloc[:-5]
    # turn first row into proper column names (WEIGHTAGE, RATING, …)
    first = df.iloc[0]
    df.columns = [c if pd.isna(h) else str(h).strip() for c, h in zip(df.columns, first)]
    df = df.iloc[1:].reset_index(drop=True)              # drop that header row

    # rename the first (unnamed) column to RAW
    df.rename(columns={df.columns[0]: "RAW"}, inplace=True)

    # keep only RAW + numeric columns
    keep = {"RAW", "WEIGHTAGE", "RATING", "SCORE", "REMARKS"}
    extra = [c for c in df.columns if c not in keep and df[c].isna().all()]
    df.drop(columns=extra, inplace=True)

    # ───────────────────────────────────────────────────────────────
    # 2. Helper to detect ALL-CAPS section rows
    # ───────────────────────────────────────────────────────────────
    def is_header(text: str) -> bool:
        """Return True if text is ALL CAPS (letters only)."""
        letters = re.sub(r"[^A-Za-z]+", "", text)
        return bool(letters) and letters.isupper()

    # ───────────────────────────────────────────────────────────────
    # 3. Build tidy rows  (Section, Question No, Question, numeric cols)
    # ───────────────────────────────────────────────────────────────
    rows = []
    section_idx = 0
    q_counter   = 0
    section     = None

    for _, r in df.iterrows():
        raw = str(r["RAW"]).strip()
        if not raw:
            continue

        if is_header(raw):                       # new section
            section_idx += 1
            section      = raw.title()           # nicer capitalisation
            q_counter    = 0
            continue

        if section is None:                      # ignore preamble rows
            continue

        # build question number  e.g. 6.1, 6.2 …
        q_counter += 1
        q_no = f"{section_idx}.{q_counter}"

        rows.append({
            "Section":      section,
            "Question No":  q_no,
            "Question":     raw,
            "WEIGHTAGE":    r.get("WEIGHTAGE"),
            "RATING":       r.get("RATING"),
            "SCORE":        r.get("SCORE"),
            "REMARKS":      r.get("REMARKS")
        })

    tidy = pd.DataFrame(rows)

    # # ───────────────────────────────────────────────────────────────
    # # 4. Save / inspect
    # # ───────────────────────────────────────────────────────────────
    # OUT = SRC.with_name("psc_checklist_exploded.csv")
    # tidy.to_csv(OUT, index=False)
    # print(f"Saved → {OUT}")
    # print(tidy.head())
    return tidy

# def find_closest_deficiency_category_subcategory(checklist, k=2):
#     # Compute embedding for the query
#     checklist_embedding = model.encode([checklist], convert_to_numpy=True)
#     # print(index_categories)
#     # Search the FAISS index for the nearest neighbors
#     distances_categories, indices_categories = index_categories.search(checklist_embedding, k)
#     #print(distances_categories, indices_categories)
#     distances_subcategories, indices_subcategories = index_subcategories.search(checklist_embedding, k)
#     distances_categories_desc, indices_categories_desc = index_category_desc.search(checklist_embedding, k)
#     distances_subcategories_desc, indices_subcategories_desc = index_subcategory_desc.search(checklist_embedding, k)

#     categories_results = []
#     for idx, dist in zip(indices_categories[0], distances_categories[0]):
#         # print(idx)
#         categories_results.append((category_codes[idx], category_items[idx], dist))

#     subcategories_results = []
#     for idx, dist in zip(indices_subcategories[0], distances_subcategories[0]):
#         subcategories_results.append((subcategory_codes[idx], subcategory_items[idx], dist))

#     # categories_desc_results = []
#     # for idx, dist in zip(indices_categories_desc[0], distances_categories_desc[0]):
#     #     categories_desc_results.append((category_codes[idx], category_desc_items[idx], dist))

#     # subcategories_desc_results = []
#     # for idx, dist in zip(indices_subcategories_desc[0], distances_subcategories_desc[0]):
#     #     subcategories_desc_results.append((subcategory_codes[idx], subcategory_desc_items[idx], dist))

#     return categories_results,subcategories_results,None,None

# def get_closest_deficiency_category(checklist_item):
#     if isinstance(checklist_item, float):
#         checklist_item = str(checklist_item)
#     categories_results,subcategories_results,categories_desc_results,subcategories_desc_results = find_closest_deficiency_category_subcategory(checklist_item, k=2)
#     return categories_results,subcategories_results,categories_desc_results,subcategories_desc_results

# def get_closest_deficiency_category_lambda(row):
#     # print(row)
#     checklist_item = row["Section"] +" - "+row["Question"]
#     if isinstance(checklist_item, float):
#         checklist_item = str(checklist_item)
#     categories_results,subcategories_results,categories_desc_results,subcategories_desc_results = find_closest_deficiency_category_subcategory(checklist_item, k=2)
#     category_code, category_title = categories_results[0][0],categories_results[0][1]
#     subcategory_code, subcategory_title = subcategories_results[0][0],subcategories_results[0][1]
#     return category_code, category_title,subcategory_code, subcategory_title


def find_closest_checklist_item(query, checks_internal, category_internal, checks_external, category_external,index_internal,index_external, k=3):
    """
    Given a deficiency description query, find the top-k matching deficiency codes.
    Returns a list of tuples: (Code, Description, Distance)
    """
    # Compute embedding for the query
    query_embedding = model.encode([query], convert_to_numpy=True)

    # Search the FAISS index for the nearest neighbors
    distances_internal, indices_internal = index_internal.search(query_embedding, k)
    distances_external, indices_external = index_external.search(query_embedding, k)

    internal_results = []
    for idx, dist in zip(indices_internal[0], distances_internal[0]):
        internal_results.append((category_internal[idx], checks_internal[idx], dist))

    external_results = []
    for idx, dist in zip(indices_external[0], distances_external[0]):
        external_results.append((category_external[idx], checks_external[idx], dist))

    return internal_results,external_results


# Sample function to calculate replacement value based on the row
def get_closest_checklist_items(nature_of_deficiency, checks_internal, category_internal, checks_external, category_external,index_internal,index_external):
    # For example, replace NaN in column 'A' with 10 times the value from column 'B'
    #return row['Nature of deficiency'] * 10
    #nature_of_deficiency = row['Nature of deficiency']
    # Check if nature_of_deficiency is a float and convert it to string if necessary
    if isinstance(nature_of_deficiency, float):
        nature_of_deficiency = str(nature_of_deficiency)
    internal_results,external_results = find_closest_checklist_item(nature_of_deficiency, checks_internal, category_internal, checks_external, category_external,index_internal,index_external, k=5)
    ranked_results_internal = rerank_cross_encoder(nature_of_deficiency, internal_results)
    ranked_results_external = rerank_cross_encoder(nature_of_deficiency, external_results)
    #historical_deficiency = ranked_results_internal[0][0]
    #historical_combo_text = ranked_results_internal[0][1]
    return ranked_results_internal,ranked_results_external

def rerank_cross_encoder(query, results):
    """
    Rerank candidates using a cross-encoder model.

    Args:
      query (str): The query string.
      candidate_texts (list[str]): A list of candidate texts.

    Returns:
      list of tuples: Each tuple contains (candidate_text, score) sorted descending.
    """
    # Initialize a pretrained cross-encoder model; change model name as needed.
    # model_rerank = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    # Create pairs of (query, candidate_text)
    candidate_checklist = [result[1] for result in results]
    candidate_category = [result[0] for result in results]
    #print("Checking")
    #print([(query, text[1]) for text in candidate_texts])
    pairs = [(query, text[1]) for text in candidate_checklist]
    scores = model_rerank.predict(pairs)
    ranked_candidates = sorted(zip(candidate_checklist,candidate_category, scores), key=lambda x: x[2], reverse=True)
    return ranked_candidates



def get_internal_external_checklist_items(internal_checklist,external_checklist,vessel_defects,index_internal,index_external):
    # Extract the descriptions and codes as lists
    checks_internal = internal_checklist["Question"].values
    category_internal = internal_checklist["Section"].values

    checks_external = external_checklist["Question"].values
    category_external = external_checklist["Section"].values

    # deficiency = vessel_defects["Nature of deficiency"].values[0]
    deficiency_recommendations = {}
    for deficiency in vessel_defects["Nature of deficiency"].values:
        ranked_results_internal,ranked_results_external = get_closest_checklist_items(deficiency, checks_internal, category_internal, checks_external, category_external,index_internal,index_external)
        deficiency_recommendations[deficiency] = {}
        deficiency_recommendations[deficiency]["internal"] = ranked_results_internal
        deficiency_recommendations[deficiency]["external"] = ranked_results_external
    return deficiency_recommendations


def load_recommendations_data():
    internaldf  = pd.read_csv('PSC Internal Checklist.csv', encoding='latin-1')
    internal_checklist = clean_internal_checklist(internaldf)
    external_checklist = pd.read_csv('PSC External Checklist.csv', encoding='latin-1')
    open_defects = pd.read_csv('Open Defects.csv', encoding='latin-1')
    deficiency_codes = pd.read_csv('PSC_Codes_Scored.csv', dtype={'Code': object})
    cols = ["Title", "Description"]
    deficiency_codes = deficiency_codes.dropna(subset=cols, how="all")
    
    open_defects.dropna(axis="columns", how="all",inplace=True)
    open_defects.dropna(axis="rows", how="all",inplace=True)
    open_defects.reset_index(drop=True, inplace=True)
    del open_defects["Unnamed: 0"]
    deficiency_codes_category_df = pd.DataFrame(list(category_dict.items()), columns=['Code', 'Title'])
    deficiency_codes_sub_category_df = deficiency_codes.dropna(subset=["Description"]).copy()
    
    # # sample a chunk of the file to guess the encoding
    # with open('PSC Recommendations.csv', 'rb') as f:
    #     raw = f.read(50_000)            # read first 50 kB
    
    # enc_guess = chardet.detect(raw)['encoding']
    # print(f"Detected encoding → {enc_guess}")
    
    psc_category_recommenders = pd.read_csv('PSC_Recommendations_clean.csv')
    # psc_category_recommenders = pd.read_csv('PSC Recommendations.csv',encoding='utf-8-sig')
    
    return internal_checklist,external_checklist,open_defects,deficiency_codes,deficiency_codes_category_df,deficiency_codes_sub_category_df,psc_category_recommenders

@st.cache_data(show_spinner=True)
def prepare_recommendations_data(internal_checklist,external_checklist,deficiency_codes):
    print("Prepare Recommendations Data")
    internal_checklist_items = (internal_checklist["Section"] +" - "+ internal_checklist["Question"]).values
    external_checklist_items = (external_checklist["Section"] +" - "+ external_checklist["Question"]).values
    
    category_items = deficiency_codes["Title"].values
    category_codes = deficiency_codes["Code"].values
    subcategory_items = deficiency_codes["Title"].values
    subcategory_codes = deficiency_codes["Code"].values
    category_desc_items = deficiency_codes["Title"].values
    subcategory_desc_items = deficiency_codes["Description"].astype(str).values
    return internal_checklist_items,external_checklist_items,category_items,subcategory_items,category_desc_items,subcategory_desc_items
    
@st.cache_data(show_spinner=True)
def create_indexes_and_embeddings(internal_checklist_items,external_checklist_items,category_items,
                                  subcategory_items,category_desc_items,subcategory_desc_items):
    
    
    print("Creating Indexes and Embeddings.....")
    embeddings_internal = model.encode(internal_checklist_items)
    embeddings_external = model.encode(external_checklist_items)
    embeddings_categories = model.encode(category_items)
    embeddings_subcategories = model.encode(subcategory_items)
    
    embeddings_category_desc = model.encode(category_desc_items)
    embeddings_subcategory_desc = model.encode(subcategory_desc_items)
    # embeddings = joblib.load("./data/historical_embeddings.joblib")
    
    # Build a FAISS index using L2 distance
    dimension = embeddings_internal.shape[1]
    index_internal = faiss.IndexFlatL2(dimension)
    index_internal.add(embeddings_internal)
    
    index_external = faiss.IndexFlatL2(dimension)
    index_external.add(embeddings_external)
    
    index_categories = faiss.IndexFlatL2(dimension)
    index_categories.add(embeddings_categories)
    
    index_subcategories = faiss.IndexFlatL2(dimension)
    index_subcategories.add(embeddings_subcategories)
    
    index_category_desc = faiss.IndexFlatL2(dimension)
    index_category_desc.add(embeddings_category_desc)
    
    index_subcategory_desc = faiss.IndexFlatL2(dimension)
    index_subcategory_desc.add(embeddings_subcategory_desc)
    
    return index_internal,index_external


def generate_open_defect_recommendations(internal_checklist,external_checklist,open_defects,index_internal,index_external,imo):
    vessel_defects = open_defects[open_defects["IMO_NO"] == imo]
    deficiency_recommendations = get_internal_external_checklist_items(internal_checklist,external_checklist,vessel_defects,index_internal,index_external)
    return deficiency_recommendations

    

