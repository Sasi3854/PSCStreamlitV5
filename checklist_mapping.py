#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:57:37 2025

@author: sasidharankumar
"""

"""
Map reframed checklist questions to PSC Subcategories using sentence embeddings.

Install:
    pip install -U pandas sentence-transformers scikit-learn tqdm

Run:
    python map_questions_to_psc.py \
        --questions "/path/PSC_Internal_Checklist_Refined.csv" \
        --psc "/path/PSC_Codes_Cleaned_Formatted.csv" \
        --out_map "/path/Question_to_PSC_Subcategory.csv" \
        --out_consolidated "/path/PSC_Consolidated_With_Mapped_Questions.csv" \
        --model "BAAI/bge-large-en-v1.5" \
        --threshold 0.35 \
        --topk 3

Notes:
- For multilingual content, consider: --model "intfloat/multilingual-e5-large-instruct"
- Tune --threshold (0.30–0.45 typical). Increase to reduce weak matches.
"""

import argparse
import re
import json
from typing import List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# ----------------------------- Helpers -----------------------------

def normalize_text(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def build_psc_corpus(row: pd.Series) -> Tuple[str, str]:
    """
    Returns (id_text, full_corpus_text). We will treat Title as the "subcategory name".
    """
    fields = [
        "Title",
        "Description",
        "Paris MOU Recommendations",
        "AMSA Trends",
        "AMSA Recommendations",
        "Tokyo MOU Trends",
        "Tokyo MOU Recommendations",
        "Paris MOU Trends",
    ]
    title = normalize_text(row.get("Title", ""))
    parts = [normalize_text(row.get(f, "")) for f in fields]
    full = " ".join([p for p in parts if p])
    return title, full or title  # fall back to title if description empty

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
    return out


# ----------------------------- Core -----------------------------

def main():
    # Load data
    qdf = pd.read_csv("PSC_External_Checklist_Refined.csv")
    pdf = pd.read_csv("PSC_Consolidated_With_Mapped_Questions.csv")

    # Validate columns
    if "Checklist Question" not in qdf.columns:
        raise ValueError("Questions file must contain column: 'Checklist Question'")

    # Build PSC subcategory corpus
    psc_titles = []
    psc_corpus = []
    keep_idx = []
    for i, row in pdf.iterrows():
        title, corpus = build_psc_corpus(row)
        if title:  # keep rows that have a subcategory title
            psc_titles.append(title)
            psc_corpus.append(corpus)
            keep_idx.append(i)

    # Optionally keep only needed rows
    psc_df = pdf.iloc[keep_idx].reset_index(drop=True)

    # Prepare model
    model_name = "BAAI/bge-large-en-v1.5"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    # Model-specific note for BGE:
    # BGE models recommend prefix "query: " for queries and "passage: " for documents to improve performance.
    # We'll apply that if model contains "bge".
    use_bge_prefix = "bge" in model_name.lower()

    # Encode PSC subcategories (documents/passages)
    passages = [f"passage: {c}" if use_bge_prefix else c for c in psc_corpus]
    emb_psc = model.encode(passages, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)

    # Encode questions (queries)
    questions = [normalize_text(x) for x in qdf["Checklist Question"].astype(str).tolist()]
    queries = [f"query: {q}" if use_bge_prefix else q for q in questions]
    emb_q = model.encode(queries, batch_size=64, convert_to_tensor=True, normalize_embeddings=True)

    # Similarity matrix (cosine)
    sim = util.cos_sim(emb_q, emb_psc).cpu().numpy()  # shape: (nQ, nPSC)

    # For each question, choose matches:
    #  - Keep up to topk
    #  - Filter by threshold
    topk = 3
    threshold = 0.6

    all_matches = []
    for qi, q in enumerate(questions):
        row_scores = sim[qi]
        top_idx = np.argsort(row_scores)[::-1][:topk]
        picks = []
        for j in top_idx:
            score = float(row_scores[j])
            if score >= threshold:
                picks.append((psc_titles[j], score))
        # collapse
        titles = [t for t, _ in picks]
        scores = [f"{s:.3f}" for _, s in picks]
        if not titles:
            titles = ["Others"]
            scores = [""]

        all_matches.append({
            "Checklist Question": q,
            "Matched Subcategories": " # ".join(dedupe_keep_order(titles)),
            "Scores": " # ".join(scores)
        })

    map_df = pd.DataFrame(all_matches)

    # Save question -> subcategory mapping
    out_map = "External_Question_to_PSC_Subcategory.csv" 
    map_df.to_csv(out_map, index=False)
    print(f"Saved question-to-subcategory map: {out_map}")

    # Optional: produce a consolidated PSC table with a new "Mapped Questions" column
    # if args.out_consolidated:
    if True:
        psc_df["Mapped Questions"] = ""

        # Build reverse index: for each subcategory title, collect questions
        from collections import defaultdict
        bucket = defaultdict(list)
        for _, r in map_df.iterrows():
            titles = [t.strip() for t in str(r["Matched Subcategories"]).split("#")]
            qtext = r["Checklist Question"]
            for t in titles:
                if t and t != "Others":
                    bucket[t].append(qtext)

        # Attach to PSC rows
        for i, title in enumerate(psc_df["Title"].astype(str)):
            qs = bucket.get(title, [])
            if qs:
                psc_df.at[i, "Mapped Questions"] = " # ".join(dedupe_keep_order(qs))

        # Add Others row if any questions were unmatched
        unmatched_qs = [r["Checklist Question"] for _, r in map_df.iterrows()
                        if r["Matched Subcategories"] == "Others"]
        if unmatched_qs:
            others_row = {
                "Code": "OTH",
                "Title": "Others",
                "Description": "Uncategorized questions (embedding score below threshold or no good match)",
                "Paris MOU Recommendations": "",
                "AMSA Trends": "",
                "AMSA Recommendations": "",
                "Tokyo MOU Trends": "",
                "Tokyo MOU Recommendations": "",
                "Paris MOU Trends": "",
                "Mapped Questions": " # ".join(dedupe_keep_order(unmatched_qs))
            }
            psc_df = pd.concat([psc_df, pd.DataFrame([others_row])], ignore_index=True)

        psc_out = "PSC_Consolidated_With_Mapped_Questions_External.csv"
        psc_df.to_csv(psc_out, index=False)
        print(f"Saved consolidated PSC with mapped questions: {psc_out}")

    # Summary
    n_total = len(questions)
    n_unmatched = sum(1 for _, r in map_df.iterrows() if r["Matched Subcategories"] == "Others")
    print(json.dumps({
        "total_questions": n_total,
        "threshold": threshold,
        "topk": topk,
        "unmatched": n_unmatched,
        "match_rate_%": round(100*(n_total-n_unmatched)/max(1,n_total), 2)
    }, indent=2))


if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--questions", required=True, help="Path to PSC_Internal_Checklist_Refined.csv")
    # ap.add_argument("--psc", required=True, help="Path to PSC_Codes_Cleaned_Formatted.csv")
    # ap.add_argument("--out_map", required=True, help="Output CSV: Question_to_PSC_Subcategory.csv")
    # ap.add_argument("--out_consolidated", default="", help="Optional: consolidated PSC CSV with 'Mapped Questions'")
    # ap.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="Embedding model name")
    # ap.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity threshold")
    # ap.add_argument("--topk", type=int, default=3, help="Max subcategories per question")
    # ap.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    # args = ap.parse_args()
    main()
