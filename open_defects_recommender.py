# -*- coding: utf-8 -*-
"""
Open Defects Recommender

This module provides functionality to generate recommendations based on open defects
by finding similar items in internal and external checklists using semantic search.
"""

import pandas as pd
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import faiss
import joblib
import numpy as np

class OpenDefectsRecommender:
    """Handles recommendations based on open defects using semantic search."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the recommender with sentence transformer models."""
        self.model_sentence_transformer = SentenceTransformer(model_name)
        self.model_rerank = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize indices
        self.index_internal = None
        self.index_external = None
        self.index_categories = None
        self.index_subcategories = None
        self.index_category_desc = None
        self.index_subcategory_desc = None
        
        # Data storage
        self.internal_checklist = None
        self.external_checklist = None
        self.deficiency_codes = None
        self.category_items = None
        self.category_codes = None
        self.subcategory_items = None
        self.subcategory_codes = None
        self.category_desc_items = None
        self.subcategory_desc_items = None
        
    def clean_internal_checklist(self, path):
        """Clean and structure the internal checklist CSV file."""
        df = pd.read_csv(path, encoding='latin-1')
        df.dropna(axis="rows", how="all", inplace=True)
        df = df.iloc[:-5]
        
        # Turn first row into proper column names
        first = df.iloc[0]
        df.columns = [c if pd.isna(h) else str(h).strip() for c, h in zip(df.columns, first)]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Rename the first column to RAW
        df.rename(columns={df.columns[0]: "RAW"}, inplace=True)
        
        # Keep only relevant columns
        keep = {"RAW", "WEIGHTAGE", "RATING", "SCORE", "REMARKS"}
        extra = [c for c in df.columns if c not in keep and df[c].isna().all()]
        df.drop(columns=extra, inplace=True)
        
        def is_header(text: str) -> bool:
            """Return True if text is ALL CAPS (letters only)."""
            letters = re.sub(r"[^A-Za-z]+", "", text)
            return bool(letters) and letters.isupper()
        
        # Build tidy rows
        rows = []
        section_idx = 0
        q_counter = 0
        section = None
        
        for _, r in df.iterrows():
            raw = str(r["RAW"]).strip()
            if not raw:
                continue
                
            if is_header(raw):
                section_idx += 1
                section = raw.title()
                q_counter = 0
                continue
                
            if section is None:
                continue
                
            q_counter += 1
            q_no = f"{section_idx}.{q_counter}"
            
            rows.append({
                "Section": section,
                "Question No": q_no,
                "Question": raw,
                "WEIGHTAGE": r.get("WEIGHTAGE"),
                "RATING": r.get("RATING"),
                "SCORE": r.get("SCORE"),
                "REMARKS": r.get("REMARKS")
            })
        
        return pd.DataFrame(rows)
    
    def load_data(self, internal_checklist_path='PSC Internal Checklist.csv',
                  external_checklist_path='PSC External Checklist.csv',
                  deficiency_codes_path='PSC_Codes_Scored.csv'):
        """Load all required data files."""
        try:
            # Load internal checklist
            self.internal_checklist = self.clean_internal_checklist(internal_checklist_path)
            
            # Load external checklist
            self.external_checklist = pd.read_csv(external_checklist_path, encoding='latin-1')
            
            # Load deficiency codes
            self.deficiency_codes = pd.read_csv(deficiency_codes_path, dtype={'Code': object})
            cols = ["Title", "Description"]
            self.deficiency_codes = self.deficiency_codes.dropna(subset=cols, how="all")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def setup_embeddings_and_indices(self):
        """Set up embeddings and FAISS indices for semantic search."""
        if self.internal_checklist is None or self.external_checklist is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        # Prepare text data
        internal_checklist_items = (self.internal_checklist["Section"] + " - " + 
                                  self.internal_checklist["Question"]).values
        external_checklist_items = (self.external_checklist["Section"] + " - " + 
                                   self.external_checklist["Question"]).values
        
        self.category_items = self.deficiency_codes["Title"].values
        self.category_codes = self.deficiency_codes["Code"].values
        self.subcategory_items = self.deficiency_codes["Title"].values
        self.subcategory_codes = self.deficiency_codes["Code"].values
        self.category_desc_items = self.deficiency_codes["Title"].values
        self.subcategory_desc_items = self.deficiency_codes["Description"].astype(str).values
        
        # Generate embeddings
        embeddings_internal = self.model_sentence_transformer.encode(internal_checklist_items)
        embeddings_external = self.model_sentence_transformer.encode(external_checklist_items)
        embeddings_categories = self.model_sentence_transformer.encode(self.category_items)
        embeddings_subcategories = self.model_sentence_transformer.encode(self.subcategory_items)
        embeddings_category_desc = self.model_sentence_transformer.encode(self.category_desc_items)
        embeddings_subcategory_desc = self.model_sentence_transformer.encode(self.subcategory_desc_items)
        
        # Build FAISS indices
        dimension = embeddings_internal.shape[1]
        
        self.index_internal = faiss.IndexFlatL2(dimension)
        self.index_internal.add(embeddings_internal)
        
        self.index_external = faiss.IndexFlatL2(dimension)
        self.index_external.add(embeddings_external)
        
        self.index_categories = faiss.IndexFlatL2(dimension)
        self.index_categories.add(embeddings_categories)
        
        self.index_subcategories = faiss.IndexFlatL2(dimension)
        self.index_subcategories.add(embeddings_subcategories)
        
        self.index_category_desc = faiss.IndexFlatL2(dimension)
        self.index_category_desc.add(embeddings_category_desc)
        
        self.index_subcategory_desc = faiss.IndexFlatL2(dimension)
        self.index_subcategory_desc.add(embeddings_subcategory_desc)
    
    def find_closest_deficiency_category_subcategory(self, checklist, k=2):
        """Find closest deficiency categories and subcategories for a checklist item."""
        checklist_embedding = self.model_sentence_transformer.encode([checklist], convert_to_numpy=True)
        
        # Search indices
        distances_categories, indices_categories = self.index_categories.search(checklist_embedding, k)
        distances_subcategories, indices_subcategories = self.index_subcategories.search(checklist_embedding, k)
        distances_categories_desc, indices_categories_desc = self.index_category_desc.search(checklist_embedding, k)
        distances_subcategories_desc, indices_subcategories_desc = self.index_subcategory_desc.search(checklist_embedding, k)
        
        # Format results
        categories_results = []
        for idx, dist in zip(indices_categories[0], distances_categories[0]):
            categories_results.append((self.category_codes[idx], self.category_items[idx], dist))
        
        subcategories_results = []
        for idx, dist in zip(indices_subcategories[0], distances_subcategories[0]):
            subcategories_results.append((self.subcategory_codes[idx], self.subcategory_items[idx], dist))
        
        categories_desc_results = []
        for idx, dist in zip(indices_categories_desc[0], distances_categories_desc[0]):
            categories_desc_results.append((self.category_codes[idx], self.category_desc_items[idx], dist))
        
        subcategories_desc_results = []
        for idx, dist in zip(indices_subcategories_desc[0], distances_subcategories_desc[0]):
            subcategories_desc_results.append((self.subcategory_codes[idx], self.subcategory_desc_items[idx], dist))
        
        return categories_results, subcategories_results, categories_desc_results, subcategories_desc_results
    
    def get_closest_deficiency_category_lambda(self, row):
        """Get closest deficiency category for a checklist row."""
        checklist_item = row["Section"] + " - " + row["Question"]
        if isinstance(checklist_item, float):
            checklist_item = str(checklist_item)
        
        categories_results, subcategories_results, _, _ = self.find_closest_deficiency_category_subcategory(checklist_item, k=2)
        category_code, category_title = categories_results[0][0], categories_results[0][1]
        subcategory_code, subcategory_title = subcategories_results[0][0], subcategories_results[0][1]
        
        return category_code, category_title, subcategory_code, subcategory_title
    
    def find_closest_checklist_item(self, query, checks_internal, category_internal, 
                                   checks_external, category_external, k=3):
        """Find closest checklist items for a given query."""
        query_embedding = self.model_sentence_transformer.encode([query], convert_to_numpy=True)
        
        # Search indices
        distances_internal, indices_internal = self.index_internal.search(query_embedding, k)
        distances_external, indices_external = self.index_external.search(query_embedding, k)
        
        internal_results = []
        for idx, dist in zip(indices_internal[0], distances_internal[0]):
            internal_results.append((category_internal[idx], checks_internal[idx], dist))
        
        external_results = []
        for idx, dist in zip(indices_external[0], distances_external[0]):
            external_results.append((category_external[idx], checks_external[idx], dist))
        
        return internal_results, external_results
    
    def rerank_cross_encoder(self, query, results):
        """Rerank candidates using a cross-encoder model."""
        candidate_checklist = [result[1] for result in results]
        candidate_category = [result[0] for result in results]
        
        pairs = [(query, text[1]) for text in candidate_checklist]
        scores = self.model_rerank.predict(pairs)
        ranked_candidates = sorted(zip(candidate_checklist, candidate_category, scores), 
                                 key=lambda x: x[2], reverse=True)
        
        return ranked_candidates
    
    def get_closest_checklist_items(self, nature_of_deficiency, checks_internal, category_internal,
                                   checks_external, category_external):
        """Get closest checklist items with reranking."""
        if isinstance(nature_of_deficiency, float):
            nature_of_deficiency = str(nature_of_deficiency)
        
        internal_results, external_results = self.find_closest_checklist_item(
            nature_of_deficiency, checks_internal, category_internal, 
            checks_external, category_external, k=5)
        
        ranked_results_internal = self.rerank_cross_encoder(nature_of_deficiency, internal_results)
        ranked_results_external = self.rerank_cross_encoder(nature_of_deficiency, external_results)
        
        return ranked_results_internal, ranked_results_external
    
    def get_recommendations_for_vessel_defects(self, vessel_defects):
        """Get recommendations for vessel defects."""
        if self.internal_checklist is None or self.external_checklist is None:
            raise ValueError("Data must be loaded and indices set up first")
        
        # Extract checklist data
        checks_internal = self.internal_checklist["Question"].values
        category_internal = self.internal_checklist["Section"].values
        checks_external = self.external_checklist["Question"].values
        category_external = self.external_checklist["Section"].values
        
        # Get deficiency description
        deficiency = vessel_defects["Nature of deficiency"].values[0]
        
        # Get recommendations
        ranked_results_internal, ranked_results_external = self.get_closest_checklist_items(
            deficiency, checks_internal, category_internal, checks_external, category_external)
        
        return ranked_results_internal, ranked_results_external
    
    def update_checklists_with_categories(self):
        """Update internal and external checklists with corresponding categories and subcategories."""
        if self.internal_checklist is None or self.external_checklist is None:
            raise ValueError("Data must be loaded first")
        
        # Update internal checklist
        self.internal_checklist[["Category Code", "Category Title", "Subcategory Code", "Subcategory Title"]] = \
            self.internal_checklist.apply(lambda q: pd.Series(self.get_closest_deficiency_category_lambda(q)), axis=1)
        
        # Update external checklist
        self.external_checklist[["Category Code", "Category Title", "Subcategory Code", "Subcategory Title"]] = \
            self.external_checklist.apply(lambda q: pd.Series(self.get_closest_deficiency_category_lambda(q)), axis=1)
    
    def get_recommendations_for_imo(self, imo_number, open_defects_df):
        """Get recommendations for a specific IMO number."""
        vessel_defects = open_defects_df[open_defects_df["IMO_NO"] == imo_number]
        
        if vessel_defects.empty:
            return None, None
        
        return self.get_recommendations_for_vessel_defects(vessel_defects)
