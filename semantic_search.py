import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Tuple


class SemanticSearch:
    """Handles semantic search functionality for deficiency codes."""
    
    def __init__(self, model_name="all-mpnet-base-v2"):
        """Initialize the semantic search with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.index = None
        self.descriptions = None
        self.codes = None
        
    def build_index(self, df: pd.DataFrame, description_column: str = "ISSUE_DETAILS", 
                   code_column: str = "PSC_CODE"):
        """Build FAISS index for semantic search."""
        self.descriptions = df[description_column].tolist()
        self.codes = df[code_column].tolist()
        
        # Compute embeddings
        embeddings = self.model.encode(self.descriptions, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def find_deficiency_code(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Find top-k matching deficiency codes for a query."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        # Compute embedding for the query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append((self.codes[idx], self.descriptions[idx], dist))
        
        return results
    
    def rerank_cross_encoder(self, query: str, candidate_texts: List[str]) -> List[Tuple[str, float]]:
        """Rerank candidates using a cross-encoder model."""
        # Create pairs of (query, candidate_text)
        pairs = [(query, text) for text in candidate_texts]
        scores = self.rerank_model.predict(pairs)
        
        # Sort by score (descending)
        ranked_candidates = sorted(
            zip(candidate_texts, scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked_candidates
    
    def search_and_rerank(self, query: str, k: int = 10, rerank_top: int = 3) -> List[Tuple[str, str, float]]:
        """Search and rerank results using cross-encoder."""
        # Get initial results
        initial_results = self.find_deficiency_code(query, k)
        
        # Extract descriptions for reranking
        candidate_texts = [desc for _, desc, _ in initial_results]
        
        # Rerank
        reranked = self.rerank_cross_encoder(query, candidate_texts)
        
        # Map back to codes and return top results
        final_results = []
        for i, (desc, score) in enumerate(reranked[:rerank_top]):
            # Find corresponding code
            for code, orig_desc, _ in initial_results:
                if orig_desc == desc:
                    final_results.append((code, desc, score))
                    break
        
        return final_results
