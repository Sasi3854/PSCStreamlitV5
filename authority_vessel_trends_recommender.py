# -*- coding: utf-8 -*-
"""
Authority and Vessel Trends Recommender

This module provides functionality to generate recommendations based on authority trends
and vessel trends using category distribution analysis.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple

class AuthorityVesselTrendsRecommender:
    """Handles recommendations based on authority and vessel trends."""
    
    def __init__(self):
        """Initialize the recommender."""
        self.authority_category_distribution = None
        self.vessel_category_distribution = None
        self.internal_checklist = None
        self.external_checklist = None
        self.deficiency_codes = None
        
        # Category mapping dictionary
        self.category_dict = {
            "00001": "Certificates & Documentation",
            "00002": "Structural condition",
            "00003": "Water/Weathertight condition",
            "00004": "Emergency Systems",
            "00005": "Radio communication",
            "00006": "Cargo operations including equipment",
            "00007": "Fire safety",
            "00008": "Alarms",
            "00009": "Working and Living Conditions",
            "00010": "Safety of Navigation",
            "00011": "Life saving appliances",
            "00012": "Dangerous Goods",
            "00013": "Propulsion and auxiliary machinery",
            "00014": "Pollution Prevention",
            "00015": "ISM",
            "00016": "ISPS",
            "00018": "MLC, 2006",
            "00099": "Other"
        }
    
    def load_distribution_data(self, authority_dist_path="./authority_category_distribution.joblib",
                              vessel_dist_path="./vessel_category_distribution.joblib"):
        """Load authority and vessel category distribution data."""
        try:
            self.authority_category_distribution = joblib.load(authority_dist_path)
            self.vessel_category_distribution = joblib.load(vessel_dist_path)
            return True
        except Exception as e:
            print(f"Error loading distribution data: {e}")
            return False
    
    def load_checklist_data(self, internal_checklist, external_checklist):
        """Load checklist data (should be already processed with categories)."""
        self.internal_checklist = internal_checklist.copy()
        self.external_checklist = external_checklist.copy()
        
        # Rename columns for consistency
        if 'Category Title' in self.internal_checklist.columns:
            self.internal_checklist.rename(columns={'Category Title': 'Category'}, inplace=True)
        if 'Category Title' in self.external_checklist.columns:
            self.external_checklist.rename(columns={'Category Title': 'Category'}, inplace=True)
    
    def get_top_categories_for_authority_vessel(self, authority: str, vessel: int, top_n: int = 3) -> pd.DataFrame:
        """Get top categories based on authority and vessel trends."""
        if (self.authority_category_distribution is None or 
            self.vessel_category_distribution is None):
            raise ValueError("Distribution data must be loaded first")
        
        # Check if authority and vessel exist in the data
        if authority not in self.authority_category_distribution:
            raise ValueError(f"Authority '{authority}' not found in distribution data")
        
        if vessel not in self.vessel_category_distribution:
            raise ValueError(f"Vessel '{vessel}' not found in distribution data")
        
        # Get distribution data
        df_auth = self.authority_category_distribution[authority]
        df_vsl = self.vessel_category_distribution[vessel]
        
        # Merge and calculate combined score
        df_mix = df_auth.merge(df_vsl, on="Category", how="outer").fillna(0)
        df_mix.columns = ["Category", "Authority Trends", "Vessel Trends"]
        df_mix["Score"] = (df_mix["Authority Trends"] + df_mix["Vessel Trends"]) / 2  # simple average
        
        # Sort by descending score
        df_mix = df_mix.sort_values("Score", ascending=False)
        
        return df_mix.head(top_n)
    
    def get_checklist_recommendations(self, authority: str, vessel: int, top_n: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get checklist recommendations based on top categories."""
        if self.internal_checklist is None or self.external_checklist is None:
            raise ValueError("Checklist data must be loaded first")
        
        # Get top categories
        top_cats_df = self.get_top_categories_for_authority_vessel(authority, vessel, top_n)
        top_cats = top_cats_df["Category"].tolist()
        
        # Filter checklists by top categories
        selected_q_internal = self.internal_checklist[
            self.internal_checklist["Category"].isin(top_cats)
        ].copy()
        
        selected_q_external = self.external_checklist[
            self.external_checklist["Category"].isin(top_cats)
        ].copy()
        
        # Merge with importance scores
        selected_q_internal = selected_q_internal.merge(
            top_cats_df[["Category", "Score"]], on="Category"
        )
        selected_q_external = selected_q_external.merge(
            top_cats_df[["Category", "Score"]], on="Category"
        )
        
        # Sort by score (descending)
        selected_q_internal = selected_q_internal.sort_values("Score", ascending=False)
        selected_q_external = selected_q_external.sort_values("Score", ascending=False)
        
        return selected_q_internal, selected_q_external
    
    def get_available_authorities(self) -> List[str]:
        """Get list of available authorities."""
        if self.authority_category_distribution is None:
            return []
        return list(self.authority_category_distribution.keys())
    
    def get_available_vessels(self) -> List[int]:
        """Get list of available vessels."""
        if self.vessel_category_distribution is None:
            return []
        return list(self.vessel_category_distribution.keys())
    
    def get_authority_vessel_overlap(self, open_defects_df: pd.DataFrame) -> np.ndarray:
        """Get overlap between vessels in open defects and vessel distribution data."""
        if self.vessel_category_distribution is None:
            return np.array([])
        
        vessel_ids_open = open_defects_df["IMO_NO"].unique()
        vessel_ids_distribution = list(self.vessel_category_distribution.keys())
        
        return np.intersect1d(vessel_ids_open, vessel_ids_distribution)
    
    def analyze_category_trends(self, authority: str, vessel: int) -> Dict:
        """Analyze detailed category trends for authority and vessel."""
        if (self.authority_category_distribution is None or 
            self.vessel_category_distribution is None):
            raise ValueError("Distribution data must be loaded first")
        
        # Get distribution data
        df_auth = self.authority_category_distribution.get(authority, pd.DataFrame())
        df_vsl = self.vessel_category_distribution.get(vessel, pd.DataFrame())
        
        if df_auth.empty or df_vsl.empty:
            return {}
        
        # Merge data
        df_mix = df_auth.merge(df_vsl, on="Category", how="outer").fillna(0)
        df_mix.columns = ["Category", "Authority Trends", "Vessel Trends"]
        df_mix["Score"] = (df_mix["Authority Trends"] + df_mix["Vessel Trends"]) / 2
        df_mix = df_mix.sort_values("Score", ascending=False)
        
        # Calculate statistics
        analysis = {
            "total_categories": len(df_mix),
            "top_category": df_mix.iloc[0]["Category"] if len(df_mix) > 0 else None,
            "top_score": df_mix.iloc[0]["Score"] if len(df_mix) > 0 else 0,
            "authority_dominant_categories": df_mix[df_mix["Authority Trends"] > df_mix["Vessel Trends"]]["Category"].tolist(),
            "vessel_dominant_categories": df_mix[df_mix["Vessel Trends"] > df_mix["Authority Trends"]]["Category"].tolist(),
            "balanced_categories": df_mix[df_mix["Authority Trends"] == df_mix["Vessel Trends"]]["Category"].tolist(),
            "category_distribution": df_mix.to_dict('records')
        }
        
        return analysis
    
    def get_recommendations_summary(self, authority: str, vessel: int, top_n: int = 3) -> Dict:
        """Get a comprehensive summary of recommendations."""
        try:
            # Get top categories
            top_cats_df = self.get_top_categories_for_authority_vessel(authority, vessel, top_n)
            
            # Get checklist recommendations
            internal_recs, external_recs = self.get_checklist_recommendations(authority, vessel, top_n)
            
            # Get trend analysis
            trend_analysis = self.analyze_category_trends(authority, vessel)
            
            summary = {
                "authority": authority,
                "vessel": vessel,
                "top_categories": top_cats_df.to_dict('records'),
                "internal_recommendations_count": len(internal_recs),
                "external_recommendations_count": len(external_recs),
                "trend_analysis": trend_analysis,
                "recommendation_generated": True
            }
            
            return summary
            
        except Exception as e:
            return {
                "authority": authority,
                "vessel": vessel,
                "error": str(e),
                "recommendation_generated": False
            }
