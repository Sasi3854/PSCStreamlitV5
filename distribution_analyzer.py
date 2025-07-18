import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, Any


class DistributionAnalyzer:
    """Handles category distribution analysis for different entities."""
    
    def __init__(self):
        """Initialize the distribution analyzer."""
        self.category_dict = {
            "01": "Certificates & Documentation",
            "02": "Structural condition",
            "03": "Water/Weathertight condition",
            "04": "Emergency Systems",
            "05": "Radio communication",
            "06": "Cargo operations including equipment",
            "07": "Fire safety",
            "08": "Alarms",
            "09": "Working and Living Conditions",
            "10": "Safety of Navigation",
            "11": "Life saving appliances",
            "12": "Dangerous Goods",
            "13": "Propulsion and auxiliary machinery",
            "14": "Pollution Prevention",
            "15": "ISM",
            "16": "ISPS",
            "18": "MLC, 2006",
            "99": "Other"
        }
        
    def setup_code_mappings(self, deficiency_codes_df: pd.DataFrame):
        """Setup code to category mappings."""
        self.deficiency_sub_category_to_category_mapping = {}
        self.code_subcategory_map = {}
        
        for _, row in deficiency_codes_df.iterrows():
            sub_category_str = str(row["Title"])
            code_str = str(row["Code"])
            
            if len(code_str) > 3:
                if len(code_str) == 4:
                    code_str = "0" + code_str
                category_code_str = code_str[:2]
                
                if category_code_str in self.category_dict:
                    category_str = self.category_dict[category_code_str]
                    self.deficiency_sub_category_to_category_mapping[sub_category_str] = category_str
                    self.code_subcategory_map[code_str] = sub_category_str
    
    def apply_code_correction(self, psc_code: str) -> str:
        """Correct PSC code format."""
        if len(psc_code) == 4:
            return "0" + psc_code
        return psc_code
    
    def extract_sub_category(self, psc_code: str) -> str:
        """Extract subcategory from PSC code."""
        if psc_code in self.code_subcategory_map:
            return self.code_subcategory_map[psc_code]
        return np.nan
    
    def extract_category(self, psc_code: str) -> str:
        """Extract category from PSC code."""
        cat = psc_code[:2]
        if cat in self.category_dict:
            return self.category_dict[cat]
        return np.nan
    
    def get_entity_category_distribution(self, df: pd.DataFrame, entity_column: str) -> Dict[str, pd.DataFrame]:
        """Get category distribution for a specific entity."""
        # Correct PSC codes
        df = df.copy()
        df["PSC_CODE"] = df["PSC_CODE"].apply(self.apply_code_correction)
        
        unique_entities = df[entity_column].unique()
        entity_category_distribution = {}
        
        for entity_item in unique_entities:
            entity_df = df[df[entity_column] == entity_item].copy()
            
            # Extract categories and subcategories
            entity_df["SUBCATEGORY"] = entity_df["PSC_CODE"].apply(self.extract_sub_category)
            entity_df["CATEGORY"] = entity_df["PSC_CODE"].apply(self.extract_category)
            
            # Drop rows with missing categories
            entity_df = entity_df.dropna(subset=["SUBCATEGORY", "CATEGORY"])
            
            if len(entity_df) > 0:
                category_counts = entity_df["CATEGORY"].value_counts()
                category_percentages = (category_counts / category_counts.sum()) * 100
                category_percentages_df = pd.DataFrame({
                    'Category': category_percentages.index,
                    'Percentage': category_percentages.values
                })
                
                entity_category_distribution[entity_item] = category_percentages_df#.head(5)
        
        return entity_category_distribution
    
    def plot_entity_distribution(self, entity_distribution: Dict[str, pd.DataFrame], 
                                entity_name: str, max_entities: int = 5):
        """Plot category distribution for entities."""
        entities_to_plot = list(entity_distribution.keys())[:max_entities]
        
        fig, axes = plt.subplots(len(entities_to_plot), 1, figsize=(12, 6 * len(entities_to_plot)))
        if len(entities_to_plot) == 1:
            axes = [axes]
        
        for i, entity in enumerate(entities_to_plot):
            data = entity_distribution[entity]
            sns.barplot(x='Category', y='Percentage', data=data, ax=axes[i])
            axes[i].set_title(f'Category Distribution for {entity_name}: {entity}')
            axes[i].set_xlabel('Category')
            axes[i].set_ylabel('Percentage')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def save_distributions(self, authority_dist: Dict, vessel_dist: Dict):
        """Save distributions to joblib files."""
        joblib.dump(authority_dist, "authority_category_distribution.joblib")
        joblib.dump(vessel_dist, "vessel_category_distribution.joblib")
    
    def load_distributions(self) -> tuple:
        """Load distributions from joblib files."""
        try:
            authority_dist = joblib.load("authority_category_distribution.joblib")
            vessel_dist = joblib.load("vessel_category_distribution.joblib")
            return authority_dist, vessel_dist
        except FileNotFoundError:
            return {}, {}
