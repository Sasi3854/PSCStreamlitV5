"""Main PSC Analysis class that orchestrates the entire analysis."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from data_loader import DataLoader
from text_normalizer import TextNormalizer
from action_code_parser import ActionCodeParser
from data_enricher import DataEnricher
from risk_calculator import RiskCalculator
from vessel_analyzer import VesselAnalyzer
from visualization import Visualizer
from config import ISSUE_HALF_LIFE, WACTION


class PSCAnalyzer:
    """Main class for PSC incident analysis."""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.text_normalizer = TextNormalizer()
        self.action_parser = ActionCodeParser()
        self.visualizer = Visualizer()
        
        # Data containers
        self.raw_data = {}
        self.analysis_df = None
        self.data_enricher = None
        self.risk_calculator = None
        self.vessel_analyzer = None
        
        # Results containers
        self.entity_results = {}
        self.vessel_scores = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess all data files."""
        print("Loading data files...")
        self.raw_data = self.data_loader.load_all_data()
        
        print("Preprocessing inspection data...")
        inspection_data = self.data_loader.preprocess_inspection_data()
        
        # Calculate issue weights
        lambda_val = np.log(2) / ISSUE_HALF_LIFE
        today = pd.Timestamp.today()
        
        def get_issue_weight(row):
            age_d = (today - row['INSPECTION_FROM_DATE']).days
            return np.exp(-lambda_val * age_d)
        
        inspection_data["ISSUE_WEIGHT"] = inspection_data.apply(get_issue_weight, axis=1)
        
        # Initialize enricher
        self.data_enricher = DataEnricher(
            self.raw_data['generic_factors'],
            self.raw_data['psc_codes_scoring']
        )
        
        return inspection_data
    
    def enrich_analysis_data(self, inspection_data: pd.DataFrame) -> pd.DataFrame:
        """Enrich inspection data with additional information."""
        print("Enriching analysis data...")
        
        # Select relevant columns
        analysis_df = inspection_data[[
            "IMO_NO", "AUTHORITY", "NATURE_OF_DEFICIENCY", "PSC_CODE", "R_OWNERS",
            "VESSEL_TYPE", "YARD", "AGE_OF_VESSEL", "FLAG_STATE", "DETENTION",
            "MANAGER_GROUP", "REFERENCE_CODE_1", "NATIONALITY_OF_THE_CREW",
            "ISSUE_WEIGHT", "TECHNICAL_MANAGER", "MARINE_SUPERINTENDENT",
            "MARINE_MANAGER", "INSPECTOR", "INSPECTION_FROM_DATE"
        ]].copy()
        
        # Add enriched data
        analysis_df["SEVERITY"] = analysis_df.apply(self.data_enricher.get_severity, axis=1)
        analysis_df["ISSUE_CLASSIFICATION"] = analysis_df.apply(self.data_enricher.get_classification, axis=1)
        analysis_df["VESSEL_CLASS"] = analysis_df.apply(self.data_enricher.get_vessel_class, axis=1)
        analysis_df["ME_MAKE"] = analysis_df.apply(self.data_enricher.get_me_make, axis=1)
        analysis_df["ME_MODEL"] = analysis_df.apply(self.data_enricher.get_me_model, axis=1)
        analysis_df["YARD"] = analysis_df.apply(self.data_enricher.get_yard_name, axis=1)
        analysis_df["YARD_COUNTRY"] = analysis_df.apply(self.data_enricher.get_yard_country, axis=1)
        analysis_df["VESSEL_TYPE"] = analysis_df.apply(self.data_enricher.get_vessel_type, axis=1)
        analysis_df["VESSEL_SUBTYPE"] = analysis_df.apply(self.data_enricher.get_vessel_subtype, axis=1)
        analysis_df["AGE_OF_VESSEL"] = analysis_df.apply(self.data_enricher.get_vessel_age, axis=1)
        
        return analysis_df
    
    def normalize_text_fields(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text fields using fuzzy matching."""
        print("Normalizing text fields...")
        
        # Create mappings for all text fields
        text_fields = [
            'R_OWNERS', 'AUTHORITY', 'VESSEL_TYPE', 'YARD', 'YARD_COUNTRY',
            'FLAG_STATE', 'MANAGER_GROUP', 'NATIONALITY_OF_THE_CREW',
            'TECHNICAL_MANAGER', 'MARINE_SUPERINTENDENT', 'MARINE_MANAGER',
            'INSPECTOR', 'ME_MAKE', 'ME_MODEL', 'VESSEL_SUBTYPE', 'VESSEL_CLASS'
        ]
        
        for field in text_fields:
            if field in analysis_df.columns:
                mapping = self.text_normalizer.normalize_strings(analysis_df[field], threshold=90)
                analysis_df[field] = analysis_df[field].map(mapping)
        
        return analysis_df
    
    def calculate_severity_scores(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate final severity scores."""
        print("Calculating severity scores...")
        
        # Parse action codes
        analysis_df["ActionCodes"] = analysis_df["REFERENCE_CODE_1"].apply(
            self.action_parser.parse_action_codes
        )
        
        # Calculate action code severity
        analysis_df["ACTIONCODE_SEVERITY"] = analysis_df["ActionCodes"].apply(
            self.action_parser.max_weight
        )
        
        # Calculate final severity
        analysis_df['FINAL_SEVERITY'] = (
            WACTION * analysis_df["ACTIONCODE_SEVERITY"] + 
            ((1 - WACTION) * analysis_df['SEVERITY'] * 10)
        )
        
        # Calculate issue base score
        analysis_df["ISSUE_BASESCORE"] = analysis_df["FINAL_SEVERITY"] * analysis_df["ISSUE_WEIGHT"]
        
        return analysis_df
    
    def analyze_entities(self, analysis_df: pd.DataFrame):
        """Analyze risk for all entity types."""
        print("Analyzing entity risks...")
        
        self.risk_calculator = RiskCalculator(analysis_df)
        
        # Define entity analysis dataframes
        entity_configs = [
            ('owners', 'R_OWNERS'),
            ('yard', 'YARD'),
            ('flag', 'FLAG_STATE'),
            ('manager', 'MANAGER_GROUP'),
            #('crew', 'NATIONALITY_OF_THE_CREW'),
            #('inspector', 'INSPECTOR'),
            #('marine_manager', 'MARINE_MANAGER'),
            #('marine_superintendent', 'MARINE_SUPERINTENDENT'),
            #('technical_manager', 'TECHNICAL_MANAGER'),
            ('make', 'ME_MAKE'),
            ('model', 'ME_MODEL'),
            ('class', 'VESSEL_CLASS')
        ]
        
        for entity_name, entity_col in entity_configs:
            # Create entity-specific dataframe
            entity_df = analysis_df[
                ["IMO_NO", "AUTHORITY", "FINAL_SEVERITY", entity_col, 
                 "ISSUE_CLASSIFICATION", "ISSUE_WEIGHT", "ISSUE_BASESCORE"]
            ].copy()
            entity_df = entity_df[entity_df[entity_col] != '']
            
            # Analyze entity
            analysis_results = self.risk_calculator.analyze_entity(entity_df, entity_col)
            
            # Convert to dataframe and sort
            results_df = pd.DataFrame(analysis_results)
            if not results_df.empty:
                results_df.sort_values(by="Risk", inplace=True, ascending=False)
                results_df.reset_index(drop=True, inplace=True)
            
            self.entity_results[entity_name] = results_df
    
    def analyze_vessels(self, analysis_df: pd.DataFrame):
        """Analyze vessel-specific risks."""
        print("Analyzing vessel risks...")
        
        self.vessel_analyzer = VesselAnalyzer(analysis_df, self.raw_data['dynamic_factors'])
        
        # Get vessel scores
        historical_scores = self.vessel_analyzer.get_vessel_historical_scores()
        change_scores = self.vessel_analyzer.get_vessel_change_scores()
        
        # Create vessel profile mapping
        unique_vessels = pd.unique(analysis_df["IMO_NO"])
        vessel_profile = {}
        
        for vessel in unique_vessels:
            row = analysis_df[analysis_df["IMO_NO"] == vessel].iloc[0]
            vessel_profile[vessel] = {
                "Owners": row["R_OWNERS"],
                "Yard": row["YARD"],
                "Flag": row["FLAG_STATE"],
                "Manager": row["MANAGER_GROUP"],
                "Crew": row["NATIONALITY_OF_THE_CREW"],
                "Inspector": row["INSPECTOR"],
                "Marine Manager": row["MARINE_MANAGER"],
                "Marine Superintendent": row["MARINE_SUPERINTENDENT"],
                "Technical Manager": row["TECHNICAL_MANAGER"],
                "Make": row["ME_MAKE"],
                "Model": row["ME_MODEL"],
                "Class": row["VESSEL_CLASS"]
            }
        
        # Calculate final vessel scores
        self._calculate_final_vessel_scores(
            unique_vessels, vessel_profile, historical_scores, change_scores
        )
    
    def _calculate_final_vessel_scores(self, unique_vessels: List[int], vessel_profile: Dict,
                                     historical_scores: Dict, change_scores: Dict):
        """Calculate final vessel risk scores."""
        # Create risk score mappings from entity results
        risk_mappings = {}
        for entity_name, results_df in self.entity_results.items():
            if not results_df.empty:
                risk_mappings[entity_name] = results_df[["Entity Name", "Risk"]].set_index('Entity Name')["Risk"].to_dict()
            else:
                risk_mappings[entity_name] = {}
        
        # Calculate vessel scores
        vessel_scores = {}
        for vessel in unique_vessels:
            vessel_scores[vessel] = {
                "Change Score": change_scores.get(int(vessel), 0),
                "Historical Score": historical_scores.get(int(vessel), 0)
            }
            
            # Add entity risk scores
            entity_mappings = [
                ("Owners", "owners", "Owners Risk Score"),
                ("Yard", "yard", "Yard Risk Score"),
                ("Flag", "flag", "Flag Risk Score"),
                ("Manager", "manager", "Manager Risk Score"),
                #("Crew", "crew", "Crew Risk Score"),
                #("Inspector", "inspector", "Inspector Risk Score"),
                #("Marine Manager", "marine_manager", "Marine Manager Risk Score"),
                #("Marine Superintendent", "marine_superintendent", "Marine Superintendent Score"),
                #("Technical Manager", "technical_manager", "Technical Manager Risk Score"),
                ("Make", "make", "ME Make Risk Score"),
                ("Model", "model", "ME Model Risk Score"),
                ("Class", "class", "Class Risk Score")
            ]
            
            for profile_key, entity_key, score_key in entity_mappings:
                entity_value = vessel_profile[vessel][profile_key]
                vessel_scores[vessel][score_key] = risk_mappings[entity_key].get(entity_value, 0)
        
        self.vessel_scores = vessel_scores
    
    def run_full_analysis(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Run the complete PSC analysis pipeline."""
        print("Starting PSC Analysis...")
        
        # Load and preprocess data
        inspection_data = self.load_and_preprocess_data()
        
        # Enrich data
        analysis_df = self.enrich_analysis_data(inspection_data)
        
        # Normalize text fields
        analysis_df = self.normalize_text_fields(analysis_df)
        
        # Calculate severity scores
        analysis_df = self.calculate_severity_scores(analysis_df)
        
        # Store analysis dataframe
        self.analysis_df = analysis_df
        
        # Analyze entities
        self.analyze_entities(analysis_df)
        
        # Analyze vessels
        self.analyze_vessels(analysis_df)
        
        self.authorities = pd.unique(analysis_df["AUTHORITY"])
        
        print("Analysis complete!")
        
        return analysis_df, self.entity_results, self.vessel_scores, self.authorities
    
    def get_entity_results(self, entity_type: str) -> pd.DataFrame:
        """Get results for a specific entity type."""
        return self.entity_results.get(entity_type, pd.DataFrame())
    
    def get_vessel_scores_df(self) -> pd.DataFrame:
        """Get vessel scores as a DataFrame."""
        df = pd.DataFrame(self.vessel_scores).T
        df.index = df.index.astype('str')

        return df
    
    def create_visualizations(self, entity_type: str = 'owners'):
        """Create visualizations for the specified entity type."""
        if entity_type not in self.entity_results:
            print(f"No results found for entity type: {entity_type}")
            return None
        
        results_df = self.entity_results[entity_type]
        if results_df.empty:
            print(f"No data available for entity type: {entity_type}")
            return None
        
        # Create 2D scatter plot
        fig_2d = self.visualizer.create_risk_scatter_plot(
            results_df.copy(), 
            f"{entity_type.title()} Risk Analysis: Actual Issues vs. Expected"
        )
        
        # Create 3D scatter plot
        fig_3d = self.visualizer.create_3d_risk_plot(
            results_df.copy(),
            f"{entity_type.title()} 3D Risk Analysis"
        )
        
        return fig_2d, fig_3d
