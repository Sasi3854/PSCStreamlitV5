import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import umap
import hdbscan
import warnings
warnings.filterwarnings("ignore")


class TopicAnalyzer:
    """Handles topic modeling analysis using BERTopic."""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize the topic analyzer with a sentence transformer model."""
        self.model = SentenceTransformer(model_name)
        self.topic_model = None
        self.embeddings = None
        
    def create_topic_model(self):
        """Create and configure the BERTopic model."""
        umap_model = umap.UMAP(
            n_neighbors=5, 
            n_components=10, 
            min_dist=0.3, 
            metric='cosine', 
            random_state=42
        )
        
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=3, 
            metric='euclidean', 
            cluster_selection_method='eom', 
            prediction_data=True
        )
        
        rep_model = KeyBERTInspired(top_n_words=8)
        
        self.topic_model = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdb,
            representation_model=rep_model,
            embedding_model=self.model,
            calculate_probabilities=False,
            verbose=False
        )
        
    def analyze_topics(self, df: pd.DataFrame, text_column: str = 'ISSUE_DETAILS'):
        """Perform topic analysis on the given dataframe."""
        if self.topic_model is None:
            self.create_topic_model()
            
        print("Generating Embeddings.....")
        # Generate embeddings
        self.embeddings = self.model.encode(df[text_column], show_progress_bar=False)
        print("Creating Topics.......")
        # Fit topic model
        topics, _ = self.topic_model.fit_transform(df[text_column], self.embeddings)
        
        # Add topics to dataframe
        df_copy = df.copy()
        df_copy['TOPIC'] = topics
        
        # Get topic information
        topic_df = self.topic_model.get_topic_info()
        topic_representation_dict = topic_df[["Topic", "Representation"]].set_index("Topic")["Representation"].to_dict()
        df_copy["TOPIC_NAMES"] = df_copy["TOPIC"].map(topic_representation_dict)
        print("Topics Created.....")
        print(df_copy)
        return df_copy, topic_df
    
    def visualize_topics(self, df: pd.DataFrame, text_column: str = 'ISSUE_DETAILS'):
        """Create topic visualizations."""
        if self.topic_model is None:
            raise ValueError("Topic model not fitted. Call analyze_topics first.")
            
        # Bar chart visualization
        barchart_fig = self.topic_model.visualize_barchart()
        
        # Document visualization
        doc_fig = self.topic_model.visualize_documents(df[text_column])
        
        return barchart_fig, doc_fig
    
    def get_topic_info(self):
        """Get topic information dataframe."""
        if self.topic_model is None:
            raise ValueError("Topic model not fitted. Call analyze_topics first.")
        return self.topic_model.get_topic_info()
