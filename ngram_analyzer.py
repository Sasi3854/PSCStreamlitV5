import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.util import ngrams
from typing import List, Tuple, Set
import warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class NgramAnalyzer:
    """Handles N-gram analysis and text processing for marine deficiency reports."""
    
    def __init__(self):
        """Initialize the NGram analyzer."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Generic phrases to filter out
        self.generic_phrases = {
            "go to", "properly fill", "fill up", "and the", "the the",
        }
        
        # Marine-specific terms to focus on
        self.marine_terms = {
            "engine", "leak", "leaking", "fire", "damper", "hydraulic", "certificate",
            "crew", "port", "starboard", "stbd", "stern", "vessel", "sewage", "cover",
            "door", "ventilation", "lifeboat", "hatch", "bilge", "pump", "cargo",
            "rust", "hull", "navigation", "gauge", "pressure", "rescue", "boat",
            "oil", "aft", "fwd", "forward"
        }
    
    def get_wordnet_pos(self, tag: str) -> str:
        """Map POS tag to first character used by WordNetLemmatizer."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return 'n'  # Default to noun
    
    def extract_useful_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Return list of n-grams that pass our filters."""
        out = []
        for gram in ngrams(tokens, n):
            phrase = " ".join(gram)
            
            # Skip if in generic phrases
            if phrase in self.generic_phrases:
                continue
                
            # Keep only if any token contains marine terms
            if any(any(marine_term in tok for marine_term in self.marine_terms) for tok in gram):
                out.append(phrase)
        
        return out
    
    def clean_issue_text(self, issue_text: str) -> Tuple[str, List[str], List[str]]:
        """Clean issue text and extract useful n-grams."""
        if pd.isna(issue_text):
            return "", [], []
        
        # Basic cleanup & tokenize
        text = issue_text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'\d+', "", text)
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        
        # POS tagging and lemmatization
        tagged = pos_tag(tokens)
        lemmas = []
        for w, tag in tagged:
            pos = self.get_wordnet_pos(tag)
            lemmas.append(self.lemmatizer.lemmatize(w, pos=pos))
        
        # Extract useful 2-grams and 3-grams
        bigrams = self.extract_useful_ngrams(lemmas, 2)
        trigrams = self.extract_useful_ngrams(lemmas, 3)
        
        # Return cleaned text + n-grams
        cleaned_text = " ".join(lemmas)
        return cleaned_text, bigrams, trigrams
    
    def merge_ngrams(self, bigrams: List[str], trigrams: List[str]) -> List[str]:
        """Merge bigrams and trigrams, removing redundant bigrams."""
        # Split trigrams into sets of words for quick subset testing
        trigram_sets = [set(ng.split()) for ng in trigrams]
        
        cleaned_bigrams = []
        for bg in bigrams:
            bg_set = set(bg.split())
            # If bg_set is NOT a subset of any trigram_set, we keep it
            if not any(bg_set.issubset(tg_set) for tg_set in trigram_sets):
                cleaned_bigrams.append(bg)
        
        # Combine cleaned bigrams + trigrams and remove duplicates
        merged = list(dict.fromkeys(cleaned_bigrams + trigrams))
        return merged
    
    def analyze_ngrams(self, df: pd.DataFrame, text_column: str = 'ISSUE_DETAILS') -> pd.DataFrame:
        """Analyze n-grams in the given dataframe."""
        df_copy = df.copy()
        
        # Apply cleaning and n-gram extraction
        ngram_results = df_copy[text_column].apply(
            lambda t: pd.Series(self.clean_issue_text(t))
        )
        
        df_copy[["CLEANED_TEXT", "USEFUL_BIGRAMS", "USEFUL_TRIGRAMS"]] = ngram_results
        
        # Merge n-grams
        df_copy['MERGED_NGRAMS'] = df_copy.apply(
            lambda row: self.merge_ngrams(
                row['USEFUL_BIGRAMS'] or [], 
                row['USEFUL_TRIGRAMS'] or []
            ),
            axis=1
        )
        
        return df_copy
    
    def get_top_ngrams(self, df: pd.DataFrame, n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get top n-grams by frequency."""
        # Flatten all n-grams
        all_bigrams = []
        all_trigrams = []
        all_merged = []
        
        for _, row in df.iterrows():
            if row['USEFUL_BIGRAMS']:
                all_bigrams.extend(row['USEFUL_BIGRAMS'])
            if row['USEFUL_TRIGRAMS']:
                all_trigrams.extend(row['USEFUL_TRIGRAMS'])
            if row['MERGED_NGRAMS']:
                all_merged.extend(row['MERGED_NGRAMS'])
        
        # Count frequencies
        bigram_counts = pd.Series(all_bigrams).value_counts().head(n)
        trigram_counts = pd.Series(all_trigrams).value_counts().head(n)
        merged_counts = pd.Series(all_merged).value_counts().head(n)
        
        return (
            bigram_counts.to_frame('Count').reset_index(),
            trigram_counts.to_frame('Count').reset_index(),
            merged_counts.to_frame('Count').reset_index()
        )
