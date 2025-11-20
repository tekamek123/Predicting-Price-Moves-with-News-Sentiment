"""
Text Analysis and Topic Modeling for Financial News Headlines
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some functions may not work.")

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

try:
    from gensim import corpora, models
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: Gensim not available. Topic modeling functions may not work.")


def preprocess_text(text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True) -> List[str]:
    """
    Preprocess text for analysis.
    
    Parameters:
    -----------
    text : str
        Input text
    remove_stopwords : bool
        Whether to remove stopwords
    lemmatize : bool
        Whether to lemmatize words
        
    Returns:
    --------
    List[str]
        List of processed tokens
    """
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    if NLTK_AVAILABLE:
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    
    # Remove stopwords
    if remove_stopwords and NLTK_AVAILABLE:
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words]
        except:
            pass
    
    # Lemmatize
    if lemmatize and NLTK_AVAILABLE:
        try:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
        except:
            pass
    
    # Filter out short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    return tokens


def extract_keywords(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Extract common keywords and phrases from headlines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'headline' column
    top_n : int
        Number of top keywords to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with keywords and their frequencies
    """
    if 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'headline' column")
    
    all_tokens = []
    for headline in df['headline'].astype(str):
        tokens = preprocess_text(headline)
        all_tokens.extend(tokens)
    
    # Count word frequencies
    word_freq = Counter(all_tokens)
    
    # Get top N keywords
    top_keywords = word_freq.most_common(top_n)
    
    keywords_df = pd.DataFrame(top_keywords, columns=['keyword', 'frequency'])
    
    return keywords_df


def extract_phrases(df: pd.DataFrame, n_grams: int = 2, top_n: int = 30) -> pd.DataFrame:
    """
    Extract common phrases (n-grams) from headlines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'headline' column
    n_grams : int
        Number of words in the phrase (2 for bigrams, 3 for trigrams, etc.)
    top_n : int
        Number of top phrases to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with phrases and their frequencies
    """
    if 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'headline' column")
    
    all_phrases = []
    
    for headline in df['headline'].astype(str):
        tokens = preprocess_text(headline)
        # Generate n-grams
        for i in range(len(tokens) - n_grams + 1):
            phrase = ' '.join(tokens[i:i+n_grams])
            all_phrases.append(phrase)
    
    # Count phrase frequencies
    phrase_freq = Counter(all_phrases)
    
    # Get top N phrases
    top_phrases = phrase_freq.most_common(top_n)
    
    phrases_df = pd.DataFrame(top_phrases, columns=['phrase', 'frequency'])
    
    return phrases_df


def identify_financial_keywords(df: pd.DataFrame) -> Dict:
    """
    Identify financial-specific keywords and events.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'headline' column
        
    Returns:
    --------
    dict
        Dictionary with financial keyword categories and their frequencies
    """
    financial_keywords = {
        'price_target': ['price target', 'target price', 'price', 'target'],
        'earnings': ['earnings', 'eps', 'revenue', 'profit', 'loss'],
        'fda_approval': ['fda', 'approval', 'approved', 'regulatory'],
        'merger_acquisition': ['merger', 'acquisition', 'm&a', 'takeover', 'buyout'],
        'stock_split': ['split', 'stock split', 'dividend'],
        'analyst_rating': ['upgrade', 'downgrade', 'rating', 'analyst', 'buy', 'sell', 'hold'],
        'guidance': ['guidance', 'forecast', 'outlook', 'expectations'],
        'ipo': ['ipo', 'initial public offering', 'going public'],
    }
    
    results = {}
    
    for category, keywords in financial_keywords.items():
        count = 0
        matching_headlines = []
        
        for headline in df['headline'].astype(str).str.lower():
            if any(keyword in headline for keyword in keywords):
                count += 1
                matching_headlines.append(headline)
        
        results[category] = {
            'count': count,
            'percentage': (count / len(df)) * 100,
            'sample_headlines': matching_headlines[:5]  # First 5 examples
        }
    
    return results


def perform_topic_modeling(df: pd.DataFrame, num_topics: int = 10, 
                          passes: int = 10) -> Tuple[LdaModel, List[List[Tuple]]]:
    """
    Perform topic modeling using LDA (Latent Dirichlet Allocation).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'headline' column
    num_topics : int
        Number of topics to extract
    passes : int
        Number of passes through the corpus
        
    Returns:
    --------
    tuple
        (LDA model, list of topic distributions for each document)
    """
    if not GENSIM_AVAILABLE:
        raise ImportError("Gensim is required for topic modeling")
    
    if 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'headline' column")
    
    # Preprocess all headlines
    processed_docs = [preprocess_text(headline) for headline in df['headline'].astype(str)]
    
    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    
    # Filter extremes
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    
    # Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    # Train LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=passes,
        alpha='auto',
        per_word_topics=True
    )
    
    # Get topic distributions for each document
    doc_topics = [lda_model[doc] for doc in corpus]
    
    return lda_model, doc_topics


def display_topics(lda_model: LdaModel, num_words: int = 10) -> pd.DataFrame:
    """
    Display top words for each topic.
    
    Parameters:
    -----------
    lda_model : LdaModel
        Trained LDA model
    num_words : int
        Number of top words to display per topic
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with topics and their top words
    """
    topics_data = []
    
    for idx, topic in lda_model.print_topics(-1, num_words=num_words):
        words = [word.split('*')[1].strip('"') for word in topic.split('+')]
        weights = [float(word.split('*')[0]) for word in topic.split('+')]
        
        topics_data.append({
            'topic_id': idx,
            'top_words': ', '.join(words),
            'word_weights': weights
        })
    
    return pd.DataFrame(topics_data)


def create_wordcloud(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a word cloud from headlines.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'headline' column
    save_path : str, optional
        Path to save the word cloud image
    """
    if not WORDCLOUD_AVAILABLE:
        raise ImportError("WordCloud is required for this function")
    
    if 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'headline' column")
    
    # Combine all headlines
    text = ' '.join(df['headline'].astype(str))
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                         max_words=100, colormap='viridis').generate(text)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Financial News Headlines', fontsize=16, pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_keywords(keywords_df: pd.DataFrame, top_n: int = 20, 
                 save_path: Optional[str] = None):
    """
    Plot top keywords.
    
    Parameters:
    -----------
    keywords_df : pd.DataFrame
        DataFrame with keywords and frequencies
    top_n : int
        Number of top keywords to display
    save_path : str, optional
        Path to save the plot
    """
    top_keywords = keywords_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_keywords)), top_keywords['frequency'], 
             color='steelblue', edgecolor='black')
    plt.yticks(range(len(top_keywords)), top_keywords['keyword'])
    plt.xlabel('Frequency')
    plt.ylabel('Keyword')
    plt.title(f'Top {top_n} Keywords in Financial News Headlines')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(top_keywords['frequency']):
        plt.text(v + max(top_keywords['frequency']) * 0.01, i, 
                str(v), va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Text Analysis and Topic Modeling Module")
    print("Import this module and use the functions with your data.")

