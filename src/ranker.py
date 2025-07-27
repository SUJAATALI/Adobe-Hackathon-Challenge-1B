from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Optional
import logging

class UniversalRankingSystem:
    """Universal ranking system that adapts to any domain"""
    
    def __init__(self, config: Dict = None):
        """Initialize with configurable parameters"""
        self.config = config or self._get_default_config()
        
        # Initialize vectorizer with universal parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_features', 10000),
            ngram_range=tuple(self.config.get('ngram_range', (1, 3))),  # <--- this is key!
            stop_words='english',
            min_df=self.config.get('min_df', 1),
            max_df=self.config.get('max_df', 0.9),
            sublinear_tf=True
        )

    
    def _get_default_config(self) -> Dict:
        """Default configuration for universal use"""
        return {
            'max_features': 10000,
            'ngram_range': (1, 3),
            'min_df': 1,
            'max_df': 0.9,
            'tfidf_weight': 0.35,
            'bm25_weight': 0.35,
            'semantic_weight': 0.30,
            'top_n': 10
        }
    
    def score_sections_tfidf(self, query_vector: str, sections: List[Dict]) -> List[float]:
        """Universal TF-IDF scoring"""
        try:
            corpus = [query_vector] + [section.get("section_vector", "") for section in sections]
            # Filter out empty strings
            corpus = [doc if doc.strip() else "empty document" for doc in corpus]
            
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            query_vec = tfidf_matrix[0]
            section_vecs = tfidf_matrix[1:]
            
            similarities = cosine_similarity(query_vec, section_vecs).flatten()
            return similarities.tolist()
        except Exception as e:
            logging.error(f"TF-IDF scoring failed: {e}")
            return [0.0] * len(sections)
    
    def score_sections_bm25(self, query_vector: str, sections: List[Dict]) -> List[float]:
        """Universal BM25 scoring with improved tokenization"""
        try:
            # Better tokenization for BM25
            corpus = []
            for section in sections:
                text = section.get("section_vector", "")
                tokens = text.split() if text.strip() else ["empty"]
                corpus.append(tokens)
            
            query_tokens = query_vector.split() if query_vector.strip() else ["empty"]
            
            if not corpus or not query_tokens:
                return [0.0] * len(sections)
            
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            
            # Robust normalization
            if len(scores) > 0 and max(scores) > 0:
                max_score = max(scores)
                min_score = min(scores)
                if max_score > min_score:
                    scores = [(score - min_score) / (max_score - min_score) for score in scores]
                else:
                    scores = [1.0 if score > 0 else 0.0 for score in scores]
            else:
                scores = [0.0] * len(sections)
            
            return scores
        except Exception as e:
            logging.error(f"BM25 scoring failed: {e}")
            return [0.0] * len(sections)
    
    def score_semantic_relevance(self, query_vector: str, sections: List[Dict]) -> List[float]:
        """Universal semantic scoring based on section metadata"""
        try:
            query_lower = query_vector.lower()
            scores = []
            
            for section in sections:
                score = 0.0
                
                # Content-based scoring
                section_text = section.get("section_text", "").lower()
                section_title = section.get("section_title", "").lower()
                
                # Title relevance (higher weight)
                title_words = set(section_title.split())
                query_words = set(query_lower.split())
                title_overlap = len(title_words & query_words) / (len(title_words) + 1)
                score += title_overlap * 2.0
                
                # Content relevance
                content_words = set(section_text.split())
                content_overlap = len(content_words & query_words) / (len(content_words) + 1)
                score += content_overlap * 1.0
                
                # Section type relevance
                section_type = section.get("section_type", "content")
                if "method" in query_lower and section_type == "methodology":
                    score += 0.5
                elif "result" in query_lower and section_type == "results":
                    score += 0.5
                elif ("introduction" in query_lower or "overview" in query_lower) and section_type == "introduction":
                    score += 0.5
                
                # Length normalization
                word_count = section.get("word_count", 1)
                if word_count > 50:  # Prefer substantial sections
                    score += 0.1
                
                scores.append(score)
            
            # Normalize scores
            if scores and max(scores) > 0:
                max_score = max(scores)
                scores = [score / max_score for score in scores]
            
            return scores
        except Exception as e:
            logging.error(f"Semantic scoring failed: {e}")
            return [0.0] * len(sections)
    
    def score_and_rank_sections(self, query_vector: str, section_chunks: List[Dict], top_n: Optional[int] = None) -> List[Dict]:
        """Universal ranking with multiple algorithms"""
        if not section_chunks:
            return []
        
        top_n = top_n or self.config['top_n']
        
        # Get scores from different methods
        tfidf_scores = self.score_sections_tfidf(query_vector, section_chunks)
        bm25_scores = self.score_sections_bm25(query_vector, section_chunks)
        semantic_scores = self.score_semantic_relevance(query_vector, section_chunks)
        
        # Combine scores with configurable weights
        combined_scores = []
        weights = self.config
        
        for i in range(len(section_chunks)):
            combined_score = (
                weights['tfidf_weight'] * tfidf_scores[i] +
                weights['bm25_weight'] * bm25_scores[i] +
                weights['semantic_weight'] * semantic_scores[i]
            )
            combined_scores.append(combined_score)
        
        # Assign scores to sections
        for i, section in enumerate(section_chunks):
            section["tfidf_score"] = tfidf_scores[i]
            section["bm25_score"] = bm25_scores[i]
            section["semantic_score"] = semantic_scores[i]
            section["combined_score"] = combined_scores[i]
        
        # Sort by combined score
        section_chunks_sorted = sorted(
            section_chunks, 
            key=lambda s: s.get("combined_score", 0), 
            reverse=True
        )
        
        # Assign importance ranks
        for idx, section in enumerate(section_chunks_sorted[:top_n], start=1):
            section["importance_rank"] = idx
        
        return section_chunks_sorted[:top_n]

# Factory function for backwards compatibility
def score_and_rank_sections(query_vector: str, section_chunks: List[Dict], top_n: int = 10) -> List[Dict]:
    ranker = UniversalRankingSystem()
    return ranker.score_and_rank_sections(query_vector, section_chunks, top_n)
