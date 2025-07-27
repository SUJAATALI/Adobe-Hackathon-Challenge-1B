"""
Universal Document Intelligence System
Adobe Hackathon Challenge 1B: Persona-Driven Document Intelligence

A modular, domain-agnostic system for extracting and ranking relevant document sections
based on user persona and job-to-be-done requirements.
"""

__version__ = "1.0.0"
__author__ = "Adobe Hackathon Team"
__description__ = "Universal Document Intelligence Pipeline"

# Import main classes for easy access
from .main import UniversalDocumentIntelligence
from .pdf_extractor import UniversalPDFExtractor
from .nlp_pipeline import UniversalTextProcessor
from .ranker import UniversalRankingSystem
from .output_writer import UniversalOutputWriter

__all__ = [
    'UniversalDocumentIntelligence',
    'UniversalPDFExtractor', 
    'UniversalTextProcessor',
    'UniversalRankingSystem',
    'UniversalOutputWriter'
]
