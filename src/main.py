import argparse
import json
import logging
from typing import Dict, List, Tuple

from utils import load_json
from pdf_extractor import UniversalPDFExtractor
from nlp_pipeline import UniversalTextProcessor
from ranker import UniversalRankingSystem
from output_writer import UniversalOutputWriter

def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file with error handling"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using defaults.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Config file error: {e}. Using defaults.")
        return {}

class UniversalDocumentIntelligence:
    """Universal Document Intelligence System"""
    
    def __init__(self, config: Dict = None, pdf_dir: str = "data/docs"):
        """Initialize with configurable parameters"""
        self.config = config or self._get_default_config()
        self.pdf_dir = pdf_dir
        # Initialize components with config
        self.pdf_extractor = UniversalPDFExtractor(
            max_workers=self.config.get('max_workers', 4),
            timeout=self.config.get('timeout', 30)
        )
        self.text_processor = UniversalTextProcessor(self.config)
        self.ranker = UniversalRankingSystem(self.config.get('ranking', {}))
        self.output_writer = UniversalOutputWriter(self.config)
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> Dict:
        """Default configuration for universal use"""
        return {
            'max_workers': 4,
            'timeout': 30,
            'min_section_length': 30,
            'max_snippet_length': 250,
            'top_n': 10,
            'ranking': {
                'tfidf_weight': 0.35,
                'bm25_weight': 0.35,
                'semantic_weight': 0.30
            }
        }
    
    def parse_input(self, input_json: Dict) -> Tuple[List[str], Dict, Dict]:
        """Universal input parsing"""
        docs = input_json.get("documents", [])
        persona = input_json.get("persona", {})
        job_to_do = input_json.get("job_to_be_done", {})
        
        # Extract filenames more robustly
        pdf_filenames = []
        for doc in docs:
            if isinstance(doc, dict) and "filename" in doc:
                pdf_filenames.append(doc["filename"])
            elif isinstance(doc, str):
                pdf_filenames.append(doc)
        
        return pdf_filenames, persona, job_to_do
    
    def enhance_query_universal(self, persona: Dict, job_to_do: Dict) -> str:
        """Universal query enhancement that works for any domain"""
        # Extract all text from persona and job
        persona_text = " ".join(str(v) for v in persona.values() if v)
        job_text = " ".join(str(v) for v in job_to_do.values() if v)
        
        # Combine into enhanced query
        base_query = f"{persona_text} {job_text}".strip()
        enhanced_query = base_query
        
        # Add universal synonyms
        universal_expansions = {
            'analyze': 'analyze examination study review',
            'research': 'research investigation study analysis',
            'find': 'find locate identify discover',
            'understand': 'understand comprehend grasp learn',
            'create': 'create develop build generate',
            'plan': 'plan strategy approach method',
            'report': 'report document summary analysis',
            'compare': 'compare contrast evaluate assess',
            'implement': 'implement execute apply deploy',
            'design': 'design architecture structure framework'
        }
        
        for term, expansion in universal_expansions.items():
            if term in base_query.lower():
                enhanced_query += f" {expansion}"
        
        return enhanced_query
    
    def process_documents(self, input_json_path: str, output_path: str = None) -> Dict:
        """Universal document processing pipeline"""
        # Step 1: Parse input
        input_json = load_json(input_json_path)
        pdf_filenames, persona, job_to_do = self.parse_input(input_json)
        
        self.logger.info(f"Processing {len(pdf_filenames)} documents")
        self.logger.info(f"Persona: {persona}")
        self.logger.info(f"Job: {job_to_do}")
        
        # Step 2: Extract PDFs
        docs_text = self.pdf_extractor.extract_documents_text(pdf_filenames, docs_dir=self.pdf_dir)
        self.logger.info(f"Extracted {sum(len(p) for p in docs_text.values())} pages from {len(docs_text)} documents")
        
        # Step 3: Split into sections with config filtering
        section_chunks = self.text_processor.split_documents_into_sections(docs_text)
        self.logger.info(f"Identified {len(section_chunks)} sections")
        
        # Step 4: Process query
        query_text = self.enhance_query_universal(persona, job_to_do)
        query_vector = self.text_processor.preprocess_for_vector(query_text)
        self.logger.info(f"Enhanced query length: {len(query_text)} chars")
        
        # Step 5: Vectorize sections
        for section in section_chunks:
            section["section_vector"] = self.text_processor.preprocess_for_vector(section["section_text"])
        
        # Step 6: Rank sections
        top_n = self.config['top_n']
        ranked_sections = self.ranker.score_and_rank_sections(query_vector, section_chunks, top_n)
        
        # Step 7: Generate output
        output_dict = self.output_writer.prepare_output_dict(
            pdf_filenames, persona, job_to_do, ranked_sections, query_vector
        )
        
        # Step 8: Save output
        if output_path:
            self.output_writer.write_output_json(output_dict, output_path)
            self.logger.info(f"Output saved to {output_path}")
        
        # Log ranking summary
        self.logger.info("Top ranked sections:")
        for section in ranked_sections[:5]:
            self.logger.info(f"{section['importance_rank']}. {section['document']} - "
                           f"{section['section_title'][:50]}... "
                           f"(Score: {section['combined_score']:.3f})")
        
        return output_dict

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(description="Universal Document Intelligence Pipeline")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=False, default="output.json", help="Output JSON file")
    parser.add_argument("--config", required=False, default="config.json", help="Path to config JSON file")
    parser.add_argument("--top-n", type=int, help="Number of top sections to return")
    parser.add_argument("--pdf-dir", required=False, default="data/docs", help="Directory containing PDF files")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.top_n:
        config['top_n'] = args.top_n
    
    # Initialize and run system
    system = UniversalDocumentIntelligence(config, pdf_dir=args.pdf_dir)
    system.process_documents(args.input, args.output)

if __name__ == "__main__":
    main()
