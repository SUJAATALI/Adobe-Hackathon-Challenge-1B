from datetime import datetime
import json
from typing import Dict, List
from nlp_pipeline import UniversalTextProcessor

class UniversalOutputWriter:
    """Universal output writer that matches sample format exactly"""
    
    def __init__(self, config: Dict = None):
        """Initialize with configuration"""
        self.config = config or {}
        self.text_processor = UniversalTextProcessor(config)
    
    def prepare_output_dict(self, pdf_filenames: List[str], persona: Dict, 
                          job_to_do: Dict, ranked_sections: List[Dict], 
                          query_vector: str) -> Dict:
        """Prepare output matching sample format exactly"""
        output_sections = []
        subsection_analysis = []
        
        # Get max snippet length from config
        max_snippet_length = self.config.get('max_snippet_length', 300)
        
        for section in ranked_sections:
            # Exact sample format - only these 4 fields in this order
            output_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"],
                "page_number": section["page_number"]
            })
            
            # Extract refined snippet using config length
            refined_text = self.text_processor.extract_refined_snippet(
                section["section_text"], 
                query_vector,
                max_length=max_snippet_length
            )
            
            # Exact sample format - only these 3 fields
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"]
            })
        
        # Match sample metadata structure exactly
        output_dict = {
            "metadata": {
                "input_documents": pdf_filenames,
                "persona": persona.get("role", "") if isinstance(persona, dict) else str(persona),
                "job_to_be_done": job_to_do.get("task", "") if isinstance(job_to_do, dict) else str(job_to_do),
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": output_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output_dict
    
    def write_output_json(self, output_dict: Dict, output_path: str) -> None:
        """Write output to JSON file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)

# Backwards compatibility functions
def prepare_output_dict(pdf_filenames: List[str], persona: Dict, job_to_do: Dict, 
                       ranked_sections: List[Dict], query_vector: str) -> Dict:
    writer = UniversalOutputWriter()
    return writer.prepare_output_dict(pdf_filenames, persona, job_to_do, ranked_sections, query_vector)

def write_output_json(output_dict: Dict, output_path: str) -> None:
    writer = UniversalOutputWriter()
    writer.write_output_json(output_dict, output_path)
