import fitz
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any
import json

class UniversalPDFExtractor:
    """Completely universal PDF extractor that adapts to any document type"""

    def __init__(self, max_workers: int = 4, timeout: int = 60, config: Dict = None):
        self.config = config or {}
        self.config.setdefault("max_workers", max_workers)
        self.config.setdefault("timeout", timeout)
        self.logger = logging.getLogger(__name__)
        
    def extract_pdf_with_metadata(self, pdf_path: str) -> List[Dict]:
        """Universal PDF extraction with comprehensive metadata"""
        try:
            self.logger.info(f"Extracting PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            pages = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Multiple extraction methods for maximum compatibility
                page_data = self._extract_page_universal(page, page_num)
                
                if page_data and len(page_data.get('text', '').strip()) > 10:
                    pages.append(page_data)

            doc.close()
            self.logger.info(f"Successfully extracted {len(pages)} pages from {pdf_path}")
            return pages

        except Exception as e:
            self.logger.error(f"Error extracting PDF {pdf_path}: {e}")
            return []

    def _extract_page_universal(self, page, page_num: int) -> Dict:
        """Universal page extraction that works for any document type"""
        page_data = {
            "page_number": page_num+1, 
            "text": "",
            "font_info": [],
            "text_blocks": [],
            "metadata": {}
        }
        
        try:
            # Method 1: Standard text extraction
            text_standard = page.get_text("text")
            
            # Method 2: Dictionary-based extraction with layout info
            dict_data = page.get_text("dict")
            
            # Method 3: Block-based extraction
            blocks_data = page.get_text("blocks")
            
            # Choose best extraction method
            best_text, font_info = self._select_best_extraction(
                text_standard, dict_data, blocks_data
            )
            
            page_data.update({
                "text": self._clean_extracted_text(best_text),
                "font_info": font_info,
                "text_blocks": self._extract_text_blocks(blocks_data),
                "metadata": self._extract_page_metadata(page, dict_data)
            })
            
        except Exception as e:
            self.logger.warning(f"Error extracting page {page_num}: {e}")
            
        return page_data

    def _select_best_extraction(self, text_standard: str, dict_data: Dict, blocks_data: List) -> tuple:
        """Select the best extraction method based on content quality"""
        font_info = []
        
        # Always try to get font info from dict_data
        if dict_data and self.config.get('enable_font_analysis', True):
            font_info = self._extract_font_info_universal(dict_data)
        
        # Evaluate text quality from different methods
        text_scores = {
            'standard': self._evaluate_text_quality(text_standard),
            'dict': self._evaluate_text_quality(self._extract_text_from_dict(dict_data)),
            'blocks': self._evaluate_text_quality(self._extract_text_from_blocks(blocks_data))
        }
        
        # Choose the method with highest quality score
        best_method = max(text_scores, key=text_scores.get)
        
        if best_method == 'standard':
            return text_standard, font_info
        elif best_method == 'dict':
            return self._extract_text_from_dict(dict_data), font_info
        else:
            return self._extract_text_from_blocks(blocks_data), font_info

    def _extract_font_info_universal(self, dict_data: Dict) -> List[Dict]:
        """Universal font information extraction with enhanced bold detection"""
        font_info = []
        
        try:
            for block in dict_data.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text", "").strip()
                            if span_text:
                                flags = span.get('flags', 0)
                                font_info.append({
                                    'text': span_text,
                                    'size': span.get('size', 12.0),
                                    'flags': flags,
                                    'font': span.get('font', ''),
                                    'bbox': span.get('bbox', []),
                                    'color': span.get('color', 0),
                                    'is_bold': bool(flags & 2**4),  # Bold flag
                                    'is_italic': bool(flags & 2**1),  # Italic flag
                                    'is_superscript': bool(flags & 2**0),  # Superscript flag
                                    'line_bbox': line.get('bbox', []),
                                    'weight': self._calculate_font_weight(flags, span.get('font', ''))
                                })
        except Exception as e:
            self.logger.warning(f"Error extracting font info: {e}")
            
        return font_info

    def _calculate_font_weight(self, flags: int, font_name: str) -> str:
        """Calculate font weight from flags and font name"""
        if flags & 2**4:  # Bold flag
            return 'bold'
        
        # Check font name for weight indicators
        font_lower = font_name.lower()
        if any(weight in font_lower for weight in ['bold', 'black', 'heavy']):
            return 'bold'
        elif any(weight in font_lower for weight in ['light', 'thin']):
            return 'light'
        elif any(weight in font_lower for weight in ['medium', 'semi']):
            return 'medium'
        
        return 'normal'

    def _extract_text_from_dict(self, dict_data: Dict) -> str:
        """Extract text from dictionary format"""
        text_parts = []
        
        try:
            for block in dict_data.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        line_text = ""
                        for span in line.get("spans", []):
                            line_text += span.get("text", "")
                        if line_text.strip():
                            text_parts.append(line_text.strip())
        except Exception:
            pass
            
        return "\n".join(text_parts)

    def _extract_text_from_blocks(self, blocks_data: List) -> str:
        """Extract text from blocks format"""
        text_parts = []
        
        try:
            for block in blocks_data:
                if len(block) >= 5 and block[4].strip():  # Block has text
                    text_parts.append(block[4].strip())
        except Exception:
            pass
            
        return "\n".join(text_parts)

    def _evaluate_text_quality(self, text: str) -> float:
        """Universal text quality evaluation"""
        if not text or len(text.strip()) < 10:
            return 0.0
            
        # Quality metrics
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        punct_chars = sum(1 for c in text if c in '.,!?;:-()[]{}')
        
        # Calculate ratios
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        digit_ratio = digit_chars / total_chars if total_chars > 0 else 0
        space_ratio = space_chars / total_chars if total_chars > 0 else 0
        punct_ratio = punct_chars / total_chars if total_chars > 0 else 0
        
        # Word and sentence analysis
        words = text.split()
        
        # Quality score calculation
        quality_score = (
            alpha_ratio * 0.4 +                    # Good alpha content
            min(digit_ratio * 5, 0.2) +           # Some digits are good
            min(space_ratio * 5, 0.2) +           # Proper spacing
            min(punct_ratio * 10, 0.1) +          # Some punctuation
            min(len(words) / 100, 0.1)            # Reasonable word count
        )
        
        return min(quality_score, 1.0)

    def _extract_text_blocks(self, blocks_data: List) -> List[Dict]:
        """Extract structured text blocks"""
        text_blocks = []
        
        try:
            for i, block in enumerate(blocks_data):
                if len(block) >= 5 and block[4].strip():
                    text_blocks.append({
                        'block_id': i,
                        'bbox': list(block[:4]),
                        'text': block[4].strip(),
                        'block_type': block[5] if len(block) > 5 else 0
                    })
        except Exception:
            pass
            
        return text_blocks

    def _extract_page_metadata(self, page, dict_data: Dict) -> Dict:
        """Extract universal page metadata"""
        metadata = {
            'page_bbox': list(page.rect),
            'rotation': page.rotation,
            'has_images': len(page.get_images()) > 0,
            'has_links': len(page.get_links()) > 0
        }
        
        try:
            # Calculate layout statistics
            if dict_data:
                blocks = dict_data.get("blocks", [])
                text_blocks = [b for b in blocks if b.get("type") == 0]
                
                metadata.update({
                    'total_blocks': len(blocks),
                    'text_blocks': len(text_blocks),
                    'image_blocks': len(blocks) - len(text_blocks)
                })
        except Exception:
            pass
            
        return metadata

    def _clean_extracted_text(self, text: str) -> str:
        """Universal text cleaning"""
        if not text:
            return ""
            
        # Split into lines and process each
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove excessive whitespace
                line = ' '.join(line.split())
                # Keep line if it has reasonable content
                if len(line) > 1 and any(c.isalnum() for c in line):
                    cleaned_lines.append(line)
        
        # Join lines back with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        return cleaned_text

    def extract_documents_text(self, pdf_filenames: List[str], docs_dir: str = "data/docs") -> Dict:
        """Universal document extraction with parallel processing"""
        extracted = {}
        failed_files = []
        
        self.logger.info(f"Starting extraction of {len(pdf_filenames)} documents")
        
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            future_to_filename = {}
            
            for filename in pdf_filenames:
                pdf_path = os.path.join(docs_dir, filename)
                
                if not os.path.isfile(pdf_path):
                    self.logger.error(f"File not found: {pdf_path}")
                    failed_files.append(filename)
                    continue
                
                future = executor.submit(self.extract_pdf_with_metadata, pdf_path)
                future_to_filename[future] = filename
            
            for future in as_completed(future_to_filename, timeout=self.config['timeout'] * len(pdf_filenames)):
                filename = future_to_filename[future]
                
                try:
                    result = future.result(timeout=self.config['timeout'])
                    if result:
                        extracted[filename] = result
                        self.logger.info(f"Successfully processed {filename} ({len(result)} pages)")
                    else:
                        failed_files.append(filename)
                        
                except Exception as e:
                    failed_files.append(filename)
                    self.logger.error(f"Failed to process {filename}: {e}")
        
        if failed_files:
            self.logger.warning(f"Failed to process files: {failed_files}")
        
        return extracted
