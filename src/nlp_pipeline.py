import re
import json
import spacy
import spacy.util
from typing import List, Tuple, Dict, Optional, Any
import logging
import os

def load_best_spacy_model():
    """Load the best available spaCy model"""
    models_to_try = [
        "en_core_web_lg",    # Large model (best)
        "en_core_web_md",    # Medium model
        "en_core_web_sm",    # Small model
        "en_core_web_trf"    # Transformer model (if available)
    ]
    
    for model_name in models_to_try:
        try:
            if spacy.util.is_package(model_name):
                nlp = spacy.load(model_name)
                print(f"Successfully loaded spaCy model: {model_name}")
                return nlp
        except (OSError, ImportError):
            continue
    
    print("Warning: No spaCy model found. Text processing will use basic methods.")
    print("Install a model with: python -m spacy download en_core_web_sm")
    return None

# Load the best available spaCy model
nlp = load_best_spacy_model()

class UniversalSectionDetector:
    """Universal section detector that works across all document types and domains"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load patterns from external file or use built-in fallback
        self.universal_patterns = self._load_universal_patterns()
        
    def _load_universal_patterns(self) -> Dict:
        """Load universal patterns from external file or use built-in fallback"""
        pattern_file = self.config.get("pattern_file", "src/patterns.json")
        
        try:
            with open(pattern_file, "r", encoding="utf-8") as f:
                raw_patterns = json.load(f)
            
            self.logger.info(f"Loaded patterns from {pattern_file}")
            
            # Convert string patterns to regex objects
            return {
                'title_patterns': [re.compile(p) for p in raw_patterns.get("title_patterns", [])],
                'section_indicators': raw_patterns.get("section_indicators", []),
                'exclusion_patterns': [re.compile(p) for p in raw_patterns.get("exclusion_patterns", [])],
                'strong_indicators': raw_patterns.get("strong_indicators", [])
            }
            
        except Exception as e:
            self.logger.warning(f"Could not load pattern file {pattern_file}: {e}. Using built-in patterns.")
            return self._builtin_patterns()
    
    def _builtin_patterns(self) -> Dict:
        """Built-in fallback patterns"""
        return {
            'title_patterns': [
                re.compile(r"^[A-Z][A-Za-z\s\-&(),:0-9]{8,120}$"),      # General titles
                re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$"),         # Title Case
                re.compile(r"^\d+[\.\)]\s*[A-Z][A-Za-z\s\-&(),:]{5,100}$"), # Numbered sections
                re.compile(r"^[A-Z\s]{3,80}$"),                         # ALL CAPS headers
                re.compile(r"^[A-Z][^.!?]*:$"),                         # Colon-ended headers
            ],
            'section_indicators': [
                'guide', 'introduction', 'overview', 'summary', 'conclusion',
                'method', 'approach', 'process', 'procedure', 'step',
                'result', 'finding', 'outcome', 'analysis', 'discussion',
                'tip', 'advice', 'recommendation', 'suggestion', 'note',
                'example', 'sample', 'case', 'instance', 'illustration',
                'checklist', 'list', 'items', 'points', 'elements',
                'chapter', 'section', 'part', 'unit', 'module',
                'background', 'context', 'setting', 'environment',
                'objective', 'goal', 'purpose', 'aim', 'target',
                'requirement', 'specification', 'criteria', 'standard',
                'activity', 'activities', 'things', 'do', 'attraction',
                'planning', 'preparation', 'organize', 'arrange'
            ],
            'exclusion_patterns': [
                re.compile(r"^[•●\-\*]\s*"),                            # Bullet points
                re.compile(r"^\d+\s*$"),                                # Just numbers
                re.compile(r"^https?://"),                              # URLs
                re.compile(r"^[^A-Za-z]*$"),                           # No letters
                re.compile(r"^.{1,3}$"),                               # Too short
                re.compile(r"^.{200,}$"),                              # Too long
            ],
            'strong_indicators': [
                'introduction', 'overview', 'guide', 'summary', 'conclusion',
                'planning', 'preparation', 'requirements', 'activities'
            ]
        }

    def detect_sections(self, page_text: str, font_info: List[Dict] = None) -> List[Tuple[str, str]]:
        """Universal section detection using multiple strategies with bold text priority"""
        if not page_text or len(page_text.strip()) < 20:
            return []
            
        lines = [line.strip() for line in page_text.split('\n') if line.strip()]
        if not lines:
            return []
        
        sections = []
        
        # Strategy 1: Font-based detection (Primary - uses bold text first)
        if font_info and self.config.get('enable_font_analysis', True):
            font_sections = self._detect_by_font_analysis(lines, font_info)
            if font_sections and len(font_sections) >= 1:
                sections.extend(font_sections)
        
        # Strategy 2: Pattern-based detection (Secondary)
        if not sections:
            pattern_sections = self._detect_by_patterns(lines)
            if pattern_sections:
                sections.extend(pattern_sections)
        
        # Strategy 3: Enhanced NLP-based detection (Tertiary - when spaCy is available)
        if not sections and nlp:
            nlp_sections = self._detect_by_nlp_analysis(lines, page_text)
            if nlp_sections:
                sections.extend(nlp_sections)
        
        # Strategy 4: Statistical detection (Fallback)
        if not sections:
            stat_sections = self._detect_by_statistics(lines, page_text)
            if stat_sections:
                sections.extend(stat_sections)
        
        # Clean and validate all detected sections
        return self._clean_and_validate_sections(sections)

    def _detect_by_font_analysis(self, lines: List[str], font_info: List[Dict]) -> List[Tuple[str, str]]:
        """Detect sections using font size and style analysis - Bold text priority"""
        if not font_info:
            return []
        
        # Calculate font statistics
        font_sizes = [span.get('size', 12) for span in font_info if span.get('text', '').strip()]
        if not font_sizes:
            return []
        
        avg_font_size = sum(font_sizes) / len(font_sizes)
        font_threshold = avg_font_size * self.config.get('font_size_threshold', 1.15)
        
        # Map lines to font information
        line_font_map = self._map_lines_to_fonts(lines, font_info)
        
        sections = []
        current_header = None
        current_content = []
        
        for line in lines:
            line_font_info = line_font_map.get(line, {})
            
            # BOLD TEXT GETS PRIORITY - Check bold first, then font size
            is_header = (
                line_font_info.get('is_bold', False) or                    # ← Bold text first priority
                line_font_info.get('weight') == 'bold' or                  # ← Font weight bold
                line_font_info.get('size', 0) >= font_threshold            # ← Then font size
            ) and self._is_valid_title_universal(line)
            
            if is_header:
                # Save previous section
                if current_header and current_content:
                    content = '\n'.join(current_content).strip()
                    if len(content) >= self.config.get('min_section_length', 30):
                        sections.append((current_header, content))
                
                current_header = line
                current_content = []
            else:
                if line.strip():
                    current_content.append(line)
        
        # Add final section
        if current_header and current_content:
            content = '\n'.join(current_content).strip()
            if len(content) >= self.config.get('min_section_length', 30):
                sections.append((current_header, content))
        
        return sections

    def _detect_by_patterns(self, lines: List[str]) -> List[Tuple[str, str]]:
        """Detect sections using universal text patterns"""
        header_indices = []
        
        for i, line in enumerate(lines):
            if self._matches_title_patterns(line) and self._is_valid_title_universal(line):
                header_indices.append(i)
        
        sections = []
        for i, header_idx in enumerate(header_indices):
            header = lines[header_idx]
            
            # Find content boundaries
            start_idx = header_idx + 1
            end_idx = header_indices[i + 1] if i + 1 < len(header_indices) else len(lines)
            
            content_lines = lines[start_idx:end_idx]
            content = '\n'.join(content_lines).strip()
            
            if len(content) >= self.config.get('min_section_length', 30):
                sections.append((header, content))
        
        return sections

    def _detect_by_nlp_analysis(self, lines: List[str], page_text: str) -> List[Tuple[str, str]]:
        """Use enhanced NLP to detect sections when other methods fail"""
        if not nlp:
            return []
        
        # Try enhanced method first
        enhanced_sections = self._enhanced_nlp_section_detection(lines, page_text)
        if enhanced_sections:
            return enhanced_sections
        
        # Fallback to basic method
        return self._basic_nlp_section_detection(lines, page_text)

    def _enhanced_nlp_section_detection(self, lines: List[str], page_text: str) -> List[Tuple[str, str]]:
        """Enhanced NLP-based section detection using advanced spaCy features"""
        if not nlp:
            return []
        
        doc = nlp(page_text)
        
        potential_headers = []
        
        # Use multiple NLP features for better detection
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            if len(sent_text) < 10 or len(sent_text) > 100:
                continue
            
            # Advanced linguistic analysis
            score = 0
            
            # 1. Part-of-speech patterns typical of headers
            pos_tags = [token.pos_ for token in sent]
            if 'NOUN' in pos_tags or 'PROPN' in pos_tags:
                score += 1
            
            # 2. Named entity recognition - headers often contain entities
            if sent.ents:
                score += 1
            
            # 3. Dependency parsing - headers have simpler structures
            root_deps = [token.dep_ for token in sent if token.dep_ == 'ROOT']
            if len(root_deps) == 1:  # Simple structure with one root
                score += 1
            
            # 4. No ending punctuation (typical of headers)
            if not sent_text.endswith(('.', '!', '?')):
                score += 1
            
            # 5. Contains section indicators from our patterns
            sent_lower = sent_text.lower()
            if any(indicator in sent_lower for indicator in self.universal_patterns['section_indicators']):
                score += 2
            
            # 6. Strong indicators get extra points
            if any(indicator in sent_lower for indicator in self.universal_patterns.get('strong_indicators', [])):
                score += 3
            
            # 7. Noun phrases analysis
            noun_phrases = [chunk.text for chunk in sent.noun_chunks]
            if len(noun_phrases) >= 1 and len(noun_phrases) <= 3:  # Typical header structure
                score += 1
            
            # 8. Token analysis for header-like patterns
            clean_tokens = [token.text for token in sent if not token.is_punct and not token.is_space]
            if 2 <= len(clean_tokens) <= 8:  # Good header length
                score += 1
            
            if score >= 4:  # Threshold for considering as header
                potential_headers.append((score, sent_text))
        
        if potential_headers:
            # Sort by score and return best headers
            potential_headers.sort(key=lambda x: x[0], reverse=True)
            best_header = potential_headers[0][1]
            return [(best_header, page_text)]
        
        return []

    def _basic_nlp_section_detection(self, lines: List[str], page_text: str) -> List[Tuple[str, str]]:
        """Basic NLP section detection (fallback method)"""
        doc = nlp(page_text)
        
        potential_headers = []
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            if (10 <= len(sent_text) <= 100 and
                not sent_text.endswith('.') and
                any(token.pos_ in ['NOUN', 'PROPN', 'ADJ'] for token in sent) and
                len([token for token in sent if token.pos_ == 'VERB']) <= 1):
                
                potential_headers.append(sent_text)
        
        if potential_headers:
            best_header = max(potential_headers, 
                            key=lambda h: len([word for word in h.lower().split() 
                                             if word in self.universal_patterns['section_indicators']]))
            
            return [(best_header, page_text)]
        
        return []

    def _detect_by_statistics(self, lines: List[str], page_text: str) -> List[Tuple[str, str]]:
        """Statistical approach to section detection"""
        if len(lines) < 3:
            return []
        
        # Find lines that are statistically different (length, capitalization, etc.)
        line_stats = []
        for line in lines:
            stats = {
                'line': line,
                'length': len(line),
                'words': len(line.split()),
                'caps_ratio': sum(1 for c in line if c.isupper()) / len(line) if line else 0,
                'alpha_ratio': sum(1 for c in line if c.isalpha()) / len(line) if line else 0
            }
            line_stats.append(stats)
        
        # Find outliers that might be headers
        avg_length = sum(s['length'] for s in line_stats) / len(line_stats)
        
        potential_headers = []
        for stats in line_stats:
            # Criteria for potential header
            if (stats['length'] < avg_length * 0.8 and  # Shorter than average
                stats['words'] >= 2 and                  # At least 2 words
                stats['caps_ratio'] > 0.1 and           # Some capitalization
                stats['alpha_ratio'] > 0.7 and          # Mostly alphabetic
                self._is_valid_title_universal(stats['line'])):
                
                potential_headers.append(stats['line'])
        
        # Create one section with the best header
        if potential_headers:
            best_header = max(potential_headers, key=len)
            return [(best_header, page_text)]
        
        return []

    def _map_lines_to_fonts(self, lines: List[str], font_info: List[Dict]) -> Dict:
        """Map text lines to their font information with enhanced matching"""
        line_font_map = {}
        
        for line in lines:
            line_fonts = []
            
            # Find all font spans that contain text from this line
            for font_span in font_info:
                span_text = font_span.get('text', '').strip()
                if span_text and (span_text in line or line.startswith(span_text[:10])):
                    line_fonts.append(font_span)
            
            if line_fonts:
                # Prioritize bold fonts, then larger fonts
                primary_font = max(line_fonts, key=lambda f: (
                    f.get('is_bold', False) * 10 +  # Bold gets priority
                    f.get('size', 12) +              # Then size
                    len(f.get('text', ''))           # Then text length
                ))
                line_font_map[line] = primary_font
        
        return line_font_map

    def _matches_title_patterns(self, line: str) -> bool:
        """Check if line matches universal title patterns"""
        if not line:
            return False
        
        # Check exclusion patterns first
        for pattern in self.universal_patterns['exclusion_patterns']:
            if pattern.match(line):
                return False
        
        # Check inclusion patterns
        for pattern in self.universal_patterns['title_patterns']:
            if pattern.match(line):
                return True
        
        # Check for section indicators
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in self.universal_patterns['section_indicators'])

    def _is_valid_title_universal(self, title: str) -> bool:
        """Universal title validation that works across all domains"""
        if not title or len(title.strip()) < 5:
            return False
        
        title_clean = title.strip()
        
        # Basic validation
        if not any(c.isalpha() for c in title_clean):
            return False
        
        words = title_clean.split()
        if len(words) < 2:
            return False
        
        # Length constraints
        if len(title_clean) > self.config.get('max_title_length', 150):
            return False
        
        # Check exclusion patterns
        for pattern in self.universal_patterns['exclusion_patterns']:
            if pattern.match(title_clean):
                return False
        
        return True

    def _clean_and_validate_sections(self, sections: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Clean and validate detected sections"""
        cleaned_sections = []
        
        for title, content in sections:
            # Clean title
            clean_title = self._clean_title(title)
            
            # Clean content
            clean_content = self._clean_content(content)
            
            # Validate section
            if (self._is_valid_title_universal(clean_title) and 
                len(clean_content) >= self.config.get('min_section_length', 30)):
                
                cleaned_sections.append((clean_title, clean_content))
        
        return cleaned_sections

    def _clean_title(self, title: str) -> str:
        """Clean and format section title"""
        if not title:
            return ""
        
        title = title.strip()
        
        # Remove common prefixes/suffixes
        title = re.sub(r'^[\d\.\)\]\}\-\*\•\●\s]+', '', title)
        title = re.sub(r'[\:\.\-\s]+$', '', title)
        
        # Normalize whitespace
        title = ' '.join(title.split())
        
        # Limit length
        max_length = self.config.get('max_title_length', 100)
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        return title

    def _clean_content(self, content: str) -> str:
        """Clean section content"""
        if not content:
            return ""
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                # Normalize whitespace
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)


class UniversalTextProcessor:
    """Universal text processor that adapts to any document type and persona"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.section_detector = UniversalSectionDetector(config)
        self.logger = logging.getLogger(__name__)

    def split_documents_into_sections(self, docs_text: Dict) -> List[Dict]:
        """Universal document splitting that works for any document type"""
        section_chunks = []
        min_section_length = self.config.get("min_section_length", 30)
        
        for doc_name, pages in docs_text.items():
            self.logger.info(f"Processing {doc_name} with {len(pages)} pages")
            
            for page in pages:
                page_num = page['page_number']
                page_text = page['text']
                font_info = page.get('font_info', [])
                
                # Universal section detection
                sections = self.section_detector.detect_sections(page_text, font_info)
                
                for section_title, section_text in sections:
                    if len(section_text.strip()) < min_section_length:
                        continue
                    
                    section_chunks.append({
                        "document": self._clean_document_name(doc_name),
                        "page_number": page_num,
                        "section_title": section_title,
                        "section_text": section_text,
                        "word_count": len(section_text.split()),
                        "char_count": len(section_text),
                        "sentence_count": len([s for s in section_text.split('.') if s.strip()]),
                        "section_type": self._classify_section_universal(section_title, section_text),
                        "metadata": {
                            "has_lists": '•' in section_text or '-' in section_text,
                            "has_numbers": any(c.isdigit() for c in section_text),
                            "avg_word_length": sum(len(w) for w in section_text.split()) / len(section_text.split()) if section_text.split() else 0
                        }
                    })
        
        self.logger.info(f"Created {len(section_chunks)} sections total")
        return section_chunks

    def _clean_document_name(self, doc_name: str) -> str:
        
        return doc_name

    def _classify_section_universal(self, title: str, content: str) -> str:
        """Universal section classification that works across all domains"""
        title_lower = (title or "").lower()
        content_lower = (content or "").lower()
        
        # Universal classification based on common patterns
        classification_rules = {
            'introduction': ['introduction', 'intro', 'overview', 'about', 'background'],
            'guide': ['guide', 'how to', 'instruction', 'tutorial', 'step'],
            'list': ['checklist', 'list', 'items', 'things', 'activities'],
            'tips': ['tip', 'advice', 'recommendation', 'suggestion', 'trick'],
            'procedure': ['method', 'procedure', 'process', 'approach', 'technique'],
            'requirements': ['requirement', 'specification', 'criteria', 'standard'],
            'example': ['example', 'sample', 'case', 'instance', 'illustration'],
            'summary': ['summary', 'conclusion', 'recap', 'final', 'closing'],
            'reference': ['reference', 'resource', 'link', 'source', 'bibliography'],
            'description': ['description', 'detail', 'information', 'explanation']
        }
        
        # Check title first, then content
        combined_text = f"{title_lower} {content_lower[:200]}"
        
        for category, keywords in classification_rules.items():
            if any(keyword in combined_text for keyword in keywords):
                return category
        
        return 'content'  # Default category

    def preprocess_for_vector(self, text: str) -> str:
        """Universal text preprocessing for vectorization"""
        if not text:
            return ""
        
        # Use spaCy if available, otherwise fall back to simple processing
        if nlp:
            return self._preprocess_with_spacy(text)
        else:
            return self._preprocess_simple(text)

    def _preprocess_with_spacy(self, text: str) -> str:
        """Advanced preprocessing with spaCy"""
        doc = nlp(text)
        tokens = []
        
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and
                token.is_alpha and 
                len(token.text) > 2):
                tokens.append(token.lemma_.lower())
        
        return ' '.join(tokens)

    def _preprocess_simple(self, text: str) -> str:
        """Simple preprocessing without spaCy"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep spaces
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Split into words and filter
        words = text.split()
        
        # Simple stop words list
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)

    def extract_refined_snippet(self, section_text: str, query_vector: str, 
                              persona: str = "", max_length: int = 400) -> str:
        """Universal snippet extraction that adapts to any persona and query"""
        if not section_text.strip():
            return ""
        
        # Smart sentence splitting
        sentences = self._split_sentences_universal(section_text)
        if not sentences:
            return section_text[:max_length].strip()
        
        # Rank sentences by relevance
        ranked_sentences = self._rank_sentences_universal(sentences, query_vector, persona)
        
        # Build snippet within length limit
        snippet_parts = []
        current_length = 0
        
        for sentence in ranked_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Ensure sentence ends properly
            if not sentence.endswith(('.', '!', '?', ':')):
                sentence += '.'
            
            if current_length + len(sentence) + 1 <= max_length:
                snippet_parts.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
        
        result = ' '.join(snippet_parts)
        
        # Add universal persona context if needed
        result = self._add_universal_persona_context(result, persona)
        
        return result.strip()

    def _split_sentences_universal(self, text: str) -> List[str]:
        """Universal sentence splitting that handles various formats"""
        # Use spaCy for better sentence splitting if available
        if nlp:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        else:
            # Fallback to regex-based splitting
            sentences = re.split(r'[.!?]+\s+', text)
        
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Clean up bullet points and formatting
            sentence = re.sub(r'^[•●\-\*]\s*', '', sentence)
            sentence = ' '.join(sentence.split())  # Normalize whitespace
            
            if len(sentence) > 10:  # Minimum sentence length
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def _rank_sentences_universal(self, sentences: List[str], query_vector: str, persona: str) -> List[str]:
        """Universal sentence ranking based on relevance with spaCy enhancement"""
        if not sentences:
            return sentences
        
        query_words = set(query_vector.lower().split())
        persona_words = set(persona.lower().split()) if persona else set()
        
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            
            # Calculate relevance scores
            query_overlap = len(sentence_words & query_words) / (len(sentence_words) + 1)
            persona_overlap = len(sentence_words & persona_words) / (len(sentence_words) + 1) if persona_words else 0
            
            # Length bonus for substantial sentences
            length_score = min(len(sentence) / 100, 1.0)
            
            # Position bonus (prefer earlier sentences)
            position_score = 1.0 / (sentences.index(sentence) + 1) * 0.1
            
            # spaCy semantic similarity (if available)
            semantic_score = 0
            if nlp:
                try:
                    sent_doc = nlp(sentence)
                    query_doc = nlp(query_vector)
                    if sent_doc.vector.any() and query_doc.vector.any():
                        semantic_score = sent_doc.similarity(query_doc) * 0.5
                except:
                    pass
            
            total_score = (
                query_overlap * 2.0 + 
                persona_overlap * 1.0 + 
                length_score * 0.3 + 
                position_score +
                semantic_score
            )
            
            scored_sentences.append((total_score, sentence))
        
        # Sort by score (descending)
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        return [sentence for score, sentence in scored_sentences]

    def _add_universal_persona_context(self, text: str, persona: str) -> str:
        """Add appropriate context based on persona universally"""
        if not persona or not text:
            return text
        
        persona_lower = persona.lower()
        
        # Universal persona context mapping
        context_mappings = {
            'travel': 'Travel planning note: ',
            'planner': 'Planning tip: ',
            'hr': 'HR consideration: ',
            'professional': 'Professional note: ',
            'contractor': 'For contractors: ',
            'food': 'Culinary note: ',
            'manager': 'Management tip: ',
            'student': 'Study guide: ',
            'teacher': 'Teaching note: ',
            'business': 'Business insight: '
        }
        
        for keyword, prefix in context_mappings.items():
            if keyword in persona_lower and not text.startswith(prefix):
                return f"{prefix}{text}"
        
        return text
