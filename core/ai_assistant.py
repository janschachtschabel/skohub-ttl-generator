"""
AI Assistant module for SkoHub TTL Generator.
Provides local AI (sentence transformers) for field mapping and structure recognition using all-MiniLM-L12-v2.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Any
import requests
import json

# Local AI imports
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class AIAssistant:
    """Handles AI-powered suggestions for field mapping and structure recognition using MiniLM."""
    
    def __init__(self):
        self.similarity_model = None
        self.initialize_local_ai()
    
    def initialize_local_ai(self):
        """Initialize sentence transformer model for local AI suggestions"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return
            
        try:
            # Use balanced model for better field recognition quality
            self.similarity_model = SentenceTransformer('all-MiniLM-L12-v2')
        except Exception as e:
            st.error(f"Sentence transformer initialization failed: {e}")
            # Fallback to smaller, faster model
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                st.error("Could not load any sentence transformer model")
    
    def get_suggestions(self, data_sample: Dict, task: str, **kwargs) -> List[str]:
        """Get AI suggestions for various tasks using MiniLM"""
        if task == "field_mapping":
            return self._get_field_mapping_suggestions(data_sample, kwargs.get('similarity_threshold', 0.3))
        elif task == "vocabulary_metadata":
            return self._get_vocabulary_metadata_suggestions(data_sample)
        elif task == "hierarchy_detection":
            return self._get_hierarchy_suggestions(data_sample)
        else:
            return []
    
    def _get_field_mapping_suggestions(self, data_sample: Dict, similarity_threshold: float = 0.3) -> List[str]:
        """Get field mapping suggestions using semantic similarity"""
        if not self.similarity_model:
            return self._get_fallback_field_suggestions(data_sample)
        
        try:
            suggestions = []
            field_names = list(data_sample.keys())
            
            # Enhanced SKOS properties with more patterns for better matching
            skos_properties = {
                'uri': ['unique identifier', 'id', 'uri', 'concept identifier', 'resource identifier', 'node uuid', 'system identifier', 'schlüssel', 'code', 'nummer', 'kennung'],
                'prefLabel': ['main label', 'preferred label', 'title', 'name', 'primary label', 'cclom title', 'content title', 'resource title', 'bezeichnung', 'titel', 'label', 'benennung'],
                'altLabel': ['alternative label', 'synonym', 'alias', 'alternative name', 'other label', 'alternativ', 'synonym', 'alias'],
                'definition': ['description', 'definition', 'scope note', 'explanation', 'comment', 'general description', 'content description', 'beschreibung', 'definition', 'erklärung', 'inhalt'],
                'note': ['note', 'remark', 'comment', 'annotation', 'editorial note', 'notiz', 'bemerkung', 'anmerkung', 'hinweis'],
                'example': ['example', 'sample', 'instance', 'illustration', 'beispiel', 'muster', 'instanz'],
                'scopeNote': ['scope note', 'scope', 'usage note', 'application note'],
                'historyNote': ['history note', 'historical note', 'change history', 'version history'],
                'editorialNote': ['editorial note', 'editor note', 'internal note', 'admin note'],
                'changeNote': ['change note', 'modification note', 'update note', 'revision note'],
                'broader': ['broader concept', 'parent', 'broader term', 'parent concept', 'hierarchy parent', 'level'],
                'narrower': ['narrower concept', 'child', 'narrower term', 'child concept', 'hierarchy child'],
                'related': ['related concept', 'related term', 'associated concept', 'see also'],
                'notation': ['notation', 'code', 'identifier code', 'classification code', 'symbol']
            }
            
            for skos_prop, descriptions in skos_properties.items():
                best_field = None
                best_score = -1
                
                # Create single embedding for all SKOS descriptions combined
                skos_text = f"{skos_prop} " + " ".join(descriptions)
                skos_embedding = self.similarity_model.encode([skos_text])[0]
                
                for field_name in field_names:
                    # Create embedding for field name and sample values
                    field_texts = [field_name]
                    
                    # Add sample values for context (first few non-null values)
                    field_value = data_sample.get(field_name)
                    if field_value is not None:
                        # Handle single values and lists
                        if isinstance(field_value, (list, tuple)):
                            sample_values = [str(v) for v in field_value if v is not None][:2]
                        else:
                            sample_values = [str(field_value)]
                        field_texts.extend(sample_values)
                    
                    # Combine field information into single text
                    field_text = " ".join(field_texts)
                    field_embedding = self.similarity_model.encode([field_text])[0]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(skos_embedding, field_embedding) / (
                        np.linalg.norm(skos_embedding) * np.linalg.norm(field_embedding)
                    )
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_field = field_name
                
                # Only suggest if similarity is above threshold
                if best_field and best_score > similarity_threshold:
                    suggestions.append(f"{skos_prop}: {best_field} (confidence: {best_score:.2f})")
            
            return suggestions
            
        except Exception as e:
            print(f"AI field mapping error: {e}")  # Use print instead of st.warning for debugging
            return self._get_fallback_field_suggestions(data_sample)
    
    def _get_fallback_field_suggestions(self, data_sample: Dict) -> List[str]:
        """Enhanced fallback field suggestions with better pattern matching"""
        suggestions = []
        field_names = list(data_sample.keys())
        
        # Enhanced pattern matching for common field names (German + English + namespace prefixes)
        patterns = {
            'uri': ['id', 'uri', 'identifier', 'uuid', 'key', 'schlüssel', 'code', 'nummer', 'kennung', 'node-uuid', 'sys:node-uuid', 'properties.sys:node-uuid'],
            'prefLabel': ['title', 'name', 'label', 'term', 'bezeichnung', 'titel', 'benennung', 'cclom:title', 'cm:name', 'properties.cclom:title', 'properties.cm:name'],
            'definition': ['description', 'definition', 'desc', 'note', 'beschreibung', 'erklärung', 'inhalt', 'text', 'general_description', 'cclom:general_description', 'properties.cclom:general_description'],
            'altLabel': ['alt', 'alternative', 'synonym', 'alias', 'alternativ', 'andere', 'altlabel'],
            'broader': ['parent', 'broader', 'super', 'category', 'oberbegriff', 'kategorie', 'gruppe', 'level'],
            'narrower': ['child', 'narrower', 'sub', 'subcategory', 'unterbegriff', 'untergruppe'],
            'notation': ['code', 'notation', 'number', 'nr', 'nummer', 'kode', 'zeichen'],
            'example': ['example', 'beispiel', 'sample', 'muster'],
            'scopeNote': ['scope', 'anwendung', 'usage', 'verwendung'],
            'note': ['note', 'notiz', 'bemerkung', 'comment', 'kommentar']
        }
        
        # Check each field against patterns
        matched_fields = set()
        for field_name in field_names:
            field_lower = field_name.lower()
            best_match = None
            best_score = 0
            
            for skos_prop, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in field_lower:
                        # Score based on how well the keyword matches
                        score = len(keyword) / len(field_lower) if len(field_lower) > 0 else 0
                        if field_lower == keyword:  # Exact match gets bonus
                            score += 0.5
                        if score > best_score:
                            best_score = score
                            best_match = skos_prop
            
            if best_match and best_score > 0.3:  # Only suggest if reasonably confident
                suggestions.append(f"✅ '{field_name}' → {best_match} (confidence: {best_score:.1%})")
                matched_fields.add(field_name)
        
        # Add suggestions for unmatched fields
        unmatched = [f for f in field_names if f not in matched_fields]
        if unmatched:
            suggestions.append(f"❓ Unmatched fields: {', '.join(unmatched[:5])}{'...' if len(unmatched) > 5 else ''}")
        
        if not suggestions:
            suggestions.append("⚠️ No automatic suggestions available. Please map fields manually.")
        else:
            suggestions.insert(0, f"🤖 Found {len([s for s in suggestions if '→' in s])} potential field mappings:")
        
        return suggestions
    
    def _get_vocabulary_metadata_suggestions(self, data_sample: Dict) -> List[str]:
        """Get vocabulary metadata suggestions based on data content"""
        suggestions = []
        field_names = list(data_sample.keys())
        
        # Simple heuristics for vocabulary metadata
        if any('skill' in field.lower() for field in field_names):
            suggestions.append("Title: Skills Vocabulary")
            suggestions.append("Description: A vocabulary of skills and competencies")
            suggestions.append("Domain: Education and Training")
        elif any('occupation' in field.lower() or 'job' in field.lower() for field in field_names):
            suggestions.append("Title: Occupations Vocabulary")
            suggestions.append("Description: A vocabulary of occupations and job roles")
            suggestions.append("Domain: Employment and Labor")
        elif any('topic' in field.lower() or 'subject' in field.lower() for field in field_names):
            suggestions.append("Title: Topics Vocabulary")
            suggestions.append("Description: A vocabulary of topics and subjects")
            suggestions.append("Domain: Knowledge Organization")
        else:
            suggestions.append("Title: Custom Vocabulary")
            suggestions.append("Description: A SKOS vocabulary for domain-specific concepts")
            suggestions.append("Domain: General")
        
        return suggestions
    
    def _get_hierarchy_suggestions(self, data_sample: Dict) -> List[str]:
        """Analyze data for potential hierarchical structures"""
        suggestions = []
        field_names = list(data_sample.keys())
        
        # Look for level indicators
        level_fields = [f for f in field_names if 'level' in f.lower() or 'ebene' in f.lower()]
        if level_fields:
            suggestions.append(f"Hierarchical structure detected: {', '.join(level_fields)}")
        
        # Look for parent/child relationships
        parent_fields = [f for f in field_names if any(keyword in f.lower() for keyword in ['parent', 'broader', 'übergeordnet'])]
        if parent_fields:
            suggestions.append(f"Parent relationships detected: {', '.join(parent_fields)}")
        
        # Look for code patterns that suggest hierarchy (e.g., "1", "1.1", "1.1.1")
        for field_name, value in data_sample.items():
            if isinstance(value, str) and '.' in value and value.replace('.', '').isdigit():
                suggestions.append(f"Hierarchical codes detected in field: {field_name}")
                break
        
        return suggestions
    
    def _get_openai_structure_analysis(self, data_sample: Dict, prompt: str = "") -> List[str]:
        """Use OpenAI API to analyze data structure and provide suggestions"""
        if not self.openai_config['api_key']:
            return ["OpenAI API key not configured"]
        
        try:
            # Prepare the data sample for analysis
            sample_text = json.dumps(data_sample, indent=2, ensure_ascii=False)[:2000]  # Limit size
            
            # Default prompt if none provided
            if not prompt:
                prompt = f"""
                Analyze this data sample and provide suggestions for creating a SKOS vocabulary:
                
                {sample_text}
                
                Please provide:
                1. Field mapping suggestions (which fields should map to SKOS properties)
                2. Hierarchical structure analysis (if any levels or parent-child relationships exist)
                3. Vocabulary metadata suggestions (title, description, domain)
                4. Any other insights for SKOS vocabulary creation
                
                Respond in a clear, structured format.
                """
            
            # Make API request
            headers = {
                'Authorization': f'Bearer {self.openai_config["api_key"]}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.openai_config['model'],
                'messages': [
                    {'role': 'system', 'content': 'You are an expert in SKOS vocabularies and knowledge organization systems.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 1000,
                'temperature': 0.3
            }
            
            response = requests.post(
                f"{self.openai_config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # Split into lines for better display
                return content.split('\n')
            else:
                return [f"OpenAI API error: {response.status_code} - {response.text}"]
                
        except Exception as e:
            return [f"OpenAI analysis failed: {str(e)}"]
    
    def is_local_ai_available(self) -> bool:
        """Check if local AI is available"""
        return SENTENCE_TRANSFORMERS_AVAILABLE and self.similarity_model is not None
    
    def is_openai_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return self.openai_config['api_key'] is not None
