import streamlit as st
import pandas as pd
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import uuid
import re
from pathlib import Path
import tempfile
import zipfile
import logging

# Configure logging for performance monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import TTL Cleaner from local directory
try:
    from ttl_cleaner import TTLCleaner
    TTL_CLEANER_AVAILABLE = True
except ImportError:
    TTL_CLEANER_AVAILABLE = False
    class TTLCleaner:
        def clean_file(self, file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content, {'duplicates_removed': 0, 'syntax_fixes': 0}

# Sentence Transformers integration for local AI
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import enhanced SKOS modules
try:
    from core.mapping_handler import MappingHandler, MappingType
    from core.collections_handler import CollectionsHandler, CollectionType
    from core.documentation_handler import DocumentationHandler, DocumentationType
    from ui.enhanced_skos_config import EnhancedSKOSConfigUI
    ENHANCED_SKOS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Enhanced SKOS features not available: {e}")
    ENHANCED_SKOS_AVAILABLE = False

# Import new multi-file modules
try:
    from core.multi_file_handler import ScenarioType, FileRole, MultiFileHandler
    from core.hierarchy_extractor import HierarchySource, HierarchyExtractor
    from core.data_enrichment import DataEnrichment
    from ui.enhanced_scenario_config import EnhancedScenarioConfigUI
    MULTI_FILE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multi-file features not available: {e}")
    MULTI_FILE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="SkoHub TTL Generator",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SkoHubTTLGenerator:
    def __init__(self):
        self.base_uri_suggestions = [
            "http://w3id.org/openeduhub/vocabs/",
            "http://w3id.org/openeduhub/vocabs/skills/",
            "http://w3id.org/openeduhub/vocabs/occupations/",
            "http://w3id.org/openeduhub/vocabs/competencies/",
            "http://w3id.org/openeduhub/vocabs/topics/",
            "http://example.org/vocabs/",
            "https://vocab.example.org/"
        ]
        
        # Complete SKOS field suggestions per W3C SKOS Reference
        self.field_suggestions = {
            # Core identification
            'uri': ['id', 'uri', 'conceptUri', 'identifier', 'uuid', 'code', 'sys:node-uuid', 'properties.sys:node-uuid'],
            
            # Lexical Labels (Section 5)
            'prefLabel': ['label', 'name', 'title', 'prefLabel', 'preferredLabel', 'term', 'cclom:title', 'properties.cclom:title', 'cm:name', 'properties.cm:name'],
            'altLabel': ['altLabel', 'alternativeLabel', 'synonym', 'aliases', 'alternative'],
            'hiddenLabel': ['hiddenLabel', 'hidden', 'hiddenTerm', 'searchTerm'],
            
            # Notations (Section 6)
            'notation': ['notation', 'code', 'number', 'symbol', 'identifier'],
            
            # Documentation Properties (Section 7) - Notes
            'note': ['note', 'remark', 'comment', 'annotation'],
            'changeNote': ['changeNote', 'change', 'modification', 'update', 'revision'],
            'definition': ['definition', 'description', 'meaning', 'explanation', 'cclom:general_description', 'properties.cclom:general_description'],
            'editorialNote': ['editorialNote', 'editorial', 'internal', 'admin', 'editor'],
            'example': ['example', 'sample', 'instance', 'illustration'],
            'historyNote': ['historyNote', 'history', 'historical', 'background'],
            'scopeNote': ['scopeNote', 'scope', 'usage', 'application', 'context'],
            
            # Semantic Relations (Section 8)
            'broader': ['broader', 'parent', 'parentId', 'broaderConcept', 'super', 'category'],
            'narrower': ['narrower', 'children', 'childId', 'narrowerConcept', 'sub', 'subcategory'],
            'related': ['related', 'relatedConcept', 'seeAlso', 'association'],
            'broaderTransitive': ['broaderTransitive', 'ancestorConcept'],
            'narrowerTransitive': ['narrowerTransitive', 'descendantConcept'],
            'semanticRelation': ['semanticRelation', 'conceptualRelation'],
            
            # Mapping Properties (Section 10)
            'broadMatch': ['broadMatch', 'broaderMatch'],
            'closeMatch': ['closeMatch', 'similarConcept'],
            'exactMatch': ['exactMatch', 'equivalentConcept', 'sameAs'],
            'mappingRelation': ['mappingRelation', 'conceptMapping'],
            'narrowMatch': ['narrowMatch', 'narrowerMatch'],
            'relatedMatch': ['relatedMatch', 'associatedConcept']
        }
        
        self.similarity_model = None
        self.initialize_similarity_model()
        
        # Initialize multi-file handlers
        if MULTI_FILE_AVAILABLE:
            self.multi_file_handler = MultiFileHandler()
            self.hierarchy_extractor = HierarchyExtractor()
            self.data_enrichment = DataEnrichment()
            self.enhanced_scenario_ui = EnhancedScenarioConfigUI()
        else:
            self.multi_file_handler = None
            self.hierarchy_extractor = None
            self.data_enrichment = None
            self.enhanced_scenario_ui = None
    
    def initialize_similarity_model(self):
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
    
    def get_ai_suggestions(self, data_sample: Dict, task: str, similarity_threshold: float = 0.3) -> List[str]:
        """Get similarity-based suggestions for field mapping using local AI"""
        if not self.similarity_model:
            return []
        
        try:
            if task == "field_mapping":
                return self._get_field_mapping_suggestions(data_sample, similarity_threshold)
            elif task == "vocabulary_metadata":
                return self._get_vocabulary_metadata_suggestions(data_sample)
            else:
                return []
        except Exception as e:
            st.warning(f"AI suggestions unavailable: {e}")
            return []
    
    def _get_field_mapping_suggestions(self, data_sample: Dict, similarity_threshold: float = 0.3) -> List[str]:
        """Get field mapping suggestions using semantic similarity"""
        suggestions = []
        field_names = list(data_sample.keys())
        
        # SKOS properties with descriptions for better matching
        skos_properties = {
            'uri': ['unique identifier', 'id', 'uri', 'concept identifier', 'resource identifier', 'node uuid', 'system identifier'],
            'prefLabel': ['main label', 'preferred label', 'title', 'name', 'primary label', 'cclom title', 'content title', 'resource title'],
            'altLabel': ['alternative label', 'synonym', 'alias', 'alternative name', 'other label'],
            'description': ['description', 'definition', 'scope note', 'explanation', 'comment', 'general description', 'content description', 'resource description'],
            'broader': ['broader concept', 'parent', 'broader term', 'parent concept', 'hierarchy parent'],
            'narrower': ['narrower concept', 'child', 'narrower term', 'child concept', 'hierarchy child']
        }
        
        for skos_prop, descriptions in skos_properties.items():
            best_field = None
            best_score = -1
            
            # Create embeddings for SKOS property descriptions
            skos_embeddings = self.similarity_model.encode(descriptions + [skos_prop])
            
            for field_name in field_names:
                # Create embedding for field name and sample values
                field_texts = [field_name]
                
                # Add sample values for context (first few non-null values)
                field_value = data_sample.get(field_name)
                if field_value is not None:
                    # Handle single values and lists
                    if isinstance(field_value, (list, tuple)):
                        sample_values = [str(v) for v in field_value if v is not None][:3]
                    else:
                        sample_values = [str(field_value)]
                    field_texts.extend(sample_values)
                
                field_embedding = self.similarity_model.encode(field_texts)
                
                # Calculate similarity scores
                similarities = []
                for skos_emb in skos_embeddings:
                    for field_emb in field_embedding:
                        similarity = np.dot(skos_emb, field_emb) / (np.linalg.norm(skos_emb) * np.linalg.norm(field_emb))
                        similarities.append(similarity)
                
                avg_similarity = np.mean(similarities)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_field = field_name
            
            # Only suggest if similarity is above threshold
            if best_field and best_score > similarity_threshold:
                suggestions.append(f"{skos_prop}: {best_field} (confidence: {best_score:.2f})")
        
        return suggestions
    
    def _get_vocabulary_metadata_suggestions(self, data_sample: Dict) -> List[str]:
        """Get vocabulary metadata suggestions based on data content"""
        suggestions = []
        
        # Analyze field names and sample values to suggest metadata
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
    
    def load_data_file(self, uploaded_file) -> Tuple[pd.DataFrame, str]:
        """Load data from uploaded file (CSV, JSON, or TTL)"""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'csv':
                # Try different encodings and separators (German CSV files often use semicolon)
                separators = [';', ',', '\t', '|']  # Semicolon first for German files
                encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']  # cp1252 higher priority for German
                
                for encoding in encodings:
                    for sep in separators:
                        try:
                            # Reset file pointer
                            uploaded_file.seek(0)
                            df = pd.read_csv(
                                uploaded_file, 
                                encoding=encoding, 
                                sep=sep,
                                quotechar='"',
                                escapechar='\\',
                                on_bad_lines='skip',
                                engine='python',  # More flexible parser
                                dtype=str  # Read all as strings to avoid type issues
                            )
                            if len(df.columns) > 1 and len(df) > 0:  # Valid DataFrame
                                return df, f"CSV loaded successfully with {encoding} encoding and '{sep}' separator"
                        except (UnicodeDecodeError, pd.errors.ParserError, Exception):
                            continue
                
                # If all fails, try with error_bad_lines=False and warn
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(
                        uploaded_file, 
                        encoding='utf-8', 
                        on_bad_lines='skip',
                        engine='python'
                    )
                    return df, "CSV loaded with some lines skipped due to parsing errors"
                except Exception as e:
                    return None, f"Failed to load CSV: {str(e)}"
            
            elif file_extension == 'json':
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    # Handle list of objects with potential nested properties
                    df = pd.json_normalize(data)
                    
                    # Special handling for nested properties structures
                    if 'properties' in df.columns:
                        # Expand properties column if it contains nested objects
                        properties_expanded = pd.json_normalize(df['properties'].tolist())
                        # Add 'properties.' prefix to avoid column name conflicts
                        properties_expanded.columns = ['properties.' + col for col in properties_expanded.columns]
                        # Combine with original dataframe (excluding original properties column)
                        df_without_props = df.drop('properties', axis=1)
                        df = pd.concat([df_without_props, properties_expanded], axis=1)
                    
                    # Also check for other nested structures and flatten them
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            # Check if column contains nested dictionaries
                            sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                            if isinstance(sample_val, dict):
                                # Flatten nested dictionaries
                                nested_df = pd.json_normalize(df[col].tolist())
                                nested_df.columns = [f'{col}.{subcol}' for subcol in nested_df.columns]
                                df = df.drop(col, axis=1)
                                df = pd.concat([df, nested_df], axis=1)
                else:
                    df = pd.json_normalize(data)
                return df, "JSON loaded successfully with nested structure flattening"
            
            elif file_extension == 'ttl':
                # Basic TTL parsing for concept extraction
                content = uploaded_file.read().decode('utf-8')
                concepts = self.extract_concepts_from_ttl(content)
                df = pd.DataFrame(concepts)
                return df, f"TTL loaded successfully, extracted {len(concepts)} concepts"
            
            else:
                return None, f"Unsupported file format: {file_extension}"
                
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    def extract_concepts_from_ttl(self, ttl_content: str) -> List[Dict]:
        """Extract concepts from TTL content"""
        concepts = []
        
        # Simple regex-based extraction
        concept_pattern = r'<([^>]+)>\s+a\s+skos:Concept\s*;'
        label_pattern = r'skos:prefLabel\s+"([^"]+)"'
        alt_label_pattern = r'skos:altLabel\s+"([^"]+)"'
        
        concept_matches = re.findall(concept_pattern, ttl_content)
        
        for concept_uri in concept_matches:
            concept_block = self.extract_concept_block(ttl_content, concept_uri)
            
            concept = {'uri': concept_uri}
            
            # Extract prefLabel
            pref_labels = re.findall(label_pattern, concept_block)
            if pref_labels:
                concept['prefLabel'] = pref_labels[0]
            
            # Extract altLabels
            alt_labels = re.findall(alt_label_pattern, concept_block)
            if alt_labels:
                concept['altLabel'] = ' | '.join(alt_labels)
            
            concepts.append(concept)
        
        return concepts
    
    def extract_concept_block(self, ttl_content: str, concept_uri: str) -> str:
        """Extract the full concept block from TTL content"""
        lines = ttl_content.split('\n')
        in_concept = False
        concept_lines = []
        
        for line in lines:
            if f'<{concept_uri}>' in line and 'skos:Concept' in line:
                in_concept = True
                concept_lines.append(line)
            elif in_concept:
                concept_lines.append(line)
                if line.strip().endswith('.'):
                    break
        
        return '\n'.join(concept_lines)
    
    def combine_datasets(self, datasets: List[Dict], join_config: Dict) -> pd.DataFrame:
        """Combine multiple datasets into one DataFrame with memory-efficient approach"""
        if len(datasets) == 1:
            return datasets[0]['data']
        
        try:
            # Start with first dataset as base
            combined_df = datasets[0]['data'].copy()
            st.info(f"Starting with {len(combined_df)} records from {datasets[0]['filename']}")
            
            # Join additional datasets
            for i, dataset in enumerate(datasets[1:], 1):
                df = dataset['data']
                st.info(f"Processing dataset {i+1}/{len(datasets)}: {dataset['filename']} ({len(df)} records)")
                
                # Check memory usage before merge
                memory_usage_mb = (combined_df.memory_usage(deep=True).sum() + df.memory_usage(deep=True).sum()) / 1024 / 1024
                if memory_usage_mb > 500:  # If combined data > 500MB, use chunked approach
                    st.warning(f"Large dataset detected ({memory_usage_mb:.1f}MB). Using memory-efficient merge...")
                    combined_df = self._memory_efficient_combine(combined_df, df, join_config, dataset['filename'])
                else:
                    # Standard approach for smaller datasets
                    combined_df = self._standard_combine(combined_df, df, join_config, dataset['filename'])
                
                st.info(f"Combined result: {len(combined_df)} records")
            
            # Remove duplicates if specified
            if join_config.get('remove_duplicates', True):
                initial_count = len(combined_df)
                combined_df = self._remove_duplicates_efficiently(combined_df)
                final_count = len(combined_df)
                if initial_count != final_count:
                    st.info(f"Removed {initial_count - final_count} duplicate records")
            
            return combined_df
            
        except MemoryError as e:
            st.error(f"Memory error during dataset combination: {e}")
            st.error("Try using 'concat' strategy instead of 'merge' for large datasets")
            # Fallback to simple concatenation
            return self._fallback_concat(datasets)
        except Exception as e:
            st.error(f"Error combining datasets: {e}")
            return datasets[0]['data']
    
    def _standard_combine(self, combined_df: pd.DataFrame, df: pd.DataFrame, join_config: Dict, filename: str) -> pd.DataFrame:
        """Standard combination approach for smaller datasets"""
        if join_config['strategy'] == 'concat':
            return pd.concat([combined_df, df], ignore_index=True)
        
        elif join_config['strategy'] == 'merge':
            join_key = join_config.get('join_key', 'id')
            if join_key in combined_df.columns and join_key in df.columns:
                return pd.merge(combined_df, df, on=join_key, how='outer', suffixes=('', f'_{filename}'))
            else:
                st.warning(f"Join key '{join_key}' not found in both datasets. Using concatenation.")
                return pd.concat([combined_df, df], ignore_index=True)
        
        return combined_df
    
    def _memory_efficient_combine(self, combined_df: pd.DataFrame, df: pd.DataFrame, join_config: Dict, filename: str) -> pd.DataFrame:
        """Memory-efficient combination for large datasets"""
        if join_config['strategy'] == 'concat':
            # Use chunked concatenation
            chunk_size = 10000
            result_chunks = [combined_df]
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                result_chunks.append(chunk)
            
            return pd.concat(result_chunks, ignore_index=True)
        
        elif join_config['strategy'] == 'merge':
            join_key = join_config.get('join_key', 'id')
            if join_key not in combined_df.columns or join_key not in df.columns:
                st.warning(f"Join key '{join_key}' not found. Using concatenation.")
                return pd.concat([combined_df, df], ignore_index=True)
            
            # Use chunked merge for large datasets
            chunk_size = 5000
            result_chunks = [combined_df]
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                try:
                    # Only merge unique keys to avoid memory explosion
                    unique_keys = chunk[join_key].unique()
                    existing_keys = combined_df[join_key].unique()
                    new_keys = set(unique_keys) - set(existing_keys)
                    
                    if new_keys:
                        new_chunk = chunk[chunk[join_key].isin(new_keys)]
                        if not new_chunk.empty:
                            result_chunks.append(new_chunk)
                except Exception as e:
                    st.warning(f"Merge failed for chunk {i//chunk_size + 1}, using concatenation: {e}")
                    result_chunks.append(chunk)
            
            return pd.concat(result_chunks, ignore_index=True)
        
        return combined_df
    
    def _remove_duplicates_efficiently(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficiently remove duplicates from large datasets"""
        # Try different duplicate removal strategies based on available columns
        if 'conceptUri' in df.columns:
            return df.drop_duplicates(subset=['conceptUri'], keep='first')
        elif 'uri' in df.columns:
            return df.drop_duplicates(subset=['uri'], keep='first')
        elif 'id' in df.columns:
            return df.drop_duplicates(subset=['id'], keep='first')
        else:
            # For large datasets, use a more memory-efficient approach
            if len(df) > 50000:
                # Remove duplicates in chunks
                chunk_size = 10000
                unique_chunks = []
                seen_hashes = set()
                
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    # Create hash of each row for duplicate detection
                    chunk_hashes = chunk.apply(lambda x: hash(tuple(x.astype(str))), axis=1)
                    
                    # Keep only rows with unseen hashes
                    mask = ~chunk_hashes.isin(seen_hashes)
                    unique_chunk = chunk[mask]
                    
                    if not unique_chunk.empty:
                        unique_chunks.append(unique_chunk)
                        seen_hashes.update(chunk_hashes[mask])
                
                return pd.concat(unique_chunks, ignore_index=True) if unique_chunks else df
            else:
                return df.drop_duplicates()
    
    def _fallback_concat(self, datasets: List[Dict]) -> pd.DataFrame:
        """Fallback concatenation when merge fails"""
        st.warning("Using fallback concatenation due to memory constraints")
        
        result_chunks = []
        for dataset in datasets:
            result_chunks.append(dataset['data'])
        
        return pd.concat(result_chunks, ignore_index=True)
    
    def generate_ttl(self, df: pd.DataFrame, config: Dict) -> str:
        """Generate TTL content from DataFrame and configuration"""
        base_uri = config['base_uri']
        vocab_title = config['vocab_title']
        vocab_description = config['vocab_description']
        field_mapping = config['field_mapping']
        vocab_name = config.get('vocab_name', 'vocabulary')
        
        # Construct full base URI with vocab name
        if not base_uri.endswith('/'):
            base_uri += '/'
        full_base_uri = f"{base_uri}{vocab_name}/"
        
        # Get additional prefixes from config
        additional_prefixes = config.get('additional_prefixes', {})
        
        # Standard prefixes
        standard_prefixes = {
            'skos': 'http://www.w3.org/2004/02/skos/core#',
            'dct': 'http://purl.org/dc/terms/',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }
        
        # Common ESCO-related prefixes
        esco_prefixes = {
            'esco': 'http://data.europa.eu/esco/skill/',
            'isced': 'http://data.europa.eu/esco/isced-f/',
            'isothes': 'http://purl.org/iso25964/skos-thes#'
        }
        
        # Combine all prefixes
        all_prefixes = {**standard_prefixes, **esco_prefixes, **additional_prefixes}
        
        # TTL header with dynamic base URI and prefixes
        ttl_content = f"@base <{full_base_uri}> .\n"
        
        # Add prefix declarations
        for prefix, namespace in all_prefixes.items():
            ttl_content += f"@prefix {prefix}: <{namespace}> .\n"
        
        ttl_content += f"""\n<> a skos:ConceptScheme ;
    dct:title "{vocab_title}"@en ;
    dct:description "{vocab_description}"@en ;
    dct:created "{datetime.now().strftime('%Y-%m-%d')}"^^xsd:date .

"""
        
        # Process each row
        for _, row in df.iterrows():
            concept_uri = self.get_concept_uri(row, field_mapping, full_base_uri)
            
            ttl_content += f"<{concept_uri}> a skos:Concept ;\n"
            
            # Add prefLabel
            if field_mapping.get('prefLabel') and pd.notna(row.get(field_mapping['prefLabel'])):
                pref_label = str(row[field_mapping['prefLabel']]).replace('"', '\\"')
                ttl_content += f'    skos:prefLabel "{pref_label}"@de ;\n'
            
            # Add altLabels
            if field_mapping.get('altLabel') and pd.notna(row.get(field_mapping['altLabel'])):
                alt_labels = str(row[field_mapping['altLabel']])
                if '|' in alt_labels:
                    for alt_label in alt_labels.split('|'):
                        alt_label = alt_label.strip().replace('"', '\\"')
                        if alt_label:
                            ttl_content += f'    skos:altLabel "{alt_label}"@de ;\n'
                else:
                    alt_label = alt_labels.replace('"', '\\"')
                    ttl_content += f'    skos:altLabel "{alt_label}"@de ;\n'
            
            # Add description/scopeNote
            if field_mapping.get('description') and pd.notna(row.get(field_mapping['description'])):
                description = str(row[field_mapping['description']]).replace('"', '\\"')
                ttl_content += f'    skos:scopeNote "{description}"@de ;\n'
            
            # Add broader relationship
            if field_mapping.get('broader') and pd.notna(row.get(field_mapping['broader'])):
                broader_uri = self.resolve_uri(str(row[field_mapping['broader']]), full_base_uri)
                ttl_content += f'    skos:broader <{broader_uri}> ;\n'
            
            # Add narrower relationship
            if field_mapping.get('narrower') and pd.notna(row.get(field_mapping['narrower'])):
                narrower_uris = str(row[field_mapping['narrower']])
                if '|' in narrower_uris:
                    for narrower_uri in narrower_uris.split('|'):
                        narrower_uri = self.resolve_uri(narrower_uri.strip(), full_base_uri)
                        ttl_content += f'    skos:narrower <{narrower_uri}> ;\n'
                else:
                    narrower_uri = self.resolve_uri(narrower_uris, full_base_uri)
                    ttl_content += f'    skos:narrower <{narrower_uri}> ;\n'
            
            # Add to scheme
            ttl_content += '    skos:inScheme <> .\n\n'
        
        return ttl_content
    
    def get_concept_uri(self, row: pd.Series, field_mapping: Dict, base_uri: str) -> str:
        """Generate or extract concept URI"""
        if field_mapping.get('uri') and pd.notna(row.get(field_mapping['uri'])):
            uri_value = str(row[field_mapping['uri']])
            if uri_value.startswith('http'):
                return uri_value
            else:
                return f"{base_uri.rstrip('/')}/{uri_value}"
        else:
            # Generate UUID-based URI
            return f"{base_uri.rstrip('/')}/{str(uuid.uuid4())}"
    
    def resolve_uri(self, uri_value: str, base_uri: str) -> str:
        """Resolve relative URI to absolute URI"""
        if uri_value.startswith('http'):
            return uri_value
        else:
            return f"{base_uri.rstrip('/')}/{uri_value}"
    
    def validate_ttl(self, ttl_content: str, enable_advanced_validation: bool = True) -> Tuple[str, Dict]:
        """Validate TTL content using advanced TTL Cleaner with SKOS validation"""
        try:
            # Write TTL to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False, encoding='utf-8') as f:
                f.write(ttl_content)
                temp_file = f.name
            
            # Initialize TTL Cleaner with advanced options
            cleaner = TTLCleaner(
                chunk_size=1000,
                enable_validation=enable_advanced_validation,
                memory_efficient=True,
                enable_skos_xl=False
            )
            
            # Run advanced cleaning and validation
            cleaned_content, basic_stats = cleaner.clean_file(temp_file)
            
            # Get detailed statistics and validation results
            advanced_stats = {
                **basic_stats,
                'total_concepts': cleaner.stats.get('total_concepts', 0),
                'duplicates_removed': cleaner.stats.get('duplicates_removed', 0),
                'malformed_uris_fixed': cleaner.stats.get('malformed_uris_fixed', 0),
                'encoding_issues_fixed': cleaner.stats.get('encoding_issues_fixed', 0),
                'text_fields_cleaned': cleaner.stats.get('text_fields_cleaned', 0),
                'labels_processed': cleaner.stats.get('labels_processed', 0),
                'final_concepts': cleaner.stats.get('final_concepts', 0),
                'empty_labels_removed': cleaner.stats.get('empty_labels_removed', 0),
                'invalid_concepts_removed': cleaner.stats.get('invalid_concepts_removed', 0),
                'comma_fixes': cleaner.stats.get('comma_fixes', 0),
                'concepts_without_preflabel': cleaner.stats.get('concepts_without_preflabel', 0),
                'validation_violations': cleaner.validation_violations,
                'validation_warnings': cleaner.validation_warnings,
                'errors': cleaner.errors,
                'warnings': cleaner.warnings,
                'change_log': cleaner.change_log
            }
            
            # Clean up
            os.unlink(temp_file)
            
            return cleaned_content, advanced_stats
            
        except Exception as e:
            logger.error(f"TTL validation failed: {str(e)}")
            return ttl_content, {'error': str(e), 'validation_failed': True}

def main():
    st.title("🏷️ SkoHub TTL Generator")
    st.markdown("**Universal tool for generating SKOS-compliant TTL vocabularies from CSV, JSON, or TTL files**")
    
    generator = SkoHubTTLGenerator()
    
    # Sidebar configuration (only AI settings)
    with st.sidebar:
        st.header("⚙️ AI Configuration")
        
        # Local AI Configuration
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            if generator.similarity_model:
                st.success("✅ Local AI model loaded: all-MiniLM-L12-v2")
                st.info("🌍 Supports multilingual field mapping (90+ languages)")
                
                # AI Similarity Threshold
                similarity_threshold = st.slider(
                    "AI Similarity Threshold",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.3,
                    step=0.05,
                    help="Higher values = more strict matching (fewer suggestions)"
                )
                st.caption(f"Current: {similarity_threshold:.2f} - {'Strict' if similarity_threshold > 0.5 else 'Moderate' if similarity_threshold > 0.3 else 'Lenient'}")
            else:
                st.warning("⚠️ Local AI model not loaded")
                similarity_threshold = 0.3  # Default fallback
        else:
            st.error("❌ Install `sentence-transformers` package for AI suggestions")
            st.code("pip install sentence-transformers torch")
            similarity_threshold = 0.3  # Default fallback
    
    # Configuration section in main area
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Base URI Configuration
        st.subheader("🌐 Base URI")
        base_uri = st.text_input(
            "Base URI",
            value="http://w3id.org/openeduhub/vocabs/",
            help="Base URI for the vocabulary namespace"
        )
        
        # Vocabulary name for URI construction
        vocab_name = st.text_input(
            "Vocabulary Name (for URI)",
            value="myVocabulary",
            help="Name used in URI construction (e.g., 'escoSkills' → /escoSkills/)"
        )
        
        # Show full base URI preview
        if not base_uri.endswith('/'):
            preview_base = f"{base_uri}/{vocab_name}/"
        else:
            preview_base = f"{base_uri}{vocab_name}/"
        st.info(f"**Full Base URI:** `{preview_base}`")
    
    with col2:
        # Vocabulary metadata
        st.subheader("📚 Vocabulary Metadata")
        vocab_title = st.text_input(
            "Vocabulary Title",
            value="My Vocabulary",
            help="Title of the vocabulary"
        )
        
        vocab_description = st.text_area(
            "Vocabulary Description",
            value="A SKOS vocabulary generated from uploaded data",
            help="Description of the vocabulary"
        )
    
    # Additional prefixes configuration
    st.subheader("🏷️ Additional Prefixes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show standard prefixes
        with st.expander("📋 Standard Prefixes (included)"):
            st.write("• `skos:` → http://www.w3.org/2004/02/skos/core#")
            st.write("• `dct:` → http://purl.org/dc/terms/")
            st.write("• `xsd:` → http://www.w3.org/2001/XMLSchema#")
            st.write("• `esco:` → http://data.europa.eu/esco/skill/")
            st.write("• `isced:` → http://data.europa.eu/esco/isced-f/")
            st.write("• `isothes:` → http://purl.org/iso25964/skos-thes#")
    
    with col2:
        # Custom prefixes input
        custom_prefixes_text = st.text_area(
            "Custom Prefixes",
            value="",
            help="Format: prefix:namespace (one per line)\nExample:\nfoaf:http://xmlns.com/foaf/0.1/",
            placeholder="foaf:http://xmlns.com/foaf/0.1/\ndc:http://purl.org/dc/elements/1.1/"
        )
        
        # Parse custom prefixes
        additional_prefixes = {}
        if custom_prefixes_text.strip():
            for line in custom_prefixes_text.strip().split('\n'):
                if ':' in line:
                    prefix, namespace = line.split(':', 1)
                    additional_prefixes[prefix.strip()] = namespace.strip()
    
    # Main content area
    if MULTI_FILE_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📁 Single File Upload", 
            "🔄 Multi-File Scenarios", 
            "🔧 Field Mapping", 
            "🎯 Enhanced SKOS", 
            "📝 Manual Entry", 
            "🌳 Hierarchy & Processing",
            "✅ Generate & Validate"
        ])
    else:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Data Upload", "🔧 Field Mapping", "🎯 Enhanced SKOS", "📝 Manual Entry", "✅ Generate & Validate"])
    
    with tab1:
        st.header("📁 Data Upload")
        
        uploaded_files = st.file_uploader(
            "Upload data files (CSV, JSON, TTL)",
            accept_multiple_files=True,
            type=['csv', 'json', 'ttl']
        )
        
        if uploaded_files:
            st.session_state['uploaded_data'] = []
            
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 {uploaded_file.name}"):
                    # FIRST: Save the RAW CSV data BEFORE any processing
                    raw_df = None
                    if uploaded_file.name.endswith('.csv'):
                        try:
                            uploaded_file.seek(0)
                            # Read CSV with minimal processing - PRESERVE ORIGINAL COLUMNS
                            raw_df = pd.read_csv(
                                uploaded_file, 
                                encoding='utf-8', 
                                sep=';',  # German CSV standard
                                dtype=str  # Keep all as strings
                            )
                            st.info(f"🔍 RAW CSV columns: {list(raw_df.columns)}")
                        except:
                            try:
                                uploaded_file.seek(0)
                                raw_df = pd.read_csv(uploaded_file, encoding='cp1252', sep=';', dtype=str)
                            except:
                                pass
                    
                    # SECOND: Process with existing logic
                    df, message = generator.load_data_file(uploaded_file)
                    
                    if df is not None:
                        st.success(message)
                        st.dataframe(df.head())
                        st.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                        st.info(f"🔍 PROCESSED columns: {list(df.columns)}")
                        
                        # Store data with TRUE ORIGINAL preserved
                        st.session_state['uploaded_data'].append({
                            'filename': uploaded_file.name,
                            'data': df,
                            'original_data': raw_df if raw_df is not None else df.copy(),  # TRUE ORIGINAL
                            'columns': list(df.columns)
                        })
                    
                    # AI suggestions for field mapping (automatic)
                    if generator.similarity_model and len(df) > 0:
                        with st.spinner("Getting AI field mapping suggestions..."):
                            sample_data = df.iloc[0].to_dict()
                            suggestions = generator.get_ai_suggestions(sample_data, "field_mapping", similarity_threshold)
                            if suggestions:
                                st.success("🤖 AI Field Mapping Suggestions:")
                                for suggestion in suggestions:
                                    st.write(f"• {suggestion}")
                            else:
                                st.info("💡 No automatic suggestions found. Use manual field mapping in the next tab.")
                                suggestions = generator.get_ai_suggestions(sample_data, "field_mapping", similarity_threshold)
                                if suggestions:
                                    st.success("🤖 AI Field Mapping Suggestions:")
                                    for suggestion in suggestions:
                                        st.write(f"• {suggestion}")
                                else:
                                    st.info("💡 No automatic suggestions found. Use manual field mapping in the next tab.")
                    else:
                        st.error(message)
    
    # New Multi-File Scenarios Tab (only if multi-file features available)
    if MULTI_FILE_AVAILABLE:
        with tab2:
            st.header("🔄 Multi-File Scenarios")
            st.markdown("**Configure advanced multi-file processing scenarios**")
            
            if not generator.enhanced_scenario_ui:
                st.error("❌ Multi-file features not available. Please check module imports.")
            else:
                # Step 1: Scenario Selection
                selected_scenario = generator.enhanced_scenario_ui.render_scenario_selection()
                
                if selected_scenario:
                    st.session_state['selected_scenario'] = selected_scenario
                    
                    # Step 2: File Upload and Role Assignment
                    file_config = generator.enhanced_scenario_ui.render_file_upload_and_roles(selected_scenario)
                    
                    if file_config:
                        st.session_state['multi_file_config'] = file_config
                        
                        # Process uploaded files
                        processed_datasets = []
                        for filename, file in file_config['files'].items():
                            role = file_config['roles'][filename]
                            
                            # Load file data
                            if filename.endswith('.ttl'):
                                content = file.read().decode('utf-8')
                                processed_datasets.append({
                                    'name': filename,
                                    'data': content,
                                    'type': 'ttl',
                                    'role': role
                                })
                            else:
                                df, message = generator.load_data_file(file)
                                if df is not None:
                                    processed_datasets.append({
                                        'name': filename,
                                        'data': df,
                                        'type': 'dataframe',
                                        'role': role
                                    })
                        
                        st.session_state['processed_datasets'] = processed_datasets
                        
                        # Step 3: Scenario-specific Configuration
                        if hasattr(selected_scenario, 'value') and selected_scenario.value == 'enrichment':
                            # Find TTL and enrichment data
                            existing_ttl = None
                            enrichment_data = None
                            
                            for dataset in processed_datasets:
                                if hasattr(dataset.get('role'), 'value') and dataset['role'].value == 'existing_ttl':
                                    existing_ttl = dataset['data']
                                elif hasattr(dataset.get('role'), 'value') and dataset['role'].value == 'enrichment_source':
                                    enrichment_data = dataset['data']
                            
                            if existing_ttl and enrichment_data is not None:
                                enrichment_config = generator.enhanced_scenario_ui.render_data_enrichment_config(
                                    existing_ttl, enrichment_data
                                )
                                st.session_state['enrichment_config'] = enrichment_config
                        
                        elif hasattr(selected_scenario, 'value') and selected_scenario.value == 'combination':
                            combination_config = generator.enhanced_scenario_ui.render_data_combination_config(
                                processed_datasets
                            )
                            st.session_state['combination_config'] = combination_config
                        
                        # Step 4: Processing Preview
                        generator.enhanced_scenario_ui.render_processing_preview({
                            'scenario_type': selected_scenario,
                            'datasets': processed_datasets,
                            **file_config
                        })
                        
                        # Step 5: Processing Controls
                        processing_controls = generator.enhanced_scenario_ui.render_processing_controls()
                        
                        if processing_controls['process_button']:
                            st.success("🚀 Multi-file processing configured! Continue to Field Mapping tab.")
                            st.info("💡 Your multi-file configuration is saved and ready for processing.")
    
    # Adjust tab numbering based on multi-file availability
    field_mapping_tab = tab3 if MULTI_FILE_AVAILABLE else tab2
    
    with field_mapping_tab:
        st.header("🔧 Field Mapping")
        
        if 'uploaded_data' in st.session_state and st.session_state['uploaded_data']:
            dataset_names = [item['filename'] for item in st.session_state['uploaded_data']]
            
            # Multi-dataset selection
            st.subheader("📊 Dataset Selection")
            
            # Initialize selected_data with default
            selected_data = None
            
            if len(dataset_names) > 1:
                # Import new multi-file modules
                from ui.scenario_config import ScenarioConfigUI
                from core.data_combiner import DataCombiner
                from core.multi_file_handler import ScenarioType
                
                scenario_ui = ScenarioConfigUI()
                data_combiner = DataCombiner()
                
                # Multi-file scenario configuration
                st.divider()
                
                # Step 1: Scenario Selection
                scenario_type, scenario_config = scenario_ui.render_scenario_selection(st.session_state['uploaded_data'])
                
                # Step 2: File Role Assignment
                file_roles = scenario_ui.render_file_role_assignment(st.session_state['uploaded_data'], scenario_type)
                
                # Step 3: Join Configuration
                join_config = scenario_ui.render_join_configuration(
                    st.session_state['uploaded_data'], file_roles, scenario_type
                )
                
                # Step 4: Data Combination
                if st.button("🔗 Combine Datasets with Scenario Configuration"):
                    with st.spinner("Combining datasets..."):
                        # Validate join compatibility
                        validation = data_combiner.validate_join_compatibility(
                            st.session_state['uploaded_data'], join_config
                        )
                        
                        if validation['is_valid']:
                            # Show preview first
                            with st.expander("👀 Preview (first 100 rows)"):
                                preview_df = data_combiner.preview_join_result(
                                    st.session_state['uploaded_data'], join_config, 100
                                )
                                st.dataframe(preview_df)
                            
                            # Perform actual combination
                            combined_df = data_combiner.combine_datasets(
                                st.session_state['uploaded_data'], scenario_type, join_config
                            )
                            
                            # Store combined dataset with scenario info
                            st.session_state['combined_dataset'] = {
                                'filename': f"Combined ({len(st.session_state['uploaded_data'])} files)",
                                'data': combined_df,
                                'columns': list(combined_df.columns),
                                'source_files': [d['filename'] for d in st.session_state['uploaded_data']],
                                'scenario_type': scenario_type.value,
                                'file_roles': {k: v.value for k, v in file_roles.items()},
                                'join_config': join_config
                            }
                            
                            # Show combination statistics
                            stats = data_combiner.get_combination_statistics(
                                st.session_state['uploaded_data'], combined_df
                            )
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Input Files", stats['input_files'])
                                st.metric("Input Rows", f"{stats['input_total_rows']:,}")
                            with col2:
                                st.metric("Output Rows", f"{stats['output_rows']:,}")
                                st.metric("Data Change", f"{stats['data_loss_percentage']:.1f}%")
                            with col3:
                                st.metric("Columns Added", stats['columns_added'])
                                st.metric("Scenario", scenario_type.value.title())
                            
                            st.success(f"✅ Successfully combined {len(st.session_state['uploaded_data'])} datasets")
                            
                        else:
                            st.error("❌ Join validation failed:")
                            for error in validation['errors']:
                                st.error(f"• {error}")
                            for warning in validation['warnings']:
                                st.warning(f"• {warning}")
                
                # Legacy simple combination option
                with st.expander("🔧 Simple Combination (Legacy)"):
                    use_simple = st.checkbox("Use simple combination instead", help="Use the old simple combination method")
                    
                    if use_simple:
                        selected_datasets = st.multiselect(
                            "Select datasets to combine",
                            dataset_names,
                            default=dataset_names
                        )
                        
                        if len(selected_datasets) > 1:
                            join_strategy = st.selectbox(
                                "Simple combination strategy",
                                ["concat", "merge"],
                                help="Concat: Stack datasets | Merge: Join on field"
                            )
                            
                            if join_strategy == "merge":
                                common_cols = set(st.session_state['uploaded_data'][0]['columns'])
                                for item in st.session_state['uploaded_data'][1:]:
                                    if item['filename'] in selected_datasets:
                                        common_cols &= set(item['columns'])
                                
                                join_key = st.selectbox(
                                    "Join key",
                                    list(common_cols) if common_cols else ["id"]
                                )
                            else:
                                join_key = None
                            
                            if st.button("🔗 Simple Combine"):
                                selected_data_items = [item for item in st.session_state['uploaded_data'] if item['filename'] in selected_datasets]
                                
                                simple_config = {
                                    'strategy': join_strategy,
                                    'join_key': join_key,
                                    'remove_duplicates': True
                                }
                                
                                combined_df = generator.combine_datasets(selected_data_items, simple_config)
                                
                                st.session_state['combined_dataset'] = {
                                    'filename': f"Simple Combined ({len(selected_datasets)} files)",
                                    'data': combined_df,
                                    'columns': list(combined_df.columns),
                                    'source_files': selected_datasets
                                }
                                
                                st.success(f"✅ Simple combination complete: {len(combined_df)} records")
                
                # Fallback for single dataset selection
                if 'combined_dataset' not in st.session_state:
                    selected_dataset = st.selectbox("Select dataset to map", dataset_names)
                    selected_data = next(item for item in st.session_state['uploaded_data'] if item['filename'] == selected_dataset)
            else:
                # Only one dataset available
                selected_dataset = dataset_names[0]
                selected_data = st.session_state['uploaded_data'][0]
                use_multiple = False
            
            # Use combined dataset if available
            if 'combined_dataset' in st.session_state:
                selected_data = st.session_state['combined_dataset']
                st.info(f"📊 Using combined dataset: {selected_data['filename']}")
                st.write(f"**Source files:** {', '.join(selected_data['source_files'])}")
            
            # Final safety check - ensure selected_data is not None
            if selected_data is None:
                selected_data = st.session_state['uploaded_data'][0]
                st.warning("⚠️ Using first dataset as fallback")
            
            # Collect all available fields from all uploaded sources
            all_columns = []
            source_info = {}
            
            # Add fields from selected dataset
            for col in selected_data['columns']:
                all_columns.append(col)
                source_info[col] = selected_data['filename']
            
            # Add fields from other uploaded datasets (especially TTL)
            for item in st.session_state['uploaded_data']:
                if item['filename'] != selected_data['filename']:
                    for col in item['columns']:
                        if col not in all_columns:
                            all_columns.append(col)
                            source_info[col] = item['filename']
            
            st.subheader(f"🔗 Multi-Source Field Mapping")
            st.info(f"**Primary dataset:** {selected_data['filename']} | **Available fields from all sources:** {len(all_columns)}")
            
            # Show source breakdown
            with st.expander("📋 Field Sources"):
                for source_file in set(source_info.values()):
                    fields_from_source = [field for field, source in source_info.items() if source == source_file]
                    st.write(f"**{source_file}:** {', '.join(fields_from_source[:10])}{'...' if len(fields_from_source) > 10 else ''}")
            
            # Get AI suggestions for all available fields
            ai_suggestions = []
            if generator.similarity_model and len(all_columns) > 0:
                # Create sample data from all sources for AI suggestions
                combined_sample = {}
                for item in st.session_state['uploaded_data']:
                    if len(item['data']) > 0:
                        sample_row = item['data'].iloc[0].to_dict()
                        combined_sample.update(sample_row)
                
                with st.spinner("🤖 Getting AI field mapping suggestions..."):
                    ai_suggestions = generator.get_ai_suggestions(combined_sample, "field_mapping", similarity_threshold)
            
            # Display AI suggestions
            if ai_suggestions:
                with st.expander("🤖 AI Field Mapping Suggestions"):
                    for suggestion in ai_suggestions:
                        st.write(f"• {suggestion}")
            
            # Hierarchy Configuration Section
            st.divider()
            st.subheader("🌳 Hierarchy Configuration")
            
            # Check if we have combined dataset with scenario info
            if ('combined_dataset' in st.session_state and 
                'scenario_type' in st.session_state['combined_dataset']):
                
                # Use the new hierarchy configuration UI
                from ui.scenario_config import ScenarioConfigUI
                scenario_ui = ScenarioConfigUI()
                
                combined_df = st.session_state['combined_dataset']['data']
                
                # Get additional files for hierarchy detection
                additional_files = []
                for item in st.session_state['uploaded_data']:
                    if item['filename'] not in st.session_state['combined_dataset']['source_files']:
                        additional_files.append(item)
                
                # Render hierarchy configuration
                hierarchy_config = scenario_ui.render_hierarchy_configuration(
                    combined_df, additional_files
                )
                
                # Store hierarchy config
                st.session_state['hierarchy_config'] = hierarchy_config
                
                # Show preview and validation
                if st.button("🔍 Preview Hierarchy Processing"):
                    processed_df = scenario_ui.render_preview_and_validation(
                        combined_df, hierarchy_config
                    )
                    
                    # Store processed data
                    st.session_state['hierarchy_processed_data'] = processed_df
                    
            else:
                # Legacy hierarchy configuration for single files or simple combinations
                st.info("💡 For advanced hierarchy features, use the multi-file scenario configuration above.")
                
                # Basic hierarchy options
                hierarchy_enabled = st.checkbox(
                    "Enable hierarchy processing",
                    help="Process hierarchical relationships in your data"
                )
                
                if hierarchy_enabled:
                    from core.hierarchy_handler import HierarchyHandler, HierarchyType
                    hierarchy_handler = HierarchyHandler()
                    
                    current_df = selected_data['data']
                    detected_type = hierarchy_handler.detect_hierarchy_type(current_df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        hierarchy_type = st.selectbox(
                            "Hierarchy type",
                            options=list(HierarchyType),
                            format_func=lambda x: {
                                HierarchyType.PARENT_FIELD: "Parent Field",
                                HierarchyType.LEVEL_FIELD: "Level/Depth Field",
                                HierarchyType.PATH_BASED: "Hierarchical Path",
                                HierarchyType.BROADER_NARROWER: "SKOS Broader/Narrower",
                                HierarchyType.HIERARCHY_FILE: "Separate Hierarchy File"
                            }[x],
                            index=list(HierarchyType).index(detected_type)
                        )
                    
                    with col2:
                        # Field suggestions based on type
                        field_suggestions = hierarchy_handler.get_hierarchy_field_suggestions(
                            current_df, hierarchy_type
                        )
                        
                        if field_suggestions:
                            st.success(f"✅ Found {len(field_suggestions)} potential fields")
                            for field in field_suggestions:
                                st.write(f"• {field}")
                        else:
                            st.warning("⚠️ No obvious hierarchy fields detected")
                    
                    # Configure hierarchy fields based on type
                    hierarchy_config = {'type': hierarchy_type.value}
                    
                    if hierarchy_type == HierarchyType.PARENT_FIELD:
                        parent_field = st.selectbox(
                            "Parent field",
                            options=current_df.columns.tolist(),
                            index=0 if field_suggestions and field_suggestions[0] in current_df.columns 
                                  else 0
                        )
                        hierarchy_config['parent_field'] = parent_field
                    
                    elif hierarchy_type == HierarchyType.LEVEL_FIELD:
                        # Get all available fields from all uploaded data
                        all_available_fields = []
                        for data in st.session_state['uploaded_data']:
                            all_available_fields.extend(data['columns'])
                        all_available_fields = list(dict.fromkeys(all_available_fields))  # Remove duplicates
                        
                        # Suggest level fields
                        level_candidates = [f for f in all_available_fields if any(keyword in f.lower() for keyword in ['level', 'ebene', 'depth', 'tier'])]
                        
                        if level_candidates:
                            st.write(f"💡 **Suggested level fields**: {', '.join(level_candidates)}")
                        
                        level_field = st.selectbox(
                            "Level field",
                            options=all_available_fields,
                            index=all_available_fields.index(level_candidates[0]) if level_candidates else 0,
                            help="Field containing hierarchy level/depth (e.g., Ebene with values 1,2,3,4,5)"
                        )
                        hierarchy_config['level_field'] = level_field
                        
                        # Show sample values if available
                        for data in st.session_state['uploaded_data']:
                            if level_field in data['columns']:
                                sample_values = data['data'][level_field].unique()[:5]
                                st.write(f"📊 **Sample values from {data['filename']}**: {list(sample_values)}")
                                break
                    
                    # ID field selection - also use all available fields
                    all_available_fields = []
                    for data in st.session_state['uploaded_data']:
                        all_available_fields.extend(data['columns'])
                    all_available_fields = list(dict.fromkeys(all_available_fields))
                    
                    id_field_options = [col for col in all_available_fields 
                                       if col.lower() in ['id', 'uri', 'concepturi', 'identifier'] or 'schlüssel' in col.lower()]
                    
                    if id_field_options:
                        st.write(f"💡 **Suggested ID fields**: {', '.join(id_field_options)}")
                        id_field = st.selectbox("ID field", options=id_field_options)
                    else:
                        id_field = st.selectbox("ID field", options=all_available_fields)
                    
                    hierarchy_config['id_field'] = id_field
                    
                    # Store config
                    st.session_state['hierarchy_config'] = hierarchy_config
                    
                    # Preview hierarchy processing
                    if st.button("🔍 Preview Hierarchy"):
                        processed_df = hierarchy_handler.process_hierarchy(current_df, hierarchy_config)
                        
                        # Show preview
                        st.write("**Processed Data Preview:**")
                        st.dataframe(processed_df.head())
                        
                        # Show validation
                        issues = hierarchy_handler.validate_hierarchy(processed_df)
                        total_issues = sum(len(issue_list) for issue_list in issues.values())
                        
                        if total_issues == 0:
                            st.success("✅ No hierarchy issues found")
                        else:
                            st.warning(f"⚠️ {total_issues} hierarchy issues found")
                            for issue_type, issue_list in issues.items():
                                if issue_list:
                                    st.write(f"• {issue_type.replace('_', ' ').title()}: {len(issue_list)}")
                        
                        # Store processed data
                        st.session_state['hierarchy_processed_data'] = processed_df
            
            # Join Key Selection for Multi-Source Data Merging
            if len(st.session_state['uploaded_data']) > 1:
                st.subheader("🔗 Data Merging Configuration")
                
                col_merge1, col_merge2 = st.columns(2)
                
                with col_merge1:
                    st.write("**Join Strategy**")
                    merge_strategy = st.selectbox(
                        "How to combine data from multiple sources?",
                        ["use_primary_only", "merge_on_key", "concat_all"],
                        format_func=lambda x: {
                            "use_primary_only": "Use primary dataset only (no merging)",
                            "merge_on_key": "Merge datasets using matching key field",
                            "concat_all": "Concatenate all datasets (stack rows)"
                        }[x],
                        help="Choose how to handle multiple data sources"
                    )
                
                with col_merge2:
                    if merge_strategy == "merge_on_key":
                        st.write("**Join Key Mapping**")
                        st.caption("💡 Map fields with matching content from different sources")
                        
                        # Allow flexible join key mapping between different datasets
                        join_key_mapping = {}
                        
                        # Show join key selection for each dataset
                        for i, dataset in enumerate(st.session_state['uploaded_data']):
                            st.write(f"**{dataset['filename']}:**")
                            
                            join_field = st.selectbox(
                                f"Join key field for {dataset['filename']}",
                                ["None"] + dataset['columns'],
                                key=f"join_key_{i}",
                                help=f"Select the field that contains matching values in {dataset['filename']}"
                            )
                            
                            if join_field != "None":
                                join_key_mapping[dataset['filename']] = join_field
                                
                                # Show sample values
                                if len(dataset['data']) > 0:
                                    sample_values = dataset['data'][join_field].head(3).tolist()
                                    st.caption(f"Sample values: {', '.join(str(v) for v in sample_values)}")
                        
                        if len(join_key_mapping) >= 2:
                            st.success(f"🔗 Will merge datasets using mapped join keys: {', '.join(f'{k}: {v}' for k, v in join_key_mapping.items())}")
                            
                            # Store merge configuration
                            st.session_state['merge_config'] = {
                                'strategy': merge_strategy,
                                'join_key_mapping': join_key_mapping
                            }
                        else:
                            st.warning("⚠️ Please select join key fields for at least 2 datasets.")
                            st.session_state['merge_config'] = {'strategy': 'use_primary_only'}
                    else:
                        st.session_state['merge_config'] = {'strategy': merge_strategy}
                        if merge_strategy == "use_primary_only":
                            st.info(f"📄 Using only primary dataset: {selected_data['filename']}")
                        elif merge_strategy == "concat_all":
                            st.info("📋 Will stack all datasets vertically")
            
            # Field mapping interface
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**SKOS Properties**")
                field_mapping = {}
                
                for skos_prop, suggestions in generator.field_suggestions.items():
                    # Find best suggestion from all available columns
                    suggested_field = None
                    for suggestion in suggestions:
                        for col in all_columns:
                            if suggestion.lower() in col.lower():
                                suggested_field = col
                                break
                        if suggested_field:
                            break
                    
                    # Create selectbox with all available fields
                    options = ["None"] + all_columns
                    default_index = options.index(suggested_field) if suggested_field else 0
                    
                    selected_field = st.selectbox(
                        f"{skos_prop}",
                        options,
                        index=default_index,
                        help=f"Suggested fields: {', '.join(suggestions)}",
                        format_func=lambda x: f"{x} ({source_info.get(x, 'unknown')})" if x != "None" else x
                    )
                    
                    if selected_field != "None":
                        field_mapping[skos_prop] = selected_field
            
            with col2:
                st.write("**Data Preview**")
                if field_mapping:
                    # Create preview from multiple sources
                    preview_data = {}
                    for skos_prop, field_name in field_mapping.items():
                        # Find which dataset contains this field
                        source_dataset = None
                        for item in st.session_state['uploaded_data']:
                            if field_name in item['columns']:
                                source_dataset = item
                                break
                        
                        if source_dataset and len(source_dataset['data']) > 0:
                            preview_data[f"{skos_prop} ({field_name})"] = source_dataset['data'][field_name].head().tolist()
                    
                    if preview_data:
                        # Convert to DataFrame for display
                        max_len = max(len(v) for v in preview_data.values())
                        for key, values in preview_data.items():
                            while len(values) < max_len:
                                values.append("")
                        
                        preview_df = pd.DataFrame(preview_data)
                        st.dataframe(preview_df, use_container_width=True)
                        
                        # Show source information
                        st.caption("**Sources:** " + ", ".join(set(source_info[field] for field in field_mapping.values())))
                    else:
                        st.warning("No preview data available")
                else:
                    st.info("Select fields to see preview")
            
            # Store field mapping
            if field_mapping:
                st.session_state['field_mapping'] = field_mapping
                st.session_state['selected_dataset'] = selected_data
                st.success(f"Field mapping configured for {len(field_mapping)} properties")
        else:
            st.info("Please upload data files first")
    
    with tab3:
        st.header("🎯 Enhanced SKOS Features")
        
        if ENHANCED_SKOS_AVAILABLE:
            # Initialize enhanced SKOS UI
            enhanced_ui = EnhancedSKOSConfigUI()
            
            if 'uploaded_data' in st.session_state and st.session_state['uploaded_data']:
                st.markdown("Configure advanced W3C SKOS features for your vocabulary:")
                
                # Performance optimized configuration sections
                with st.container():
                    # Mapping Properties Configuration
                    mapping_config = enhanced_ui.render_mapping_configuration(
                        st.session_state['uploaded_data']
                    )
                    
                    st.divider()
                    
                    # Collections Configuration
                    collections_config = enhanced_ui.render_collections_configuration(
                        st.session_state['uploaded_data']
                    )
                    
                    st.divider()
                    
                    # Enhanced Documentation Configuration
                    documentation_config = enhanced_ui.render_documentation_configuration(
                        st.session_state['uploaded_data']
                    )
                    
                    st.divider()
                    
                    # Summary of Enhanced Features
                    enhanced_config = enhanced_ui.render_enhanced_skos_summary(
                        mapping_config, collections_config, documentation_config
                    )
                    
                    # Store enhanced configuration in session state
                    st.session_state['enhanced_skos_config'] = enhanced_config
                    
                    # Performance tips
                    with st.expander("⚡ Performance Tips", expanded=False):
                        st.markdown("""
                        **For optimal performance:**
                        - ✅ Field detection is cached automatically
                        - ✅ Large datasets use sampling for validation
                        - ✅ Configuration changes update instantly
                        - 💡 Use the preview functions to test settings before full processing
                        """)
            else:
                st.info("📁 Upload data files first to configure enhanced SKOS features")
                
                # Show feature overview
                st.markdown("""
                ### 🚀 Available Enhanced Features
                
                **🔗 Mapping Properties** (W3C SKOS Section 10)
                - Cross-vocabulary concept mappings
                - exactMatch, closeMatch, broadMatch, narrowMatch, relatedMatch
                - Automatic conflict detection and validation
                
                **📚 Collections** (W3C SKOS Section 9)
                - Grouped concept collections
                - Ordered collections with member lists
                - Automatic collection detection from data
                
                **📝 Enhanced Documentation** (W3C SKOS Section 7)
                - Extended documentation properties
                - Multi-language support
                - Quality analysis and suggestions
                """)
        else:
            st.error("❌ Enhanced SKOS features not available")
            st.info("Please ensure all required modules are installed and accessible.")
            
            # Fallback information
            st.markdown("""
            ### Missing Enhanced Features
            
            The following W3C SKOS features require additional modules:
            - Mapping Properties Handler
            - Collections Handler  
            - Documentation Handler
            - Enhanced SKOS Configuration UI
            
            Please check the installation and module imports.
            """)
    
    # Adjust tab references based on multi-file availability
    enhanced_skos_tab = tab4 if MULTI_FILE_AVAILABLE else tab3
    manual_entry_tab = tab5 if MULTI_FILE_AVAILABLE else tab4
    hierarchy_processing_tab = tab6 if MULTI_FILE_AVAILABLE else None
    generate_validate_tab = tab7 if MULTI_FILE_AVAILABLE else tab5
    
    with enhanced_skos_tab:
        st.header("🎯 Enhanced SKOS Configuration")
        
        # Enhanced SKOS content (existing code moved here)
        if ENHANCED_SKOS_AVAILABLE and 'selected_dataset' in st.session_state:
            st.info("Enhanced SKOS features available - configure advanced properties")
        else:
            st.info("📁 Upload data files first or enhanced SKOS features not available")
    
    with manual_entry_tab:
        st.header("📝 Manual Vocabulary Entry")
        st.info("For small vocabularies, you can create concepts manually")
        
        # Initialize manual concepts
        if 'manual_concepts' not in st.session_state:
            st.session_state['manual_concepts'] = []
        
        # Add new concept form
        with st.form("add_concept"):
            st.subheader("Add New Concept")
            
            col1, col2 = st.columns(2)
            with col1:
                concept_uri = st.text_input("URI (optional)", placeholder="Leave empty for auto-generation")
                pref_label = st.text_input("Preferred Label*", placeholder="Main term")
            
            with col2:
                alt_labels = st.text_input("Alternative Labels", placeholder="Synonym 1 | Synonym 2")
                description = st.text_area("Description", placeholder="Definition or scope note")
            
            submitted = st.form_submit_button("Add Concept")
            
            if submitted and pref_label:
                concept = {
                    'uri': concept_uri or str(uuid.uuid4()),
                    'prefLabel': pref_label,
                    'altLabel': alt_labels,
                    'description': description
                }
                st.session_state['manual_concepts'].append(concept)
                st.success(f"Added concept: {pref_label}")
        
        # Display existing concepts
        if st.session_state['manual_concepts']:
            st.subheader("Current Concepts")
            
            for i, concept in enumerate(st.session_state['manual_concepts']):
                with st.expander(f"📄 {concept['prefLabel']}"):
                    st.write(f"**URI:** {concept['uri']}")
                    st.write(f"**Preferred Label:** {concept['prefLabel']}")
                    if concept['altLabel']:
                        st.write(f"**Alternative Labels:** {concept['altLabel']}")
                    if concept['description']:
                        st.write(f"**Description:** {concept['description']}")
                    
                    if st.button(f"Remove", key=f"remove_{i}"):
                        st.session_state['manual_concepts'].pop(i)
                        st.rerun()
            
            # Convert to DataFrame for processing
            if st.button("Use Manual Concepts"):
                manual_df = pd.DataFrame(st.session_state['manual_concepts'])
                st.session_state['selected_dataset'] = {
                    'filename': 'Manual Entry',
                    'data': manual_df,
                    'columns': list(manual_df.columns)
                }
                st.session_state['field_mapping'] = {
                    'uri': 'uri',
                    'prefLabel': 'prefLabel',
                    'altLabel': 'altLabel',
                    'description': 'description'
                }
                st.success("Manual concepts ready for TTL generation")
    
    # New Hierarchy & Processing Tab (only if multi-file features available)
    if MULTI_FILE_AVAILABLE and hierarchy_processing_tab:
        with hierarchy_processing_tab:
            st.header("🌳 Hierarchy & Processing Configuration")
            st.markdown("**Configure hierarchy extraction and advanced processing options**")
            
            # Check if we have data to work with
            datasets_available = []
            
            # From single file upload
            if 'uploaded_data' in st.session_state and st.session_state['uploaded_data']:
                for item in st.session_state['uploaded_data']:
                    datasets_available.append({
                        'name': item['filename'],
                        'data': item['data'],
                        'type': 'dataframe' if isinstance(item['data'], pd.DataFrame) else 'unknown'
                    })
            
            # From multi-file scenarios
            if 'processed_datasets' in st.session_state:
                datasets_available.extend(st.session_state['processed_datasets'])
            
            if not datasets_available:
                st.warning("⚠️ No datasets available. Please upload data in the previous tabs first.")
            else:
                # Step 1: Hierarchy Source Selection
                st.subheader("🌳 Hierarchy Extraction Configuration")
                
                available_sources = generator.hierarchy_extractor.get_available_sources(datasets_available)
                
                if available_sources:
                    # Source selection UI
                    source_names = {
                        HierarchySource.AUTO_DETECT: "🤖 Auto-Detect (Recommended)",
                        HierarchySource.TTL_BROADER_NARROWER: "📄 TTL SKOS Hierarchy",
                        HierarchySource.CSV_LEVEL_FIELD: "📈 CSV Level/Depth Field",
                        HierarchySource.CSV_PARENT_FIELD: "👥 CSV Parent ID Field",
                        HierarchySource.CSV_PATH_FIELD: "🗺️ CSV Hierarchical Path",
                        HierarchySource.HIERARCHY_FILE: "📋 Separate Hierarchy File"
                    }
                    
                    available_names = [source_names.get(source, source.value) for source in available_sources]
                    
                    selected_name = st.selectbox(
                        "Choose hierarchy extraction source:",
                        options=available_names,
                        help="Select how to extract hierarchy information from your data"
                    )
                    
                    # Find selected source
                    selected_source = None
                    for source in available_sources:
                        if source_names.get(source, source.value) == selected_name:
                            selected_source = source
                            break
                    
                    if selected_source:
                        st.session_state['hierarchy_source'] = selected_source
                        
                        # Step 2: Source-specific Configuration
                        if selected_source != HierarchySource.AUTO_DETECT:
                            st.write(f"**Configuration for {selected_name}:**")
                            hierarchy_config = generator.hierarchy_extractor.get_hierarchy_config_ui(
                                selected_source, datasets_available
                            )
                            st.session_state['hierarchy_config'] = hierarchy_config
                        else:
                            st.session_state['hierarchy_config'] = {}
                        
                        # Step 3: Extract and Preview Hierarchy
                        if st.button("🔍 Extract Hierarchy Preview", type="secondary"):
                            with st.spinner("Extracting hierarchy..."):
                                try:
                                    hierarchy_result = generator.hierarchy_extractor.extract_hierarchy(
                                        selected_source, 
                                        datasets_available, 
                                        st.session_state.get('hierarchy_config', {})
                                    )
                                    
                                    if 'error' in hierarchy_result:
                                        st.error(f"❌ Hierarchy extraction failed: {hierarchy_result['error']}")
                                    else:
                                        st.success(f"✅ Hierarchy extracted successfully from: {hierarchy_result.get('source', 'Unknown')}")
                                        
                                        # Store hierarchy result
                                        st.session_state['extracted_hierarchy'] = hierarchy_result
                                        
                                        # Display hierarchy statistics
                                        col1, col2, col3 = st.columns(3)
                                        
                                        with col1:
                                            st.metric("Total Concepts", len(hierarchy_result.get('concepts', set())))
                                        
                                        with col2:
                                            st.metric("Broader Relations", len(hierarchy_result.get('broader_relations', {})))
                                        
                                        with col3:
                                            st.metric("Narrower Relations", len(hierarchy_result.get('narrower_relations', {})))
                                        
                                        # Show sample hierarchy
                                        if hierarchy_result.get('broader_relations'):
                                            st.write("**Sample Hierarchy Relations:**")
                                            sample_relations = list(hierarchy_result['broader_relations'].items())[:5]
                                            for child, parent in sample_relations:
                                                st.write(f"• {child} → broader: {parent}")
                                            
                                            if len(hierarchy_result['broader_relations']) > 5:
                                                st.caption(f"... and {len(hierarchy_result['broader_relations']) - 5} more relations")
                                        
                                        # Auto-detected info
                                        if hierarchy_result.get('auto_detected'):
                                            st.info(f"🤖 Auto-detected hierarchy source: {hierarchy_result['source']}")
                                
                                except Exception as e:
                                    st.error(f"❌ Error during hierarchy extraction: {str(e)}")
                        
                        # Step 4: Processing Options
                        st.divider()
                        st.subheader("⚙️ Processing Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            include_transitive = st.checkbox(
                                "Include Transitive Relations",
                                value=True,
                                help="Add skos:broaderTransitive and skos:narrowerTransitive properties"
                            )
                            
                            validate_hierarchy = st.checkbox(
                                "Validate Hierarchy",
                                value=True,
                                help="Check for circular references and orphaned concepts"
                            )
                        
                        with col2:
                            generate_top_concepts = st.checkbox(
                                "Generate Top Concepts",
                                value=True,
                                help="Identify and mark top-level concepts with skos:topConceptOf"
                            )
                            
                            include_statistics = st.checkbox(
                                "Include Processing Statistics",
                                value=True,
                                help="Generate detailed statistics about the hierarchy"
                            )
                        
                        # Store processing options
                        st.session_state['processing_options'] = {
                            'include_transitive': include_transitive,
                            'validate_hierarchy': validate_hierarchy,
                            'generate_top_concepts': generate_top_concepts,
                            'include_statistics': include_statistics
                        }
                        
                        # Ready indicator
                        if 'extracted_hierarchy' in st.session_state:
                            st.success("🎉 Hierarchy configuration complete! Ready for TTL generation.")
                            st.info("💡 Continue to the Generate & Validate tab to create your TTL file.")
                
                else:
                    st.warning("⚠️ No suitable hierarchy sources detected in your data.")
                    st.info("💡 Try uploading data with level fields, parent relationships, or existing TTL with SKOS hierarchy.")
    
    # Adjust the final tab reference
    final_tab = generate_validate_tab
    
    with final_tab:
        st.header("✅ Generate & Validate TTL")
        
        if 'selected_dataset' in st.session_state and 'field_mapping' in st.session_state:
            dataset = st.session_state['selected_dataset']
            field_mapping = st.session_state['field_mapping']
            
            st.subheader("Configuration Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Dataset:** {dataset['filename']}")
                st.write(f"**Records:** {len(dataset['data'])}")
                st.write(f"**Base URI:** {base_uri}")
            
            with col2:
                st.write(f"**Title:** {vocab_title}")
                st.write(f"**Mapped Fields:** {len(field_mapping)}")
                st.write("**Field Mapping:**")
                for skos_prop, field in field_mapping.items():
                    st.write(f"• {skos_prop} → {field}")
            
            if st.button("🚀 Generate TTL", type="primary"):
                with st.spinner("Generating TTL..."):
                    config = {
                        'base_uri': base_uri,
                        'vocab_name': vocab_name,
                        'vocab_title': vocab_title,
                        'vocab_description': vocab_description,
                        'field_mapping': field_mapping,
                        'additional_prefixes': additional_prefixes
                    }
                    
                    # IMPORTANT: Process hierarchy BEFORE field mapping!
                    # Hierarchy uses original column names, field mapping renames them
                    
                    # Get original raw data from uploaded files (BEFORE AI processing)
                    original_data = None
                    if 'uploaded_data' in st.session_state and st.session_state['uploaded_data']:
                        # Find the original dataset that matches our selected dataset
                        for uploaded_item in st.session_state['uploaded_data']:
                            if uploaded_item['filename'] == dataset['filename']:
                                # Use ORIGINAL_DATA field that was preserved before AI processing
                                original_data = uploaded_item.get('original_data', uploaded_item['data']).copy()
                                break
                        
                        # If no exact match, use the first uploaded dataset as fallback
                        if original_data is None:
                            original_data = st.session_state['uploaded_data'][0].get('original_data', st.session_state['uploaded_data'][0]['data']).copy()
                    
                    # Fallback to processed data if original not found
                    raw_data = original_data if original_data is not None else dataset['data'].copy()
                    
                    # Step 1: Process hierarchy with original column names
                    hierarchy_processed = False
                    
                    # Check for new hierarchy extractor results
                    if 'extracted_hierarchy' in st.session_state:
                        try:
                            hierarchy_result = st.session_state['extracted_hierarchy']
                            st.info(f"🌳 Using extracted hierarchy from: {hierarchy_result.get('source', 'Unknown')}")
                            
                            # Apply hierarchy relationships to the data
                            broader_relations = hierarchy_result.get('broader_relations', {})
                            narrower_relations = hierarchy_result.get('narrower_relations', {})
                            
                            if broader_relations or narrower_relations:
                                # Add hierarchy columns to the data
                                raw_data['broader'] = ''
                                raw_data['narrower'] = ''
                                
                                # Apply broader relationships
                                for concept_id, broader_id in broader_relations.items():
                                    mask = raw_data[field_mapping.get('uri', 'uri')] == concept_id
                                    if mask.any():
                                        raw_data.loc[mask, 'broader'] = broader_id
                                
                                # Apply narrower relationships
                                for broader_id, narrower_list in narrower_relations.items():
                                    mask = raw_data[field_mapping.get('uri', 'uri')] == broader_id
                                    if mask.any():
                                        raw_data.loc[mask, 'narrower'] = ' | '.join(narrower_list)
                                
                                hierarchy_processed = True
                                st.success(f"✅ Hierarchy applied: {len(broader_relations)} broader relations, {len(narrower_relations)} narrower relations")
                            
                        except Exception as e:
                            st.error(f"❌ Failed to apply extracted hierarchy: {str(e)}")
                    
                    # Fallback to old hierarchy processing if available
                    elif 'hierarchy_config' in st.session_state and st.session_state['hierarchy_config']:
                        try:
                            hierarchy_config = st.session_state['hierarchy_config']
                            st.info(f"🌳 Processing legacy hierarchy configuration: {hierarchy_config}")
                            st.info(f"Original columns: {list(raw_data.columns)}")
                            
                            # Simple level-based hierarchy processing for KldB
                            if 'level_field' in hierarchy_config and 'id_field' in hierarchy_config:
                                level_field = hierarchy_config['level_field']
                                id_field = hierarchy_config['id_field']
                                
                                st.info(f"Processing hierarchy: Level field '{level_field}', ID field '{id_field}'")
                                
                                # Simple KldB-style hierarchy processing
                                raw_data_copy = raw_data.copy()
                                raw_data_copy['broader'] = ''
                                raw_data_copy['narrower'] = ''
                                
                                # Convert level to numeric and sort
                                raw_data_copy[level_field] = pd.to_numeric(raw_data_copy[level_field], errors='coerce')
                                raw_data_copy = raw_data_copy.sort_values([level_field, id_field])
                                
                                # Process hierarchy relationships
                                levels = sorted(raw_data_copy[level_field].dropna().unique())
                                broader_count = 0
                                
                                for current_level in levels:
                                    if current_level <= 1:
                                        continue  # Top level has no broader concepts
                                    
                                    current_level_rows = raw_data_copy[raw_data_copy[level_field] == current_level]
                                    parent_level_rows = raw_data_copy[raw_data_copy[level_field] == current_level - 1]
                                    
                                    if parent_level_rows.empty:
                                        continue
                                    
                                    # Find parent for each concept at current level
                                    for idx, row in current_level_rows.iterrows():
                                        concept_id = str(row[id_field]).strip()
                                        
                                        # Find parent by longest prefix match
                                        best_parent = None
                                        best_match_length = 0
                                        
                                        for _, parent_row in parent_level_rows.iterrows():
                                            parent_id = str(parent_row[id_field]).strip()
                                            
                                            if concept_id.startswith(parent_id) and len(parent_id) > best_match_length:
                                                best_parent = parent_id
                                                best_match_length = len(parent_id)
                                        
                                        if best_parent:
                                            raw_data_copy.at[idx, 'broader'] = best_parent
                                            broader_count += 1
                                
                                # Create narrower relationships
                                narrower_dict = {}
                                for _, row in raw_data_copy.iterrows():
                                    broader_id = row['broader']
                                    child_id = str(row[id_field])
                                    
                                    if broader_id and broader_id != '':
                                        if broader_id not in narrower_dict:
                                            narrower_dict[broader_id] = []
                                        narrower_dict[broader_id].append(child_id)
                                
                                # Apply narrower relationships
                                for idx, row in raw_data_copy.iterrows():
                                    concept_id = str(row[id_field])
                                    if concept_id in narrower_dict:
                                        raw_data_copy.at[idx, 'narrower'] = ';'.join(narrower_dict[concept_id])
                                
                                raw_data = raw_data_copy
                                hierarchy_processed = True
                                
                                narrower_count = len([v for v in narrower_dict.values() if v])
                                
                                if broader_count > 0 or narrower_count > 0:
                                    st.success(f"✅ Hierarchy processed: {broader_count} broader relations, {narrower_count} parent concepts with children")
                                else:
                                    st.warning("⚠️ No hierarchy relationships found - check field configuration")
                            else:
                                st.warning("⚠️ Incomplete hierarchy configuration - missing level_field or id_field")
                                
                        except Exception as e:
                            st.error(f"❌ Hierarchy processing failed: {str(e)}")
                            st.error(f"Error type: {type(e).__name__}")
                            st.info("Continuing with original data...")
                    
                    if not hierarchy_processed:
                        st.info("ℹ️ No hierarchy configuration found - generating flat vocabulary")
                        st.info("💡 To add hierarchy, use Tab 6: Hierarchy & Processing to configure hierarchy extraction")
                    
                    # Step 2: Generate TTL with hierarchy-processed data
                    # The TTL generator will apply field mapping during generation
                    ttl_content = generator.generate_ttl(raw_data, config)
                    
                    # Validate with TTL Cleaner
                    st.subheader("🔍 Validation Results")
                    with st.spinner("Validating TTL with TTL Cleaner..."):
                        cleaned_ttl, stats = generator.validate_ttl(ttl_content)
                    
                    # Display validation results
                    if 'error' in stats:
                        st.error(f"❌ Validation error: {stats['error']}")
                        if stats.get('validation_failed'):
                            st.warning("⚠️ Advanced validation features unavailable - using basic TTL generation")
                    else:
                        st.success("✅ TTL validation completed with advanced SKOS checks!")
                        
                        # Enhanced statistics display
                        st.subheader("📊 Validation Statistics")
                        
                        # Row 1: Basic metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Concepts", stats.get('total_concepts', len(dataset['data'])))
                        with col2:
                            st.metric("Final Concepts", stats.get('final_concepts', stats.get('total_concepts', len(dataset['data']))))
                        with col3:
                            st.metric("Duplicates Removed", stats.get('duplicates_removed', 0))
                        with col4:
                            st.metric("Invalid Concepts", stats.get('invalid_concepts_removed', 0))
                        
                        # Row 2: Quality improvements
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Encoding Fixes", stats.get('encoding_issues_fixed', 0))
                        with col2:
                            st.metric("URI Fixes", stats.get('malformed_uris_fixed', 0))
                        with col3:
                            st.metric("Text Cleaned", stats.get('text_fields_cleaned', 0))
                        with col4:
                            st.metric("Comma Fixes", stats.get('comma_fixes', 0))
                        
                        # SKOS Validation Results
                        violations = stats.get('validation_violations', [])
                        warnings = stats.get('validation_warnings', [])
                        
                        if violations or warnings:
                            st.subheader("🔍 SKOS Validation Results")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if violations:
                                    st.error(f"❌ {len(violations)} SKOS Integrity Violations")
                                    with st.expander("View Violations", expanded=False):
                                        for i, violation in enumerate(violations[:10], 1):
                                            st.write(f"{i}. {violation}")
                                        if len(violations) > 10:
                                            st.caption(f"... and {len(violations) - 10} more violations")
                                else:
                                    st.success("✅ No SKOS integrity violations")
                            
                            with col2:
                                if warnings:
                                    st.warning(f"⚠️ {len(warnings)} SKOS Warnings")
                                    with st.expander("View Warnings", expanded=False):
                                        for i, warning in enumerate(warnings[:10], 1):
                                            st.write(f"{i}. {warning}")
                                        if len(warnings) > 10:
                                            st.caption(f"... and {len(warnings) - 10} more warnings")
                                else:
                                    st.success("✅ No SKOS warnings")
                        else:
                            st.success("🎉 Perfect SKOS compliance - no violations or warnings!")
                        
                        # Processing Quality Summary
                        errors = stats.get('errors', [])
                        processing_warnings = stats.get('warnings', [])
                        
                        if errors or processing_warnings:
                            st.subheader("⚙️ Processing Quality")
                            
                            if errors:
                                st.error(f"❌ {len(errors)} Processing Errors")
                                with st.expander("View Processing Errors", expanded=False):
                                    for i, error in enumerate(errors[:5], 1):
                                        st.write(f"{i}. {error}")
                                    if len(errors) > 5:
                                        st.caption(f"... and {len(errors) - 5} more errors")
                            
                            if processing_warnings:
                                st.warning(f"⚠️ {len(processing_warnings)} Processing Warnings")
                                with st.expander("View Processing Warnings", expanded=False):
                                    for i, warning in enumerate(processing_warnings[:5], 1):
                                        st.write(f"{i}. {warning}")
                                    if len(processing_warnings) > 5:
                                        st.caption(f"... and {len(processing_warnings) - 5} more warnings")
                    
                    # Display TTL content
                    st.subheader("📄 Generated TTL")
                    st.code(cleaned_ttl[:2000] + "..." if len(cleaned_ttl) > 2000 else cleaned_ttl, language="turtle")
                    
                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 Download TTL",
                            cleaned_ttl,
                            file_name=f"{vocab_title.lower().replace(' ', '_')}.ttl",
                            mime="text/turtle"
                        )
                    
                    with col2:
                        if stats and 'change_log' in stats:
                            st.download_button(
                                "📋 Download Validation Log",
                                stats['change_log'],
                                file_name=f"{vocab_title.lower().replace(' ', '_')}_validation.log",
                                mime="text/plain"
                            )
        else:
            st.info("Please complete data upload and field mapping first")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**SkoHub TTL Generator** | Built with Streamlit | "
        "Powered by TTL Cleaner | AI-enhanced with all-MiniLM-L12-v2"
    )

if __name__ == "__main__":
    main()
