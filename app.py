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
        
        self.field_suggestions = {
            'uri': ['id', 'uri', 'conceptUri', 'identifier', 'uuid', 'code'],
            'prefLabel': ['label', 'name', 'title', 'prefLabel', 'preferredLabel', 'term'],
            'altLabel': ['altLabel', 'alternativeLabel', 'synonym', 'aliases', 'alternative'],
            'description': ['description', 'definition', 'scopeNote', 'note', 'comment'],
            'broader': ['broader', 'parent', 'parentId', 'broaderConcept'],
            'narrower': ['narrower', 'children', 'childId', 'narrowerConcept']
        }
        
        self.similarity_model = None
        self.initialize_similarity_model()
    
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
            'uri': ['unique identifier', 'id', 'uri', 'concept identifier', 'resource identifier'],
            'prefLabel': ['main label', 'preferred label', 'title', 'name', 'primary label'],
            'altLabel': ['alternative label', 'synonym', 'alias', 'alternative name', 'other label'],
            'description': ['description', 'definition', 'scope note', 'explanation', 'comment'],
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
                # Try different encodings
                for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        return df, f"CSV loaded successfully with {encoding} encoding"
                    except UnicodeDecodeError:
                        continue
                return None, "Failed to load CSV with any encoding"
            
            elif file_extension == 'json':
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.json_normalize(data)
                return df, "JSON loaded successfully"
            
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
    
    def validate_ttl(self, ttl_content: str) -> Tuple[str, Dict]:
        """Validate TTL content using TTL Cleaner"""
        try:
            # Write TTL to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False, encoding='utf-8') as f:
                f.write(ttl_content)
                temp_file = f.name
            
            # Run TTL Cleaner
            cleaner = TTLCleaner()
            cleaned_content, stats = cleaner.clean_file(temp_file)
            
            # Clean up
            os.unlink(temp_file)
            
            return cleaned_content, stats
            
        except Exception as e:
            return ttl_content, {'error': str(e)}

def main():
    st.title("🏷️ SkoHub TTL Generator")
    st.markdown("**Universal tool for generating SKOS-compliant TTL vocabularies from CSV, JSON, or TTL files**")
    
    generator = SkoHubTTLGenerator()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Local AI Configuration
        st.subheader("🤖 Local AI Configuration")
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
        
        # Show standard prefixes
        with st.expander("📋 Standard Prefixes (included)"):
            st.write("• `skos:` → http://www.w3.org/2004/02/skos/core#")
            st.write("• `dct:` → http://purl.org/dc/terms/")
            st.write("• `xsd:` → http://www.w3.org/2001/XMLSchema#")
            st.write("• `esco:` → http://data.europa.eu/esco/skill/")
            st.write("• `isced:` → http://data.europa.eu/esco/isced-f/")
            st.write("• `isothes:` → http://purl.org/iso25964/skos-thes#")
        
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
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "🔧 Field Mapping", "📝 Manual Entry", "✅ Generate & Validate"])
    
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
                    df, message = generator.load_data_file(uploaded_file)
                    
                    if df is not None:
                        st.success(message)
                        st.dataframe(df.head())
                        st.info(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
                        
                        # Store data
                        st.session_state['uploaded_data'].append({
                            'filename': uploaded_file.name,
                            'data': df,
                            'columns': list(df.columns)
                        })
                        
                        # AI suggestions for field mapping
                        if generator.similarity_model and len(df) > 0:
                            if st.button(f"🤖 Get AI suggestions for {uploaded_file.name}"):
                                with st.spinner("Getting AI suggestions..."):
                                    sample_data = df.iloc[0].to_dict()
                                    suggestions = generator.get_ai_suggestions(sample_data, "field_mapping", similarity_threshold)
                                    if suggestions:
                                        st.info("AI Suggestions:")
                                        for suggestion in suggestions:
                                            st.write(f"• {suggestion}")
                    else:
                        st.error(message)
    
    with tab2:
        st.header("🔧 Field Mapping")
        
        if 'uploaded_data' in st.session_state and st.session_state['uploaded_data']:
            dataset_names = [item['filename'] for item in st.session_state['uploaded_data']]
            
            # Multi-dataset selection
            st.subheader("📊 Dataset Selection")
            
            # Initialize selected_data with default
            selected_data = None
            
            if len(dataset_names) > 1:
                use_multiple = st.checkbox("🔗 Combine multiple datasets", help="Combine data from multiple files into one vocabulary")
                
                if use_multiple:
                    # Multi-select for datasets
                    selected_datasets = st.multiselect(
                        "Select datasets to combine",
                        dataset_names,
                        default=dataset_names
                    )
                    
                    if len(selected_datasets) > 1:
                        # Combination strategy
                        col1, col2 = st.columns(2)
                        with col1:
                            join_strategy = st.selectbox(
                                "Combination strategy",
                                ["concat", "merge"],
                                help="Concat: Stack datasets vertically | Merge: Join on common field"
                            )
                        
                        with col2:
                            if join_strategy == "merge":
                                # Find common columns across selected datasets
                                common_cols = set(st.session_state['uploaded_data'][0]['columns'])
                                for item in st.session_state['uploaded_data'][1:]:
                                    if item['filename'] in selected_datasets:
                                        common_cols &= set(item['columns'])
                                
                                join_key = st.selectbox(
                                    "Join key",
                                    list(common_cols) if common_cols else ["id"],
                                    help="Field to join datasets on"
                                )
                            else:
                                join_key = None
                        
                        remove_duplicates = st.checkbox("Remove duplicates", value=True)
                        
                        # Combine datasets
                        if st.button("🔗 Combine Datasets"):
                            selected_data_items = [item for item in st.session_state['uploaded_data'] if item['filename'] in selected_datasets]
                            
                            join_config = {
                                'strategy': join_strategy,
                                'join_key': join_key,
                                'remove_duplicates': remove_duplicates
                            }
                            
                            combined_df = generator.combine_datasets(selected_data_items, join_config)
                            
                            # Store combined dataset
                            st.session_state['combined_dataset'] = {
                                'filename': f"Combined ({len(selected_datasets)} files)",
                                'data': combined_df,
                                'columns': list(combined_df.columns),
                                'source_files': selected_datasets
                            }
                            
                            st.success(f"✅ Combined {len(selected_datasets)} datasets into {len(combined_df)} records")
                            st.dataframe(combined_df.head())
                    else:
                        st.info("Select at least 2 datasets to combine")
                        # Fallback to first dataset if no combination yet
                        selected_data = st.session_state['uploaded_data'][0]
                else:
                    # Single dataset selection
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
            
            columns = selected_data['columns']
            
            st.subheader(f"Map fields for: {selected_data['filename']}")
            
            # Field mapping interface
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**SKOS Properties**")
                field_mapping = {}
                
                for skos_prop, suggestions in generator.field_suggestions.items():
                    # Find best suggestion
                    suggested_field = None
                    for suggestion in suggestions:
                        for col in columns:
                            if suggestion.lower() in col.lower():
                                suggested_field = col
                                break
                        if suggested_field:
                            break
                    
                    # Create selectbox
                    options = ["None"] + columns
                    default_index = options.index(suggested_field) if suggested_field else 0
                    
                    selected_field = st.selectbox(
                        f"{skos_prop}",
                        options,
                        index=default_index,
                        help=f"Suggested fields: {', '.join(suggestions)}"
                    )
                    
                    if selected_field != "None":
                        field_mapping[skos_prop] = selected_field
            
            with col2:
                st.write("**Data Preview**")
                if field_mapping:
                    preview_df = selected_data['data'][list(field_mapping.values())].head()
                    st.dataframe(preview_df)
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
    
    with tab4:
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
                    
                    # Generate TTL
                    ttl_content = generator.generate_ttl(dataset['data'], config)
                    
                    # Validate with TTL Cleaner
                    st.subheader("🔍 Validation Results")
                    with st.spinner("Validating TTL with TTL Cleaner..."):
                        cleaned_ttl, stats = generator.validate_ttl(ttl_content)
                    
                    # Display validation results
                    if 'error' in stats:
                        st.error(f"Validation error: {stats['error']}")
                    else:
                        st.success("TTL validation completed!")
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Concepts", len(dataset['data']))
                        with col2:
                            st.metric("Duplicates Removed", stats.get('duplicates_removed', 0))
                        with col3:
                            st.metric("Syntax Fixes", stats.get('syntax_fixes', 0))
                    
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
        "Powered by TTL Cleaner | AI-enhanced with OpenAI"
    )

if __name__ == "__main__":
    main()
