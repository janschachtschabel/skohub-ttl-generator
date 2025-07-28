"""
Data loading and processing module for SkoHub TTL Generator.
Handles CSV, JSON, and TTL file loading with proper encoding detection.
"""

import pandas as pd
import json
import re
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st


class DataLoader:
    """Handles loading and preprocessing of various data formats."""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'ttl']
    
    def load_data(self, uploaded_files) -> Tuple[Optional[pd.DataFrame], str, Dict[str, List[str]]]:
        """Load data from uploaded files and return available fields from all sources"""
        if not uploaded_files:
            return None, "No files uploaded", {}
        
        # Handle multiple files
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        merged_df = None
        status_messages = []
        available_fields = {}
        
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.lower().split('.')[-1]
            file_name = uploaded_file.name
            
            if file_extension == 'csv':
                df, status = self._load_csv(uploaded_file)
                if df is not None:
                    available_fields[f"CSV ({file_name})"] = list(df.columns)
                    if merged_df is None:
                        merged_df = df
                    else:
                        # For multiple CSVs, we could implement merging logic here
                        pass
                status_messages.append(f"CSV: {status}")
                
            elif file_extension == 'json':
                df, status = self._load_json(uploaded_file)
                if df is not None:
                    available_fields[f"JSON ({file_name})"] = list(df.columns)
                    if merged_df is None:
                        merged_df = df
                status_messages.append(f"JSON: {status}")
                
            elif file_extension in ['ttl', 'turtle', 'rdf']:
                df, status = self._load_ttl(uploaded_file)
                if df is not None:
                    available_fields[f"TTL ({file_name})"] = list(df.columns)
                    # TTL data is used for reference, not merged into main data
                status_messages.append(f"TTL: {status}")
                
            else:
                status_messages.append(f"Unsupported file format: {file_extension}")
        
        combined_status = "; ".join(status_messages)
        return merged_df, combined_status, available_fields
    
    def merge_data_with_join_keys(self, data_sources: Dict[str, pd.DataFrame], join_keys: Dict[str, str]) -> Tuple[Optional[pd.DataFrame], str]:
        """Merge data from multiple sources using specified join keys"""
        if len(data_sources) < 2 or len(join_keys) < 2:
            return None, "Need at least 2 data sources with join keys to merge"
        
        try:
            # Start with the first data source (usually CSV with main data)
            source_names = list(data_sources.keys())
            merged_df = data_sources[source_names[0]].copy()
            merge_info = [f"Started with {source_names[0]} ({len(merged_df)} rows)"]
            
            # Merge with each additional source
            for source_name in source_names[1:]:
                if source_name not in join_keys:
                    continue
                    
                source_df = data_sources[source_name]
                base_join_key = join_keys[source_names[0]]
                source_join_key = join_keys[source_name]
                
                if base_join_key not in merged_df.columns:
                    merge_info.append(f"Warning: Join key '{base_join_key}' not found in base data")
                    continue
                    
                if source_join_key not in source_df.columns:
                    merge_info.append(f"Warning: Join key '{source_join_key}' not found in {source_name}")
                    continue
                
                # Perform left join to preserve all records from base data
                before_count = len(merged_df)
                merged_df = merged_df.merge(
                    source_df,
                    left_on=base_join_key,
                    right_on=source_join_key,
                    how='left',
                    suffixes=('', f'_{source_name.split("(")[0].strip()}')
                )
                
                matched_count = len(merged_df[merged_df[source_join_key].notna()])
                merge_info.append(f"Merged with {source_name}: {matched_count}/{before_count} records matched")
            
            status = "; ".join(merge_info)
            return merged_df, f"Data merged successfully: {status}"
            
        except Exception as e:
            return None, f"Error merging data: {str(e)}"
    
    def _load_csv(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load CSV file with encoding detection and flexible parsing"""
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Try different combinations of encoding and CSV parameters
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        separators = [';', ',', '\t', '|']  # Try semicolon first for German CSV
        quote_chars = ['"', "'", None]
        
        best_result = None
        best_score = 0
        
        for encoding in encodings:
            for separator in separators:
                for quote_char in quote_chars:
                    try:
                        uploaded_file.seek(0)
                        
                        # Try different parsing strategies
                        parse_configs = [
                            # Standard parsing
                            {
                                'sep': separator,
                                'quotechar': quote_char,
                                'quoting': 1 if quote_char else 3,  # QUOTE_ALL or QUOTE_NONE
                                'engine': 'python',
                                'on_bad_lines': 'skip'
                            },
                            # More lenient parsing
                            {
                                'sep': separator,
                                'quotechar': quote_char,
                                'quoting': 3,  # QUOTE_NONE
                                'engine': 'python',
                                'on_bad_lines': 'skip',
                                'skipinitialspace': True
                            },
                            # Very lenient parsing
                            {
                                'sep': separator,
                                'quotechar': quote_char,
                                'quoting': 3,  # QUOTE_NONE
                                'engine': 'python',
                                'on_bad_lines': 'skip',
                                'skipinitialspace': True,
                                'doublequote': False,
                                'escapechar': '\\'
                            }
                        ]
                        
                        for config in parse_configs:
                            try:
                                uploaded_file.seek(0)
                                if quote_char is None:
                                    config.pop('quotechar', None)
                                
                                df = pd.read_csv(uploaded_file, encoding=encoding, **config)
                                
                                # Score the result based on number of columns and rows
                                if len(df) > 0:
                                    score = len(df.columns) * len(df)
                                    
                                    # Bonus for having more than 2 columns (better separation)
                                    if len(df.columns) > 2:
                                        score *= 2
                                    
                                    # Check if data looks reasonable (not all in one column)
                                    if len(df.columns) > 1 and score > best_score:
                                        best_result = (df, f"CSV loaded with {encoding} encoding, '{separator}' separator, {len(df.columns)} columns, {len(df)} rows")
                                        best_score = score
                                        
                                        # If we have a good result with many columns, use it
                                        if len(df.columns) >= 5:
                                            return best_result
                                            
                            except Exception:
                                continue
                                
                    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                        continue
                    except Exception:
                        continue
        
        if best_result:
            return best_result
        
        return None, "Failed to load CSV with any encoding/separator combination. Please check file format."
    
    def _load_json(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load JSON file and convert to DataFrame"""
        data = json.load(uploaded_file)
        
        if isinstance(data, list):
            # Handle list of objects with potential nested properties
            df = self._flatten_json_data(data)
            return df, f"JSON loaded successfully ({len(df)} records)"
        elif isinstance(data, dict):
            # Single object - convert to single-row DataFrame
            flattened_data = self._flatten_json_data([data])
            return flattened_data, "JSON loaded successfully (1 record)"
        else:
            return None, "JSON format not supported (must be object or array of objects)"
    
    def _load_ttl(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Load TTL file and extract SKOS concepts"""
        content = uploaded_file.read().decode('utf-8')
        concepts = self._extract_concepts_from_ttl(content)
        
        if not concepts:
            return None, "No SKOS concepts found in TTL file"
        
        df = pd.DataFrame(concepts)
        return df, f"TTL loaded successfully ({len(df)} concepts extracted)"
    
    def _flatten_json_data(self, data: List[Dict]) -> pd.DataFrame:
        """Flatten nested JSON data using pandas json_normalize"""
        # First, normalize the data
        df = pd.json_normalize(data)
        
        # Check for 'properties' columns and flatten them further
        properties_columns = [col for col in df.columns if col == 'properties' or col.startswith('properties.')]
        
        if properties_columns:
            # Process properties columns
            for col in properties_columns:
                if col == 'properties':
                    # Expand the properties column
                    properties_df = pd.json_normalize(df[col].dropna())
                    # Add 'properties.' prefix to new columns
                    properties_df.columns = [f'properties.{c}' for c in properties_df.columns]
                    # Merge back to main dataframe
                    df = pd.concat([df.drop(columns=[col]), properties_df], axis=1)
        
        # Recursively flatten any remaining object-type columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains dictionaries
                sample_values = df[col].dropna().head(3)
                if not sample_values.empty and isinstance(sample_values.iloc[0], dict):
                    # Flatten this column
                    nested_df = pd.json_normalize(df[col].dropna())
                    nested_df.columns = [f'{col}.{c}' for c in nested_df.columns]
                    # Replace original column with flattened columns
                    df = pd.concat([df.drop(columns=[col]), nested_df], axis=1)
        
        return df
    
    def _extract_concepts_from_ttl(self, content: str) -> List[Dict]:
        """Extract SKOS concepts from TTL content using regex"""
        concepts = []
        
        # Pattern to match SKOS concept blocks
        concept_pattern = r'<([^>]+)>\s+a\s+skos:Concept\s*;([^.]+)\.'
        
        matches = re.findall(concept_pattern, content, re.MULTILINE | re.DOTALL)
        
        for uri, properties in matches:
            concept = {'uri': uri}
            
            # Extract prefLabel
            pref_label_match = re.search(r'skos:prefLabel\s+"([^"]+)"(?:@(\w+))?', properties)
            if pref_label_match:
                concept['prefLabel'] = pref_label_match.group(1)
                if pref_label_match.group(2):
                    concept['language'] = pref_label_match.group(2)
            
            # Extract altLabel
            alt_labels = re.findall(r'skos:altLabel\s+"([^"]+)"', properties)
            if alt_labels:
                concept['altLabel'] = ' | '.join(alt_labels)
            
            # Extract definition
            definition_match = re.search(r'skos:definition\s+"([^"]+)"', properties)
            if definition_match:
                concept['definition'] = definition_match.group(1)
            
            # Extract broader relationships
            broader_matches = re.findall(r'skos:broader\s+<([^>]+)>', properties)
            if broader_matches:
                concept['broader'] = ' | '.join(broader_matches)
            
            concepts.append(concept)
        
        return concepts
    
    def merge_datasets(self, datasets: List[Dict]) -> Tuple[Optional[pd.DataFrame], str]:
        """Merge multiple datasets into a single DataFrame"""
        if not datasets:
            return None, "No datasets to merge"
        
        if len(datasets) == 1:
            return datasets[0]['data'], "Single dataset loaded"
        
        try:
            # Try to merge datasets intelligently
            merged_df = self._smart_merge(datasets)
            total_rows = sum(len(d['data']) for d in datasets)
            return merged_df, f"Merged {len(datasets)} datasets ({total_rows} total rows → {len(merged_df)} unique rows)"
        
        except Exception as e:
            # Fallback to simple concatenation
            st.warning(f"Smart merge failed: {e}. Using simple concatenation.")
            return self._fallback_concat(datasets), f"Concatenated {len(datasets)} datasets"
    
    def _smart_merge(self, datasets: List[Dict]) -> pd.DataFrame:
        """Intelligently merge datasets by aligning columns and removing duplicates"""
        all_dfs = []
        
        for dataset in datasets:
            df = dataset['data'].copy()
            
            # Add source information
            df['_source'] = dataset['name']
            all_dfs.append(df)
        
        # Concatenate all dataframes
        merged_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Remove duplicates efficiently
        return self._remove_duplicates_efficiently(merged_df)
    
    def _remove_duplicates_efficiently(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates efficiently for large datasets"""
        if len(df) <= 10000:
            # For small datasets, use standard pandas method
            return df.drop_duplicates()
        else:
            # For large datasets, use chunk-based approach
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
    
    def _fallback_concat(self, datasets: List[Dict]) -> pd.DataFrame:
        """Fallback concatenation when merge fails"""
        result_chunks = []
        for dataset in datasets:
            result_chunks.append(dataset['data'])
        
        return pd.concat(result_chunks, ignore_index=True)
