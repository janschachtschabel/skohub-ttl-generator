"""
Multi-file scenario handler for SkoHub TTL Generator.
Handles different data distribution scenarios (Kldb-style and Esco-style).
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import streamlit as st
from urllib.parse import urlparse


class ScenarioType(Enum):
    """Types of multi-file scenarios"""
    DISTRIBUTED_DATA = "distributed"  # Kldb-style: data split across files
    COMPLETE_RECORDS = "complete"     # Esco-style: complete records in separate files
    DATA_ENRICHMENT = "enrichment"    # Enrich existing TTL with additional data
    DATA_COMBINATION = "combination"  # Combine multiple datasets into larger collection


class FileRole(Enum):
    """Roles that files can play in multi-file scenarios"""
    PRIMARY = "primary"           # Main data file (base dataset)
    SECONDARY = "secondary"       # Additional data file
    HIERARCHY = "hierarchy"       # Hierarchy definition file
    METADATA = "metadata"         # Metadata/descriptions file
    ENRICHMENT_SOURCE = "enrichment_source"  # Source for enriching existing data
    EXISTING_TTL = "existing_ttl"  # Existing TTL to be enriched
    COMBINATION_SOURCE = "combination_source"  # Source for data combination


class MultiFileHandler:
    """Handles multi-file data scenarios and combinations."""
    
    def __init__(self):
        self.scenario_configs = {
            ScenarioType.DISTRIBUTED_DATA: {
                'name': 'Distributed Data',
                'description': 'Data for each record is split across multiple files',
                'example': 'URI and labels in file 1, descriptions in file 2',
                'required_roles': [FileRole.PRIMARY, FileRole.SECONDARY]
            },
            ScenarioType.COMPLETE_RECORDS: {
                'name': 'Complete Records', 
                'description': 'Each file contains complete records of different types',
                'example': 'Skills in file 1, occupations in file 2, both complete',
                'required_roles': [FileRole.PRIMARY]
            },
            ScenarioType.DATA_ENRICHMENT: {
                'name': 'Data Enrichment',
                'description': 'Enrich existing TTL with additional information from other files',
                'example': 'Existing TTL + CSV with descriptions/metadata',
                'required_roles': [FileRole.EXISTING_TTL, FileRole.ENRICHMENT_SOURCE]
            },
            ScenarioType.DATA_COMBINATION: {
                'name': 'Data Combination',
                'description': 'Combine multiple datasets into a larger collection',
                'example': 'Multiple CSV/JSON files merged into one vocabulary',
                'required_roles': [FileRole.COMBINATION_SOURCE]
            }
        }
    
    def get_scenario_config(self, scenario_type: ScenarioType) -> Dict:
        """Get configuration for a scenario type"""
        return self.scenario_configs.get(scenario_type, {})
    
    def detect_scenario_type(self, file_data: List[Dict]) -> ScenarioType:
        """Auto-detect the most likely scenario type based on file data"""
        if len(file_data) < 2:
            return ScenarioType.COMPLETE_RECORDS
        
        # Check if files have overlapping columns (distributed data indicator)
        all_columns = [set(data['columns']) for data in file_data]
        
        # Calculate column overlap
        overlap_scores = []
        for i, cols1 in enumerate(all_columns):
            for j, cols2 in enumerate(all_columns[i+1:], i+1):
                overlap = len(cols1.intersection(cols2))
                total = len(cols1.union(cols2))
                overlap_scores.append(overlap / total if total > 0 else 0)
        
        avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
        
        # If high overlap (>0.3), likely distributed data
        if avg_overlap > 0.3:
            return ScenarioType.DISTRIBUTED_DATA
        else:
            return ScenarioType.COMPLETE_RECORDS
    
    def configure_file_roles(self, file_data: List[Dict], scenario_type: ScenarioType) -> Dict[str, FileRole]:
        """Configure roles for files based on scenario type"""
        file_roles = {}
        
        if scenario_type == ScenarioType.DISTRIBUTED_DATA:
            # For distributed data, assign roles based on data characteristics
            for i, data in enumerate(file_data):
                filename = data['filename']
                columns = data['columns']
                
                # Primary file: has URI/ID and basic labels
                if any(col.lower() in ['uri', 'id', 'concepturi', 'identifier'] for col in columns):
                    if any(col.lower() in ['label', 'name', 'title', 'preflabel'] for col in columns):
                        file_roles[filename] = FileRole.PRIMARY
                        continue
                
                # Hierarchy file: has parent/broader relationships
                if any(col.lower() in ['parent', 'broader', 'parentid', 'level'] for col in columns):
                    file_roles[filename] = FileRole.HIERARCHY
                    continue
                
                # Secondary file: additional data
                file_roles[filename] = FileRole.SECONDARY
        
        else:  # COMPLETE_RECORDS
            # For complete records, all files are primary by default
            for data in file_data:
                file_roles[data['filename']] = FileRole.PRIMARY
        
        return file_roles
    
    def get_join_field_suggestions(self, file_data: List[Dict], file_roles: Dict[str, FileRole]) -> Dict[str, List[str]]:
        """Get all available fields from each file for join configuration"""
        file_fields = {}
        
        # Get columns from each file
        for data in file_data:
            filename = data['filename']
            columns = data['columns']
            file_fields[filename] = columns
        
        return file_fields
    
    def get_join_field_patterns(self) -> List[str]:
        """Get common join field patterns for suggestions"""
        return [
            'uri', 'id', 'concepturi', 'identifier', 'uuid', 'code',
            'sys:node-uuid', 'properties.sys:node-uuid',
            'conceptId', 'skillId', 'occupationId',
            'Schlüssel KldB 2010, V. 2020'  # KldB specific
        ]
    
    def normalize_uri_for_join(self, uri_value: str, base_uri: Optional[str] = None) -> str:
        """Normalize URI for joining (remove base URI if present)"""
        if not isinstance(uri_value, str):
            return str(uri_value)
        
        # If it's a full URI, extract the local part
        if uri_value.startswith('http'):
            parsed = urlparse(uri_value)
            # Extract the last part after the last slash
            local_part = parsed.path.split('/')[-1] or parsed.fragment
            return local_part
        
        # If base_uri is provided, remove it
        if base_uri and uri_value.startswith(base_uri):
            return uri_value[len(base_uri):].lstrip('/')
        
        return uri_value
    
    def prepare_join_data(self, file_data: List[Dict], join_config: Dict) -> List[Dict]:
        """Prepare data for joining based on configuration"""
        prepared_data = []
        
        join_field = join_config.get('join_field')
        normalize_uris = join_config.get('normalize_uris', False)
        base_uri = join_config.get('base_uri')
        
        for data in file_data:
            df = data['data'].copy()
            
            # Normalize join field if needed
            if normalize_uris and join_field in df.columns:
                df[join_field] = df[join_field].apply(
                    lambda x: self.normalize_uri_for_join(x, base_uri)
                )
            
            prepared_data.append({
                **data,
                'data': df
            })
        
        return prepared_data


class JoinStrategy:
    """Strategies for joining multiple datasets"""
    
    @staticmethod
    def distributed_join(datasets: List[Dict], join_config: Dict) -> pd.DataFrame:
        """Join datasets for distributed data scenario"""
        if not datasets:
            return pd.DataFrame()
        
        join_field = join_config.get('join_field', 'id')
        how = join_config.get('how', 'outer')
        
        # Start with the first dataset
        result_df = datasets[0]['data'].copy()
        
        # Join with remaining datasets
        for dataset in datasets[1:]:
            df = dataset['data']
            
            if join_field in result_df.columns and join_field in df.columns:
                # Perform the join
                result_df = pd.merge(
                    result_df, 
                    df, 
                    on=join_field, 
                    how=how,
                    suffixes=('', f'_{dataset["filename"].split(".")[0]}')
                )
            else:
                st.warning(f"Join field '{join_field}' not found in {dataset['filename']}")
        
        return result_df
    
    @staticmethod
    def complete_records_concat(datasets: List[Dict], join_config: Dict) -> pd.DataFrame:
        """Concatenate complete record datasets"""
        if not datasets:
            return pd.DataFrame()
        
        # Add source file information
        dataframes = []
        for dataset in datasets:
            df = dataset['data'].copy()
            df['_source_file'] = dataset['filename']
            dataframes.append(df)
        
        # Concatenate all datasets
        result_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # Remove duplicates if requested
        if join_config.get('remove_duplicates', True):
            # Try to use URI/ID field for deduplication
            dedupe_fields = ['uri', 'id', 'conceptUri', 'identifier']
            dedupe_field = None
            
            for field in dedupe_fields:
                if field in result_df.columns:
                    dedupe_field = field
                    break
            
            if dedupe_field:
                result_df = result_df.drop_duplicates(subset=[dedupe_field], keep='first')
            else:
                result_df = result_df.drop_duplicates()
        
        return result_df


class HierarchyProcessor:
    """Processes different types of hierarchy definitions"""
    
    @staticmethod
    def detect_hierarchy_type(df: pd.DataFrame) -> str:
        """Detect the type of hierarchy definition in the data"""
        columns = [col.lower() for col in df.columns]
        
        # Parent field based
        if any(field in columns for field in ['parent', 'parentid', 'broader', 'broaderid']):
            return 'parent_field'
        
        # Level based
        if any(field in columns for field in ['level', 'depth', 'tier', 'ebene']):
            return 'level_field'
        
        # Separate hierarchy file indicators
        if len(df.columns) == 2 and any(field in columns for field in ['child', 'parent', 'narrow', 'broad']):
            return 'hierarchy_pairs'
        
        return 'none'
    
    @staticmethod
    def process_parent_field_hierarchy(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process hierarchy defined by parent field"""
        parent_field = config.get('parent_field', 'parent')
        id_field = config.get('id_field', 'id')
        
        if parent_field not in df.columns or id_field not in df.columns:
            return df
        
        # Add broader relationships
        df['broader'] = df[parent_field]
        
        # Create narrower relationships (reverse mapping)
        narrower_map = df.groupby(parent_field)[id_field].apply(list).to_dict()
        df['narrower'] = df[id_field].map(narrower_map).fillna('').apply(
            lambda x: ';'.join(x) if isinstance(x, list) else ''
        )
        
        return df
    
    @staticmethod
    def process_level_hierarchy(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process hierarchy defined by level field"""
        level_field = config.get('level_field', 'level')
        id_field = config.get('id_field', 'id')
        
        if level_field not in df.columns:
            return df
        
        # Sort by level to process hierarchically
        df_sorted = df.sort_values([level_field, id_field])
        
        # For each concept, find its broader concept (previous level)
        df_sorted['broader'] = ''
        df_sorted['narrower'] = ''
        
        for idx, row in df_sorted.iterrows():
            current_level = row[level_field]
            if pd.isna(current_level):
                continue
                
            # Find broader concept (one level up)
            if current_level > 1:
                broader_candidates = df_sorted[
                    df_sorted[level_field] == current_level - 1
                ]
                if not broader_candidates.empty:
                    # Take the last concept from the previous level as broader
                    broader_id = broader_candidates.iloc[-1][id_field]
                    df_sorted.at[idx, 'broader'] = broader_id
        
        return df_sorted
