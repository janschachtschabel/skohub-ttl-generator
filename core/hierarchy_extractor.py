"""
Hierarchy extraction module for SkoHub TTL Generator.
Provides flexible hierarchy extraction from different sources (TTL, CSV, JSON).
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum
import streamlit as st
from urllib.parse import urlparse


class HierarchySource(Enum):
    """Sources for hierarchy extraction"""
    TTL_BROADER_NARROWER = "ttl_skos"      # Extract from TTL skos:broader/narrower
    CSV_LEVEL_FIELD = "csv_level"          # Extract from CSV level/depth field
    CSV_PARENT_FIELD = "csv_parent"        # Extract from CSV parent ID field
    CSV_PATH_FIELD = "csv_path"            # Extract from CSV hierarchical path
    HIERARCHY_FILE = "hierarchy_file"       # Separate hierarchy definition file
    AUTO_DETECT = "auto_detect"            # Automatically detect best source


class HierarchyExtractor:
    """Extracts hierarchy information from different data sources."""
    
    def __init__(self):
        self.extraction_methods = {
            HierarchySource.TTL_BROADER_NARROWER: self._extract_from_ttl_skos,
            HierarchySource.CSV_LEVEL_FIELD: self._extract_from_csv_level,
            HierarchySource.CSV_PARENT_FIELD: self._extract_from_csv_parent,
            HierarchySource.CSV_PATH_FIELD: self._extract_from_csv_path,
            HierarchySource.HIERARCHY_FILE: self._extract_from_hierarchy_file,
            HierarchySource.AUTO_DETECT: self._auto_detect_and_extract
        }
        
        # Field patterns for auto-detection
        self.field_patterns = {
            'level_fields': [
                'level', 'depth', 'tier', 'ebene', 'stufe', 'hierarchy_level',
                'classification_level', 'taxonomic_level', 'rang'
            ],
            'parent_fields': [
                'parent', 'parentid', 'parent_id', 'broader', 'broaderid', 
                'broader_id', 'super', 'superid', 'category', 'categoryid',
                'übergeordnet', 'parent_code', 'parent_key'
            ],
            'path_fields': [
                'path', 'hierarchy_path', 'classification_path', 'breadcrumb',
                'pfad', 'hierarchie_pfad'
            ]
        }
    
    def get_available_sources(self, datasets: List[Dict]) -> List[HierarchySource]:
        """Get available hierarchy sources based on provided datasets."""
        available_sources = [HierarchySource.AUTO_DETECT]
        
        for dataset in datasets:
            data = dataset.get('data')
            file_type = dataset.get('type', '').lower()
            
            if file_type == 'ttl':
                # Check if TTL has SKOS hierarchy
                if self._has_skos_hierarchy(data):
                    available_sources.append(HierarchySource.TTL_BROADER_NARROWER)
            
            elif isinstance(data, pd.DataFrame):
                columns_lower = [col.lower() for col in data.columns]
                
                # Check for level fields
                if any(field in columns_lower for field in self.field_patterns['level_fields']):
                    available_sources.append(HierarchySource.CSV_LEVEL_FIELD)
                
                # Check for parent fields
                if any(field in columns_lower for field in self.field_patterns['parent_fields']):
                    available_sources.append(HierarchySource.CSV_PARENT_FIELD)
                
                # Check for path fields
                if any(field in columns_lower for field in self.field_patterns['path_fields']):
                    available_sources.append(HierarchySource.CSV_PATH_FIELD)
        
        return list(set(available_sources))
    
    def extract_hierarchy(self, source: HierarchySource, datasets: List[Dict], 
                         config: Dict = None) -> Dict[str, Any]:
        """Extract hierarchy from specified source."""
        if source not in self.extraction_methods:
            raise ValueError(f"Unsupported hierarchy source: {source}")
        
        return self.extraction_methods[source](datasets, config or {})
    
    def _extract_from_ttl_skos(self, datasets: List[Dict], config: Dict) -> Dict[str, Any]:
        """Extract hierarchy from TTL SKOS broader/narrower relationships."""
        ttl_dataset = None
        for dataset in datasets:
            if dataset.get('type', '').lower() == 'ttl':
                ttl_dataset = dataset
                break
        
        if not ttl_dataset:
            return {'error': 'No TTL dataset found for SKOS hierarchy extraction'}
        
        ttl_content = ttl_dataset['data']
        hierarchy = {
            'broader_relations': {},
            'narrower_relations': {},
            'concepts': set(),
            'source': 'TTL SKOS'
        }
        
        # Extract broader relationships
        broader_pattern = r'<([^>]+)>\s+skos:broader\s+<([^>]+)>'
        for match in re.finditer(broader_pattern, ttl_content):
            concept, broader = match.groups()
            hierarchy['broader_relations'][concept] = broader
            hierarchy['concepts'].add(concept)
            hierarchy['concepts'].add(broader)
        
        # Extract narrower relationships
        narrower_pattern = r'<([^>]+)>\s+skos:narrower\s+<([^>]+)>'
        for match in re.finditer(narrower_pattern, ttl_content):
            broader, narrower = match.groups()
            if broader not in hierarchy['narrower_relations']:
                hierarchy['narrower_relations'][broader] = []
            hierarchy['narrower_relations'][broader].append(narrower)
            hierarchy['concepts'].add(broader)
            hierarchy['concepts'].add(narrower)
        
        return hierarchy
    
    def _extract_from_csv_level(self, datasets: List[Dict], config: Dict) -> Dict[str, Any]:
        """Extract hierarchy from CSV level/depth field."""
        csv_dataset = self._find_csv_dataset(datasets)
        if not csv_dataset:
            return {'error': 'No CSV dataset found for level-based hierarchy extraction'}
        
        df = csv_dataset['data']
        level_field = config.get('level_field')
        id_field = config.get('id_field')
        
        if not level_field:
            # Auto-detect level field
            level_field = self._detect_level_field(df)
        
        if not id_field:
            # Auto-detect ID field
            id_field = self._detect_id_field(df)
        
        if not level_field or not id_field:
            return {'error': f'Could not detect level field ({level_field}) or ID field ({id_field})'}
        
        hierarchy = {
            'broader_relations': {},
            'narrower_relations': {},
            'levels': {},
            'concepts': set(),
            'source': f'CSV Level Field: {level_field}'
        }
        
        # Sort by level and build hierarchy
        df_sorted = df.sort_values([level_field, id_field])
        level_stacks = {}  # level -> last concept at that level
        
        for _, row in df_sorted.iterrows():
            concept_id = str(row[id_field])
            level = int(row[level_field])
            
            hierarchy['concepts'].add(concept_id)
            hierarchy['levels'][concept_id] = level
            
            # Find parent (last concept at level-1)
            if level > 1 and (level - 1) in level_stacks:
                parent_id = level_stacks[level - 1]
                hierarchy['broader_relations'][concept_id] = parent_id
                
                if parent_id not in hierarchy['narrower_relations']:
                    hierarchy['narrower_relations'][parent_id] = []
                hierarchy['narrower_relations'][parent_id].append(concept_id)
            
            # Update level stack
            level_stacks[level] = concept_id
            # Clear deeper levels
            keys_to_remove = [k for k in level_stacks.keys() if k > level]
            for k in keys_to_remove:
                del level_stacks[k]
        
        return hierarchy
    
    def _extract_from_csv_parent(self, datasets: List[Dict], config: Dict) -> Dict[str, Any]:
        """Extract hierarchy from CSV parent ID field."""
        csv_dataset = self._find_csv_dataset(datasets)
        if not csv_dataset:
            return {'error': 'No CSV dataset found for parent-based hierarchy extraction'}
        
        df = csv_dataset['data']
        parent_field = config.get('parent_field')
        id_field = config.get('id_field')
        
        if not parent_field:
            parent_field = self._detect_parent_field(df)
        
        if not id_field:
            id_field = self._detect_id_field(df)
        
        if not parent_field or not id_field:
            return {'error': f'Could not detect parent field ({parent_field}) or ID field ({id_field})'}
        
        hierarchy = {
            'broader_relations': {},
            'narrower_relations': {},
            'concepts': set(),
            'source': f'CSV Parent Field: {parent_field}'
        }
        
        for _, row in df.iterrows():
            concept_id = str(row[id_field])
            parent_id = row.get(parent_field)
            
            hierarchy['concepts'].add(concept_id)
            
            if pd.notna(parent_id) and str(parent_id).strip():
                parent_id = str(parent_id)
                hierarchy['broader_relations'][concept_id] = parent_id
                hierarchy['concepts'].add(parent_id)
                
                if parent_id not in hierarchy['narrower_relations']:
                    hierarchy['narrower_relations'][parent_id] = []
                hierarchy['narrower_relations'][parent_id].append(concept_id)
        
        return hierarchy
    
    def _extract_from_csv_path(self, datasets: List[Dict], config: Dict) -> Dict[str, Any]:
        """Extract hierarchy from CSV hierarchical path field."""
        csv_dataset = self._find_csv_dataset(datasets)
        if not csv_dataset:
            return {'error': 'No CSV dataset found for path-based hierarchy extraction'}
        
        df = csv_dataset['data']
        path_field = config.get('path_field')
        id_field = config.get('id_field')
        separator = config.get('path_separator', '/')
        
        if not path_field:
            path_field = self._detect_path_field(df)
        
        if not id_field:
            id_field = self._detect_id_field(df)
        
        if not path_field or not id_field:
            return {'error': f'Could not detect path field ({path_field}) or ID field ({id_field})'}
        
        hierarchy = {
            'broader_relations': {},
            'narrower_relations': {},
            'paths': {},
            'concepts': set(),
            'source': f'CSV Path Field: {path_field}'
        }
        
        for _, row in df.iterrows():
            concept_id = str(row[id_field])
            path = str(row.get(path_field, ''))
            
            hierarchy['concepts'].add(concept_id)
            hierarchy['paths'][concept_id] = path
            
            # Split path and build hierarchy
            path_parts = [part.strip() for part in path.split(separator) if part.strip()]
            
            if len(path_parts) > 1:
                # Find parent (second-to-last part or match by path)
                parent_path = separator.join(path_parts[:-1])
                
                # Find concept with parent path
                for other_id, other_path in hierarchy['paths'].items():
                    if other_path == parent_path and other_id != concept_id:
                        hierarchy['broader_relations'][concept_id] = other_id
                        
                        if other_id not in hierarchy['narrower_relations']:
                            hierarchy['narrower_relations'][other_id] = []
                        hierarchy['narrower_relations'][other_id].append(concept_id)
                        break
        
        return hierarchy
    
    def _extract_from_hierarchy_file(self, datasets: List[Dict], config: Dict) -> Dict[str, Any]:
        """Extract hierarchy from separate hierarchy definition file."""
        hierarchy_dataset = None
        for dataset in datasets:
            if dataset.get('role') == 'hierarchy':
                hierarchy_dataset = dataset
                break
        
        if not hierarchy_dataset:
            return {'error': 'No hierarchy file found'}
        
        df = hierarchy_dataset['data']
        child_field = config.get('child_field', 'child')
        parent_field = config.get('parent_field', 'parent')
        
        hierarchy = {
            'broader_relations': {},
            'narrower_relations': {},
            'concepts': set(),
            'source': 'Separate Hierarchy File'
        }
        
        for _, row in df.iterrows():
            child_id = str(row.get(child_field, ''))
            parent_id = str(row.get(parent_field, ''))
            
            if child_id and parent_id:
                hierarchy['broader_relations'][child_id] = parent_id
                hierarchy['concepts'].add(child_id)
                hierarchy['concepts'].add(parent_id)
                
                if parent_id not in hierarchy['narrower_relations']:
                    hierarchy['narrower_relations'][parent_id] = []
                hierarchy['narrower_relations'][parent_id].append(child_id)
        
        return hierarchy
    
    def _auto_detect_and_extract(self, datasets: List[Dict], config: Dict) -> Dict[str, Any]:
        """Automatically detect best hierarchy source and extract."""
        available_sources = self.get_available_sources(datasets)
        
        # Priority order for auto-detection
        priority_order = [
            HierarchySource.TTL_BROADER_NARROWER,
            HierarchySource.CSV_LEVEL_FIELD,
            HierarchySource.CSV_PARENT_FIELD,
            HierarchySource.CSV_PATH_FIELD,
            HierarchySource.HIERARCHY_FILE
        ]
        
        for source in priority_order:
            if source in available_sources:
                result = self.extract_hierarchy(source, datasets, config)
                if 'error' not in result:
                    result['auto_detected'] = True
                    return result
        
        return {'error': 'No suitable hierarchy source found for auto-detection'}
    
    # Helper methods
    def _find_csv_dataset(self, datasets: List[Dict]) -> Optional[Dict]:
        """Find first CSV dataset."""
        for dataset in datasets:
            if isinstance(dataset.get('data'), pd.DataFrame):
                return dataset
        return None
    
    def _has_skos_hierarchy(self, ttl_content: str) -> bool:
        """Check if TTL content has SKOS hierarchy relationships."""
        broader_pattern = r'skos:broader'
        narrower_pattern = r'skos:narrower'
        return bool(re.search(broader_pattern, ttl_content) or re.search(narrower_pattern, ttl_content))
    
    def _detect_level_field(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect level field in DataFrame."""
        columns_lower = {col.lower(): col for col in df.columns}
        
        for pattern in self.field_patterns['level_fields']:
            if pattern in columns_lower:
                return columns_lower[pattern]
        return None
    
    def _detect_parent_field(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect parent field in DataFrame."""
        columns_lower = {col.lower(): col for col in df.columns}
        
        for pattern in self.field_patterns['parent_fields']:
            if pattern in columns_lower:
                return columns_lower[pattern]
        return None
    
    def _detect_path_field(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect path field in DataFrame."""
        columns_lower = {col.lower(): col for col in df.columns}
        
        for pattern in self.field_patterns['path_fields']:
            if pattern in columns_lower:
                return columns_lower[pattern]
        return None
    
    def _detect_id_field(self, df: pd.DataFrame) -> Optional[str]:
        """Auto-detect ID field in DataFrame."""
        # Common ID field patterns
        id_patterns = [
            'id', 'uri', 'concept_id', 'conceptid', 'identifier', 'code', 
            'key', 'schlüssel', 'uuid', 'concept_uri', 'iri'
        ]
        
        columns_lower = {col.lower(): col for col in df.columns}
        
        for pattern in id_patterns:
            if pattern in columns_lower:
                return columns_lower[pattern]
        
        # Fallback: first column
        if len(df.columns) > 0:
            return df.columns[0]
        
        return None
    
    def get_hierarchy_config_ui(self, source: HierarchySource, datasets: List[Dict]) -> Dict:
        """Generate Streamlit UI configuration for hierarchy extraction."""
        config = {}
        
        if source == HierarchySource.CSV_LEVEL_FIELD:
            csv_dataset = self._find_csv_dataset(datasets)
            if csv_dataset:
                df = csv_dataset['data']
                
                # Level field selection
                level_options = [col for col in df.columns 
                               if col.lower() in self.field_patterns['level_fields']]
                if level_options:
                    config['level_field'] = st.selectbox(
                        "Level/Depth Field", 
                        options=level_options,
                        help="Field containing hierarchy level/depth information"
                    )
                
                # ID field selection
                id_options = df.columns.tolist()
                config['id_field'] = st.selectbox(
                    "ID Field",
                    options=id_options,
                    help="Field containing unique concept identifiers"
                )
        
        elif source == HierarchySource.CSV_PARENT_FIELD:
            csv_dataset = self._find_csv_dataset(datasets)
            if csv_dataset:
                df = csv_dataset['data']
                
                # Parent field selection
                parent_options = [col for col in df.columns 
                                if col.lower() in self.field_patterns['parent_fields']]
                if parent_options:
                    config['parent_field'] = st.selectbox(
                        "Parent ID Field",
                        options=parent_options,
                        help="Field containing parent concept IDs"
                    )
                
                # ID field selection
                id_options = df.columns.tolist()
                config['id_field'] = st.selectbox(
                    "ID Field",
                    options=id_options,
                    help="Field containing unique concept identifiers"
                )
        
        elif source == HierarchySource.CSV_PATH_FIELD:
            csv_dataset = self._find_csv_dataset(datasets)
            if csv_dataset:
                df = csv_dataset['data']
                
                # Path field selection
                path_options = [col for col in df.columns 
                              if col.lower() in self.field_patterns['path_fields']]
                if path_options:
                    config['path_field'] = st.selectbox(
                        "Hierarchical Path Field",
                        options=path_options,
                        help="Field containing hierarchical path (e.g., 'A/B/C')"
                    )
                
                # Path separator
                config['path_separator'] = st.text_input(
                    "Path Separator",
                    value="/",
                    help="Character used to separate path levels"
                )
                
                # ID field selection
                id_options = df.columns.tolist()
                config['id_field'] = st.selectbox(
                    "ID Field",
                    options=id_options,
                    help="Field containing unique concept identifiers"
                )
        
        return config
