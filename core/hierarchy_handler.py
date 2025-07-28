"""
Hierarchy handling module for SkoHub TTL Generator.
Processes different types of SKOS hierarchy definitions and relationships.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import streamlit as st


class HierarchyType(Enum):
    """Types of hierarchy definitions supported"""
    PARENT_FIELD = "parent_field"      # Parent ID in each record
    LEVEL_FIELD = "level_field"        # Numeric level/depth field
    HIERARCHY_FILE = "hierarchy_file"   # Separate file with parent-child pairs
    BROADER_NARROWER = "broader_narrower"  # Direct SKOS broader/narrower fields
    PATH_BASED = "path_based"          # Hierarchical path (e.g., "A/B/C")


class HierarchyHandler:
    """Handles different types of hierarchy processing for SKOS vocabularies."""
    
    def __init__(self):
        self.hierarchy_patterns = {
            'parent_fields': [
                'parent', 'parentid', 'parent_id', 'broader', 'broaderid', 
                'broader_id', 'super', 'superid', 'category', 'categoryid'
            ],
            'level_fields': [
                'level', 'depth', 'tier', 'ebene', 'stufe', 'hierarchy_level',
                'classification_level', 'taxonomic_level'
            ],
            'path_fields': [
                'path', 'hierarchy_path', 'classification_path', 'breadcrumb'
            ],
            'broader_fields': [
                'broader', 'broaderconcept', 'broader_concept', 'skos:broader'
            ],
            'narrower_fields': [
                'narrower', 'narrowerconcept', 'narrower_concept', 'skos:narrower'
            ]
        }
    
    def detect_hierarchy_type(self, df: pd.DataFrame, additional_files: List[Dict] = None) -> HierarchyType:
        """Detect the type of hierarchy definition in the data"""
        columns_lower = [col.lower() for col in df.columns]
        
        # Check for direct SKOS fields first
        if any(field in columns_lower for field in self.hierarchy_patterns['broader_fields']):
            return HierarchyType.BROADER_NARROWER
        
        # Check for parent fields
        if any(field in columns_lower for field in self.hierarchy_patterns['parent_fields']):
            return HierarchyType.PARENT_FIELD
        
        # Check for level fields
        if any(field in columns_lower for field in self.hierarchy_patterns['level_fields']):
            return HierarchyType.LEVEL_FIELD
        
        # Check for path fields
        if any(field in columns_lower for field in self.hierarchy_patterns['path_fields']):
            return HierarchyType.PATH_BASED
        
        # Check for separate hierarchy file
        if additional_files:
            for file_data in additional_files:
                file_columns = [col.lower() for col in file_data.get('columns', [])]
                if (len(file_columns) >= 2 and 
                    any('parent' in col or 'child' in col for col in file_columns)):
                    return HierarchyType.HIERARCHY_FILE
        
        return HierarchyType.PARENT_FIELD  # Default fallback
    
    def get_hierarchy_field_suggestions(self, df: pd.DataFrame, hierarchy_type: HierarchyType) -> List[str]:
        """Get field suggestions based on hierarchy type"""
        columns = df.columns.tolist()
        suggestions = []
        
        if hierarchy_type == HierarchyType.PARENT_FIELD:
            for col in columns:
                if col.lower() in self.hierarchy_patterns['parent_fields']:
                    suggestions.append(col)
        
        elif hierarchy_type == HierarchyType.LEVEL_FIELD:
            for col in columns:
                if col.lower() in self.hierarchy_patterns['level_fields']:
                    suggestions.append(col)
        
        elif hierarchy_type == HierarchyType.PATH_BASED:
            for col in columns:
                if col.lower() in self.hierarchy_patterns['path_fields']:
                    suggestions.append(col)
        
        elif hierarchy_type == HierarchyType.BROADER_NARROWER:
            for col in columns:
                col_lower = col.lower()
                if (col_lower in self.hierarchy_patterns['broader_fields'] or 
                    col_lower in self.hierarchy_patterns['narrower_fields']):
                    suggestions.append(col)
        
        return suggestions
    
    def process_hierarchy(self, df: pd.DataFrame, hierarchy_config: Dict) -> pd.DataFrame:
        """Process hierarchy based on configuration"""
        hierarchy_type = HierarchyType(hierarchy_config.get('type', 'parent_field'))
        
        if hierarchy_type == HierarchyType.PARENT_FIELD:
            return self._process_parent_field_hierarchy(df, hierarchy_config)
        elif hierarchy_type == HierarchyType.LEVEL_FIELD:
            return self._process_level_hierarchy(df, hierarchy_config)
        elif hierarchy_type == HierarchyType.PATH_BASED:
            return self._process_path_hierarchy(df, hierarchy_config)
        elif hierarchy_type == HierarchyType.BROADER_NARROWER:
            return self._process_skos_hierarchy(df, hierarchy_config)
        else:
            return df
    
    def _process_parent_field_hierarchy(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process hierarchy defined by parent field"""
        parent_field = config.get('parent_field')
        id_field = config.get('id_field', 'id')
        
        if not parent_field or parent_field not in df.columns:
            return df
        
        df_result = df.copy()
        
        # Create broader relationships from parent field
        df_result['broader'] = df_result[parent_field].fillna('')
        
        # Create narrower relationships (reverse mapping)
        narrower_dict = {}
        for _, row in df_result.iterrows():
            parent_id = row[parent_field]
            child_id = row[id_field]
            
            if pd.notna(parent_id) and parent_id != '':
                if parent_id not in narrower_dict:
                    narrower_dict[parent_id] = []
                narrower_dict[parent_id].append(str(child_id))
        
        # Add narrower field
        df_result['narrower'] = df_result[id_field].map(narrower_dict).fillna([]).apply(
            lambda x: ';'.join(x) if x else ''
        )
        
        return df_result
    
    def _process_level_hierarchy(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process hierarchy defined by level field"""
        level_field = config.get('level_field')
        id_field = config.get('id_field', 'id')
        label_field = config.get('label_field', 'label')
        
        if not level_field or level_field not in df.columns:
            return df
        
        df_result = df.copy()
        
        # Convert level to numeric
        df_result[level_field] = pd.to_numeric(df_result[level_field], errors='coerce')
        
        # Sort by level and position
        df_result = df_result.sort_values([level_field, id_field])
        
        # Initialize hierarchy fields
        df_result['broader'] = ''
        df_result['narrower'] = ''
        
        # Process each level
        levels = sorted(df_result[level_field].dropna().unique())
        
        for current_level in levels:
            if current_level <= 1:
                continue  # Top level has no broader concepts
            
            current_level_df = df_result[df_result[level_field] == current_level]
            parent_level_df = df_result[df_result[level_field] == current_level - 1]
            
            if parent_level_df.empty:
                continue
            
            # KldB-style hierarchy: find parent based on ID structure
            for idx, row in current_level_df.iterrows():
                concept_id = str(row[id_field])
                
                # For KldB-style numeric hierarchies, find parent by ID prefix
                parent_id = self._find_numeric_parent(concept_id, parent_level_df, id_field)
                
                if parent_id:
                    df_result.at[idx, 'broader'] = parent_id
        
        # Create narrower relationships
        narrower_dict = {}
        for _, row in df_result.iterrows():
            broader_id = row['broader']
            child_id = row[id_field]
            
            if broader_id and broader_id != '':
                if broader_id not in narrower_dict:
                    narrower_dict[broader_id] = []
                narrower_dict[broader_id].append(str(child_id))
        
        df_result['narrower'] = df_result[id_field].map(narrower_dict).fillna([]).apply(
            lambda x: ';'.join(x) if x else ''
        )
        
        return df_result
    
    def _find_numeric_parent(self, concept_id: str, parent_level_df: pd.DataFrame, id_field: str) -> str:
        """Find parent for KldB-style numeric hierarchies based on ID structure"""
        concept_id = str(concept_id).strip()
        
        # For KldB: 0 -> 01 -> 011 -> 0110 -> 01104
        # Parent is the longest matching prefix from parent level
        best_parent = None
        best_match_length = 0
        
        for _, parent_row in parent_level_df.iterrows():
            parent_id = str(parent_row[id_field]).strip()
            
            # Check if concept_id starts with parent_id (prefix match)
            if concept_id.startswith(parent_id) and len(parent_id) > best_match_length:
                best_parent = parent_id
                best_match_length = len(parent_id)
        
        # Alternative: if no prefix match, try to find parent by removing last digit(s)
        if not best_parent and len(concept_id) > 1:
            # Try removing last character: 011 -> 01, 0110 -> 011
            potential_parent = concept_id[:-1]
            
            for _, parent_row in parent_level_df.iterrows():
                parent_id = str(parent_row[id_field]).strip()
                if parent_id == potential_parent:
                    best_parent = parent_id
                    break
            
            # If still no match, try removing more characters for longer IDs
            if not best_parent and len(concept_id) > 2:
                potential_parent = concept_id[:-2] if concept_id[-2:].isdigit() else concept_id[:-1]
                
                for _, parent_row in parent_level_df.iterrows():
                    parent_id = str(parent_row[id_field]).strip()
                    if parent_id == potential_parent:
                        best_parent = parent_id
                        break
        
        return best_parent
    
    def _process_path_hierarchy(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process hierarchy defined by hierarchical path"""
        path_field = config.get('path_field')
        id_field = config.get('id_field', 'id')
        separator = config.get('path_separator', '/')
        
        if not path_field or path_field not in df.columns:
            return df
        
        df_result = df.copy()
        df_result['broader'] = ''
        df_result['narrower'] = ''
        
        # Create path-to-id mapping
        path_to_id = {}
        for _, row in df_result.iterrows():
            path = str(row[path_field])
            concept_id = row[id_field]
            path_to_id[path] = concept_id
        
        # Process each concept's path
        narrower_dict = {}
        
        for _, row in df_result.iterrows():
            path = str(row[path_field])
            concept_id = row[id_field]
            
            if separator in path:
                path_parts = path.split(separator)
                if len(path_parts) > 1:
                    # Parent path is all parts except the last
                    parent_path = separator.join(path_parts[:-1])
                    
                    # Find parent ID
                    if parent_path in path_to_id:
                        parent_id = path_to_id[parent_path]
                        df_result.loc[df_result[id_field] == concept_id, 'broader'] = parent_id
                        
                        # Add to narrower mapping
                        if parent_id not in narrower_dict:
                            narrower_dict[parent_id] = []
                        narrower_dict[parent_id].append(str(concept_id))
        
        # Add narrower relationships
        df_result['narrower'] = df_result[id_field].map(narrower_dict).fillna([]).apply(
            lambda x: ';'.join(x) if x else ''
        )
        
        return df_result
    
    def _process_skos_hierarchy(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Process existing SKOS broader/narrower relationships"""
        broader_field = config.get('broader_field')
        narrower_field = config.get('narrower_field')
        id_field = config.get('id_field', 'id')
        
        df_result = df.copy()
        
        # Ensure broader and narrower columns exist
        if 'broader' not in df_result.columns:
            df_result['broader'] = ''
        if 'narrower' not in df_result.columns:
            df_result['narrower'] = ''
        
        # Copy from existing fields if they exist
        if broader_field and broader_field in df_result.columns:
            df_result['broader'] = df_result[broader_field].fillna('')
        
        if narrower_field and narrower_field in df_result.columns:
            df_result['narrower'] = df_result[narrower_field].fillna('')
        
        return df_result
    
    def validate_hierarchy(self, df: pd.DataFrame) -> Dict[str, List]:
        """Validate hierarchy for common issues"""
        issues = {
            'circular_references': [],
            'orphaned_concepts': [],
            'missing_parents': [],
            'invalid_relationships': []
        }
        
        if 'broader' not in df.columns or 'id' not in df.columns:
            return issues
        
        id_field = 'id'
        broader_field = 'broader'
        
        # Get all concept IDs
        all_ids = set(df[id_field].astype(str))
        
        # Check for circular references
        for _, row in df.iterrows():
            concept_id = str(row[id_field])
            broader_id = str(row[broader_field]) if pd.notna(row[broader_field]) else ''
            
            if broader_id and broader_id != '':
                # Check if broader concept exists
                if broader_id not in all_ids:
                    issues['missing_parents'].append({
                        'concept_id': concept_id,
                        'missing_parent': broader_id
                    })
                
                # Check for circular reference (concept is its own ancestor)
                if self._has_circular_reference(df, concept_id, broader_id, id_field, broader_field):
                    issues['circular_references'].append({
                        'concept_id': concept_id,
                        'broader_id': broader_id
                    })
        
        return issues
    
    def _has_circular_reference(self, df: pd.DataFrame, concept_id: str, broader_id: str, 
                               id_field: str, broader_field: str, visited: Set[str] = None) -> bool:
        """Check for circular references in hierarchy"""
        if visited is None:
            visited = set()
        
        if concept_id in visited:
            return True
        
        if broader_id == concept_id:
            return True
        
        visited.add(concept_id)
        
        # Find the broader concept's broader concept
        broader_row = df[df[id_field] == broader_id]
        if broader_row.empty:
            return False
        
        next_broader = str(broader_row.iloc[0][broader_field]) if pd.notna(broader_row.iloc[0][broader_field]) else ''
        
        if next_broader and next_broader != '':
            return self._has_circular_reference(df, broader_id, next_broader, id_field, broader_field, visited)
        
        return False
    
    def generate_hierarchy_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate statistics about the hierarchy"""
        stats = {
            'total_concepts': len(df),
            'top_level_concepts': 0,
            'max_depth': 0,
            'concepts_with_children': 0,
            'orphaned_concepts': 0
        }
        
        if 'broader' not in df.columns:
            return stats
        
        # Count top-level concepts (no broader concept)
        stats['top_level_concepts'] = len(df[df['broader'].isna() | (df['broader'] == '')])
        
        # Count concepts with children
        if 'narrower' in df.columns:
            stats['concepts_with_children'] = len(df[df['narrower'].notna() & (df['narrower'] != '')])
        
        # Calculate max depth (simplified - could be more sophisticated)
        if 'level' in df.columns:
            stats['max_depth'] = df['level'].max() if df['level'].notna().any() else 0
        
        return stats
