"""
SKOS Mapping Properties Handler

Implements W3C SKOS mapping properties for cross-vocabulary alignments:
- skos:exactMatch, skos:closeMatch, skos:broadMatch, skos:narrowMatch, skos:relatedMatch
- Validation and integrity checking according to W3C SKOS Reference Section 10

Author: SkoHub TTL Generator
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MappingType(Enum):
    """SKOS mapping property types according to W3C SKOS Reference Section 10"""
    EXACT_MATCH = "exactMatch"
    CLOSE_MATCH = "closeMatch"
    BROAD_MATCH = "broadMatch"
    NARROW_MATCH = "narrowMatch"
    RELATED_MATCH = "relatedMatch"
    MAPPING_RELATION = "mappingRelation"

class MappingHandler:
    """
    Handles SKOS mapping properties for cross-vocabulary concept alignment.
    
    Implements W3C SKOS Reference Section 10 requirements:
    - Mapping properties are sub-properties of semantic relations
    - exactMatch is transitive and symmetric
    - Integrity conditions for mapping conflicts
    """
    
    def __init__(self):
        self.mapping_patterns = {
            'uri_fields': ['uri', 'concepturi', 'concept_uri', 'id', 'identifier', 'url'],
            'target_uri_fields': ['target_uri', 'target_concept', 'mapped_uri', 'equivalent_uri'],
            'mapping_type_fields': ['mapping_type', 'relation_type', 'match_type'],
            'confidence_fields': ['confidence', 'score', 'certainty', 'strength']
        }
        
        # W3C SKOS mapping property hierarchy
        self.mapping_hierarchy = {
            MappingType.MAPPING_RELATION: {
                'sub_properties': [
                    MappingType.EXACT_MATCH,
                    MappingType.CLOSE_MATCH,
                    MappingType.BROAD_MATCH,
                    MappingType.NARROW_MATCH,
                    MappingType.RELATED_MATCH
                ]
            },
            MappingType.EXACT_MATCH: {
                'super_property': MappingType.CLOSE_MATCH,
                'symmetric': True,
                'transitive': True
            },
            MappingType.CLOSE_MATCH: {
                'symmetric': True,
                'transitive': False
            },
            MappingType.BROAD_MATCH: {
                'inverse': MappingType.NARROW_MATCH,
                'super_property': 'skos:broader'
            },
            MappingType.NARROW_MATCH: {
                'inverse': MappingType.BROAD_MATCH,
                'super_property': 'skos:narrower'
            },
            MappingType.RELATED_MATCH: {
                'symmetric': True,
                'super_property': 'skos:related'
            }
        }
    
    def detect_mapping_fields(self, df: pd.DataFrame) -> Dict[str, Optional[str]]:
        """
        Detect mapping-related fields in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping field types to detected column names
        """
        detected_fields = {
            'source_uri': None,
            'target_uri': None,
            'mapping_type': None,
            'confidence': None
        }
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Detect source URI field
        for pattern in self.mapping_patterns['uri_fields']:
            if pattern in columns_lower:
                detected_fields['source_uri'] = df.columns[columns_lower.index(pattern)]
                break
        
        # Detect target URI field
        for pattern in self.mapping_patterns['target_uri_fields']:
            if pattern in columns_lower:
                detected_fields['target_uri'] = df.columns[columns_lower.index(pattern)]
                break
        
        # Detect mapping type field
        for pattern in self.mapping_patterns['mapping_type_fields']:
            if pattern in columns_lower:
                detected_fields['mapping_type'] = df.columns[columns_lower.index(pattern)]
                break
        
        # Detect confidence field
        for pattern in self.mapping_patterns['confidence_fields']:
            if pattern in columns_lower:
                detected_fields['confidence'] = df.columns[columns_lower.index(pattern)]
                break
        
        return detected_fields
    
    def validate_mapping_data(self, df: pd.DataFrame, field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate mapping data according to W3C SKOS integrity conditions.
        
        Args:
            df: DataFrame with mapping data
            field_mapping: Mapping of field types to column names
            
        Returns:
            Validation results with issues and statistics
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'conflicts': []
        }
        
        if not field_mapping.get('source_uri') or not field_mapping.get('target_uri'):
            validation_result['is_valid'] = False
            validation_result['issues'].append("Missing required URI fields for mapping")
            return validation_result
        
        source_col = field_mapping['source_uri']
        target_col = field_mapping['target_uri']
        type_col = field_mapping.get('mapping_type')
        
        # Check for missing URIs
        missing_source = df[source_col].isna().sum()
        missing_target = df[target_col].isna().sum()
        
        if missing_source > 0:
            validation_result['warnings'].append(f"{missing_source} rows with missing source URIs")
        if missing_target > 0:
            validation_result['warnings'].append(f"{missing_target} rows with missing target URIs")
        
        # Check for self-mappings (reflexive mappings)
        self_mappings = df[df[source_col] == df[target_col]]
        if len(self_mappings) > 0:
            validation_result['warnings'].append(f"{len(self_mappings)} self-mappings detected")
        
        # Check for mapping conflicts (W3C SKOS integrity condition S46)
        if type_col:
            conflicts = self._detect_mapping_conflicts(df, source_col, target_col, type_col)
            validation_result['conflicts'] = conflicts
            if conflicts:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"{len(conflicts)} mapping conflicts detected")
        
        # Generate statistics
        validation_result['statistics'] = {
            'total_mappings': len(df),
            'unique_source_concepts': df[source_col].nunique(),
            'unique_target_concepts': df[target_col].nunique(),
            'self_mappings': len(self_mappings),
            'missing_source_uris': missing_source,
            'missing_target_uris': missing_target
        }
        
        if type_col:
            validation_result['statistics']['mapping_types'] = df[type_col].value_counts().to_dict()
        
        return validation_result
    
    def _detect_mapping_conflicts(self, df: pd.DataFrame, source_col: str, 
                                target_col: str, type_col: str) -> List[Dict[str, Any]]:
        """
        Detect conflicts between mapping types according to W3C SKOS integrity conditions.
        
        W3C SKOS S46: skos:exactMatch is disjoint with skos:broadMatch and skos:relatedMatch
        """
        conflicts = []
        
        # Group by source-target pairs
        grouped = df.groupby([source_col, target_col])[type_col].apply(list).reset_index()
        
        for _, row in grouped.iterrows():
            source_uri = row[source_col]
            target_uri = row[target_col]
            mapping_types = row[type_col]
            
            # Check for exactMatch conflicts
            if 'exactMatch' in mapping_types:
                conflicting_types = []
                if 'broadMatch' in mapping_types:
                    conflicting_types.append('broadMatch')
                if 'narrowMatch' in mapping_types:
                    conflicting_types.append('narrowMatch')
                if 'relatedMatch' in mapping_types:
                    conflicting_types.append('relatedMatch')
                
                if conflicting_types:
                    conflicts.append({
                        'source_uri': source_uri,
                        'target_uri': target_uri,
                        'conflict_type': 'exactMatch_conflict',
                        'conflicting_types': conflicting_types,
                        'description': f"exactMatch conflicts with {', '.join(conflicting_types)}"
                    })
            
            # Check for hierarchical-associative conflicts
            hierarchical = ['broadMatch', 'narrowMatch']
            associative = ['relatedMatch']
            
            has_hierarchical = any(t in mapping_types for t in hierarchical)
            has_associative = any(t in mapping_types for t in associative)
            
            if has_hierarchical and has_associative:
                conflicts.append({
                    'source_uri': source_uri,
                    'target_uri': target_uri,
                    'conflict_type': 'hierarchical_associative_conflict',
                    'description': "Hierarchical and associative mappings are disjoint"
                })
        
        return conflicts
    
    def process_mapping_data(self, df: pd.DataFrame, field_mapping: Dict[str, str],
                           base_uri: str = "") -> pd.DataFrame:
        """
        Process mapping data and normalize according to SKOS standards.
        
        Args:
            df: Input DataFrame with mapping data
            field_mapping: Field mapping configuration
            base_uri: Base URI for concept scheme
            
        Returns:
            Processed DataFrame with normalized mappings
        """
        processed_df = df.copy()
        
        source_col = field_mapping['source_uri']
        target_col = field_mapping['target_uri']
        type_col = field_mapping.get('mapping_type')
        confidence_col = field_mapping.get('confidence')
        
        # Normalize URIs
        processed_df[source_col] = processed_df[source_col].apply(
            lambda x: self._normalize_uri(x, base_uri) if pd.notna(x) else x
        )
        processed_df[target_col] = processed_df[target_col].apply(
            lambda x: self._normalize_uri(x, base_uri) if pd.notna(x) else x
        )
        
        # Normalize mapping types
        if type_col:
            processed_df[type_col] = processed_df[type_col].apply(self._normalize_mapping_type)
        
        # Add confidence scores if missing
        if not confidence_col:
            processed_df['mapping_confidence'] = 1.0  # Default confidence
        
        # Remove invalid mappings
        processed_df = processed_df.dropna(subset=[source_col, target_col])
        
        return processed_df
    
    def _normalize_uri(self, uri: str, base_uri: str) -> str:
        """Normalize URI according to SKOS standards"""
        if not uri or pd.isna(uri):
            return uri
        
        uri = str(uri).strip()
        
        # If already absolute URI, return as-is
        if uri.startswith(('http://', 'https://', 'urn:')):
            return uri
        
        # If relative URI and base_uri provided, make absolute
        if base_uri and not uri.startswith('/'):
            return f"{base_uri.rstrip('/')}/{uri}"
        
        return uri
    
    def _normalize_mapping_type(self, mapping_type: str) -> str:
        """Normalize mapping type to SKOS standard"""
        if not mapping_type or pd.isna(mapping_type):
            return 'closeMatch'  # Default mapping type
        
        mapping_type = str(mapping_type).lower().strip()
        
        # Mapping variations to standard SKOS types
        type_mappings = {
            'exact': 'exactMatch',
            'equivalent': 'exactMatch',
            'same': 'exactMatch',
            'identical': 'exactMatch',
            'close': 'closeMatch',
            'similar': 'closeMatch',
            'related': 'relatedMatch',
            'associated': 'relatedMatch',
            'broader': 'broadMatch',
            'narrower': 'narrowMatch',
            'parent': 'broadMatch',
            'child': 'narrowMatch'
        }
        
        return type_mappings.get(mapping_type, mapping_type)
    
    def generate_mapping_ttl(self, df: pd.DataFrame, field_mapping: Dict[str, str],
                           concept_scheme_uri: str = "") -> str:
        """
        Generate TTL representation of SKOS mappings.
        
        Args:
            df: DataFrame with mapping data
            field_mapping: Field mapping configuration
            concept_scheme_uri: URI of the concept scheme
            
        Returns:
            TTL string with SKOS mappings
        """
        ttl_lines = []
        
        source_col = field_mapping['source_uri']
        target_col = field_mapping['target_uri']
        type_col = field_mapping.get('mapping_type', 'mapping_type')
        confidence_col = field_mapping.get('confidence')
        
        # Group by source URI to combine mappings
        grouped = df.groupby(source_col)
        
        for source_uri, group in grouped:
            if pd.isna(source_uri):
                continue
            
            ttl_lines.append(f"<{source_uri}> rdf:type skos:Concept ;")
            
            if concept_scheme_uri:
                ttl_lines.append(f"    skos:inScheme <{concept_scheme_uri}> ;")
            
            # Add mappings
            for _, row in group.iterrows():
                target_uri = row[target_col]
                mapping_type = row.get(type_col, 'closeMatch')
                
                if pd.notna(target_uri):
                    ttl_lines.append(f"    skos:{mapping_type} <{target_uri}> ;")
            
            # Remove last semicolon and add period
            if ttl_lines and ttl_lines[-1].endswith(' ;'):
                ttl_lines[-1] = ttl_lines[-1][:-2] + ' .'
            
            ttl_lines.append("")  # Empty line between concepts
        
        return "\n".join(ttl_lines)
    
    def get_mapping_statistics(self, df: pd.DataFrame, field_mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics for mapping data.
        
        Args:
            df: DataFrame with mapping data
            field_mapping: Field mapping configuration
            
        Returns:
            Dictionary with detailed mapping statistics
        """
        source_col = field_mapping['source_uri']
        target_col = field_mapping['target_uri']
        type_col = field_mapping.get('mapping_type')
        
        stats = {
            'total_mappings': len(df),
            'unique_source_concepts': df[source_col].nunique(),
            'unique_target_concepts': df[target_col].nunique(),
            'coverage': {
                'source_concepts_with_mappings': df[source_col].nunique(),
                'target_concepts_mapped': df[target_col].nunique()
            }
        }
        
        if type_col:
            stats['mapping_type_distribution'] = df[type_col].value_counts().to_dict()
            
            # Calculate mapping density by type
            for mapping_type in df[type_col].unique():
                type_subset = df[df[type_col] == mapping_type]
                stats[f'{mapping_type}_count'] = len(type_subset)
        
        # Bidirectional mapping analysis
        bidirectional_pairs = set()
        for _, row in df.iterrows():
            source = row[source_col]
            target = row[target_col]
            reverse_exists = ((df[source_col] == target) & (df[target_col] == source)).any()
            if reverse_exists:
                bidirectional_pairs.add(tuple(sorted([source, target])))
        
        stats['bidirectional_mappings'] = len(bidirectional_pairs)
        
        return stats
