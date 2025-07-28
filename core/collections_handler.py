"""
SKOS Collections Handler

Implements W3C SKOS Collection support according to Section 9:
- skos:Collection and skos:OrderedCollection
- skos:member and skos:memberList properties
- Collection validation and TTL generation

Author: SkoHub TTL Generator
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import logging
import uuid

logger = logging.getLogger(__name__)

class CollectionType(Enum):
    """SKOS collection types according to W3C SKOS Reference Section 9"""
    COLLECTION = "Collection"
    ORDERED_COLLECTION = "OrderedCollection"

class CollectionsHandler:
    """
    Handles SKOS Collections for grouping and organizing concepts.
    
    Implements W3C SKOS Reference Section 9 requirements:
    - skos:Collection for labeled groups of concepts
    - skos:OrderedCollection for ordered groups with memberList
    - Proper member relationships and validation
    """
    
    def __init__(self):
        self.collection_patterns = {
            'collection_fields': ['collection', 'group', 'category', 'theme', 'cluster'],
            'member_fields': ['member', 'concept', 'item', 'element'],
            'order_fields': ['order', 'position', 'sequence', 'rank', 'index'],
            'label_fields': ['label', 'title', 'name', 'description']
        }
    
    def detect_collection_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect collection structure in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with detected collection information
        """
        detection_result = {
            'has_collections': False,
            'collection_field': None,
            'member_field': None,
            'order_field': None,
            'label_field': None,
            'collection_type': CollectionType.COLLECTION,
            'collections_found': []
        }
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Detect collection grouping field
        for pattern in self.collection_patterns['collection_fields']:
            if pattern in columns_lower:
                detection_result['collection_field'] = df.columns[columns_lower.index(pattern)]
                detection_result['has_collections'] = True
                break
        
        # Detect member field (concept URI)
        for pattern in self.collection_patterns['member_fields']:
            if pattern in columns_lower:
                detection_result['member_field'] = df.columns[columns_lower.index(pattern)]
                break
        
        # Detect order field
        for pattern in self.collection_patterns['order_fields']:
            if pattern in columns_lower:
                detection_result['order_field'] = df.columns[columns_lower.index(pattern)]
                detection_result['collection_type'] = CollectionType.ORDERED_COLLECTION
                break
        
        # Detect label field
        for pattern in self.collection_patterns['label_fields']:
            if pattern in columns_lower:
                detection_result['label_field'] = df.columns[columns_lower.index(pattern)]
                break
        
        # Find unique collections
        if detection_result['collection_field']:
            collections = df[detection_result['collection_field']].dropna().unique()
            detection_result['collections_found'] = list(collections)
        
        return detection_result
    
    def create_collections_from_data(self, df: pd.DataFrame, 
                                   collection_config: Dict[str, Any],
                                   base_uri: str = "") -> Dict[str, Any]:
        """
        Create SKOS collections from dataset.
        
        Args:
            df: Input DataFrame
            collection_config: Collection configuration
            base_uri: Base URI for collections
            
        Returns:
            Dictionary with collection data and metadata
        """
        collections_data = {
            'collections': {},
            'statistics': {},
            'validation_issues': []
        }
        
        collection_field = collection_config.get('collection_field')
        member_field = collection_config.get('member_field', 'uri')
        order_field = collection_config.get('order_field')
        label_field = collection_config.get('label_field')
        collection_type = collection_config.get('collection_type', CollectionType.COLLECTION)
        
        if not collection_field:
            collections_data['validation_issues'].append("No collection field specified")
            return collections_data
        
        # Group by collection
        grouped = df.groupby(collection_field)
        
        for collection_name, group in grouped:
            if pd.isna(collection_name):
                continue
            
            collection_uri = self._generate_collection_uri(collection_name, base_uri)
            
            # Get members
            members = group[member_field].dropna().tolist()
            
            # Create collection data
            collection_data = {
                'uri': collection_uri,
                'type': collection_type.value,
                'label': collection_name,
                'members': members,
                'member_count': len(members)
            }
            
            # Add custom label if specified
            if label_field and label_field in group.columns:
                custom_labels = group[label_field].dropna().unique()
                if len(custom_labels) > 0:
                    collection_data['label'] = custom_labels[0]
            
            # Handle ordered collections
            if collection_type == CollectionType.ORDERED_COLLECTION and order_field:
                ordered_group = group.sort_values(order_field)
                collection_data['ordered_members'] = ordered_group[member_field].dropna().tolist()
                collection_data['member_list'] = collection_data['ordered_members']
            
            collections_data['collections'][collection_name] = collection_data
        
        # Generate statistics
        collections_data['statistics'] = self._generate_collection_statistics(collections_data['collections'])
        
        return collections_data
    
    def _generate_collection_uri(self, collection_name: str, base_uri: str) -> str:
        """Generate URI for collection"""
        if not base_uri:
            base_uri = "http://example.org/collections"
        
        # Clean collection name for URI
        clean_name = str(collection_name).replace(' ', '_').replace('/', '_')
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '_-')
        
        return f"{base_uri.rstrip('/')}/collection/{clean_name}"
    
    def validate_collections(self, collections_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate collections according to W3C SKOS standards.
        
        Args:
            collections_data: Collections data to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        collections = collections_data.get('collections', {})
        
        for collection_name, collection_data in collections.items():
            # Check for empty collections
            if collection_data['member_count'] == 0:
                validation_result['warnings'].append(
                    f"Collection '{collection_name}' has no members"
                )
            
            # Check for duplicate members
            members = collection_data.get('members', [])
            if len(members) != len(set(members)):
                validation_result['warnings'].append(
                    f"Collection '{collection_name}' has duplicate members"
                )
            
            # Validate ordered collections
            if collection_data['type'] == CollectionType.ORDERED_COLLECTION.value:
                ordered_members = collection_data.get('ordered_members', [])
                if not ordered_members:
                    validation_result['issues'].append(
                        f"Ordered collection '{collection_name}' has no member list"
                    )
                    validation_result['is_valid'] = False
        
        # Generate validation statistics
        validation_result['statistics'] = {
            'total_collections': len(collections),
            'ordered_collections': sum(1 for c in collections.values() 
                                     if c['type'] == CollectionType.ORDERED_COLLECTION.value),
            'unordered_collections': sum(1 for c in collections.values() 
                                       if c['type'] == CollectionType.COLLECTION.value),
            'total_members': sum(c['member_count'] for c in collections.values()),
            'average_collection_size': sum(c['member_count'] for c in collections.values()) / len(collections) if collections else 0
        }
        
        return validation_result
    
    def _generate_collection_statistics(self, collections: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistics for collections"""
        if not collections:
            return {}
        
        member_counts = [c['member_count'] for c in collections.values()]
        
        return {
            'total_collections': len(collections),
            'total_members': sum(member_counts),
            'average_members_per_collection': sum(member_counts) / len(collections),
            'largest_collection_size': max(member_counts),
            'smallest_collection_size': min(member_counts),
            'ordered_collections': sum(1 for c in collections.values() 
                                     if c['type'] == CollectionType.ORDERED_COLLECTION.value)
        }
    
    def generate_collections_ttl(self, collections_data: Dict[str, Any],
                               concept_scheme_uri: str = "") -> str:
        """
        Generate TTL representation of SKOS collections.
        
        Args:
            collections_data: Collections data
            concept_scheme_uri: URI of the concept scheme
            
        Returns:
            TTL string with SKOS collections
        """
        ttl_lines = []
        collections = collections_data.get('collections', {})
        
        for collection_name, collection_data in collections.items():
            collection_uri = collection_data['uri']
            collection_type = collection_data['type']
            label = collection_data['label']
            members = collection_data.get('members', [])
            
            # Collection definition
            ttl_lines.append(f"<{collection_uri}> rdf:type skos:{collection_type} ;")
            ttl_lines.append(f'    rdfs:label "{label}"@en ;')
            
            if concept_scheme_uri:
                ttl_lines.append(f"    skos:inScheme <{concept_scheme_uri}> ;")
            
            # Add members
            if members:
                if collection_type == CollectionType.ORDERED_COLLECTION.value:
                    # Use memberList for ordered collections
                    ordered_members = collection_data.get('member_list', members)
                    member_list = " ".join(f"<{member}>" for member in ordered_members if member)
                    ttl_lines.append(f"    skos:memberList ( {member_list} ) ;")
                
                # Always add skos:member properties (inferred for ordered collections)
                for member in members:
                    if member:  # Skip empty/None members
                        ttl_lines.append(f"    skos:member <{member}> ;")
            
            # Remove last semicolon and add period
            if ttl_lines and ttl_lines[-1].endswith(' ;'):
                ttl_lines[-1] = ttl_lines[-1][:-2] + ' .'
            
            ttl_lines.append("")  # Empty line between collections
        
        return "\n".join(ttl_lines)
    
    def suggest_collection_groupings(self, df: pd.DataFrame, 
                                   concept_uri_field: str,
                                   max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest potential collection groupings based on data patterns.
        
        Args:
            df: Input DataFrame
            concept_uri_field: Field containing concept URIs
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of collection suggestions
        """
        suggestions = []
        
        # Analyze categorical fields for potential groupings
        for col in df.columns:
            if col == concept_uri_field:
                continue
            
            # Skip numeric fields
            if df[col].dtype in ['int64', 'float64']:
                continue
            
            # Check if field has reasonable number of unique values for grouping
            unique_values = df[col].dropna().nunique()
            total_rows = len(df)
            
            if 2 <= unique_values <= min(50, total_rows // 2):
                # Calculate grouping quality
                value_counts = df[col].value_counts()
                avg_group_size = total_rows / unique_values
                size_variance = value_counts.var()
                
                suggestion = {
                    'field': col,
                    'unique_groups': unique_values,
                    'average_group_size': avg_group_size,
                    'size_variance': size_variance,
                    'quality_score': avg_group_size / (1 + size_variance / avg_group_size),
                    'sample_groups': value_counts.head(3).to_dict()
                }
                
                suggestions.append(suggestion)
        
        # Sort by quality score
        suggestions.sort(key=lambda x: x['quality_score'], reverse=True)
        
        return suggestions[:max_suggestions]
    
    def create_thematic_collections(self, df: pd.DataFrame,
                                  concept_uri_field: str,
                                  grouping_field: str,
                                  base_uri: str = "") -> Dict[str, Any]:
        """
        Create thematic collections based on a grouping field.
        
        Args:
            df: Input DataFrame
            concept_uri_field: Field containing concept URIs
            grouping_field: Field to group concepts by
            base_uri: Base URI for collections
            
        Returns:
            Collections data
        """
        collection_config = {
            'collection_field': grouping_field,
            'member_field': concept_uri_field,
            'collection_type': CollectionType.COLLECTION
        }
        
        return self.create_collections_from_data(df, collection_config, base_uri)
    
    def merge_collections(self, collections_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple collection datasets.
        
        Args:
            collections_list: List of collection data dictionaries
            
        Returns:
            Merged collections data
        """
        merged_data = {
            'collections': {},
            'statistics': {},
            'validation_issues': []
        }
        
        for collections_data in collections_list:
            collections = collections_data.get('collections', {})
            
            for collection_name, collection_data in collections.items():
                if collection_name in merged_data['collections']:
                    # Merge members
                    existing_members = set(merged_data['collections'][collection_name]['members'])
                    new_members = set(collection_data['members'])
                    merged_members = list(existing_members.union(new_members))
                    
                    merged_data['collections'][collection_name]['members'] = merged_members
                    merged_data['collections'][collection_name]['member_count'] = len(merged_members)
                else:
                    merged_data['collections'][collection_name] = collection_data.copy()
        
        # Regenerate statistics
        merged_data['statistics'] = self._generate_collection_statistics(
            merged_data['collections']
        )
        
        return merged_data
