"""
TTL Generation module for SkoHub TTL Generator.
Handles SKOS-compliant TTL generation with full W3C SKOS Reference support.
"""

import pandas as pd
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import re


class TTLGenerator:
    """Generates SKOS-compliant TTL content from structured data."""
    
    def __init__(self):
        self.standard_prefixes = {
            'skos': 'http://www.w3.org/2004/02/skos/core#',
            'dct': 'http://purl.org/dc/terms/',
            'xsd': 'http://www.w3.org/2001/XMLSchema#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }
        
        self.esco_prefixes = {
            'esco': 'http://data.europa.eu/esco/skill/',
            'isced': 'http://data.europa.eu/esco/isced-f/',
            'isothes': 'http://purl.org/iso25964/skos-thes#'
        }
        
        # Supported SKOS properties based on W3C SKOS Reference
        self.skos_properties = {
            # Lexical Labels (Section 5)
            'prefLabel': 'skos:prefLabel',
            'altLabel': 'skos:altLabel', 
            'hiddenLabel': 'skos:hiddenLabel',
            
            # Notations (Section 6)
            'notation': 'skos:notation',
            
            # Documentation Properties (Section 7)
            'note': 'skos:note',
            'changeNote': 'skos:changeNote',
            'definition': 'skos:definition',
            'editorialNote': 'skos:editorialNote',
            'example': 'skos:example',
            'historyNote': 'skos:historyNote',
            'scopeNote': 'skos:scopeNote',
            
            # Semantic Relations (Section 8)
            'broader': 'skos:broader',
            'narrower': 'skos:narrower',
            'related': 'skos:related',
            'broaderTransitive': 'skos:broaderTransitive',
            'narrowerTransitive': 'skos:narrowerTransitive',
            'semanticRelation': 'skos:semanticRelation',
            
            # Mapping Properties (Section 10)
            'broadMatch': 'skos:broadMatch',
            'closeMatch': 'skos:closeMatch',
            'exactMatch': 'skos:exactMatch',
            'mappingRelation': 'skos:mappingRelation',
            'narrowMatch': 'skos:narrowMatch',
            'relatedMatch': 'skos:relatedMatch'
        }
    
    def generate_ttl(self, df: pd.DataFrame, config: Dict) -> str:
        """Generate complete TTL content from DataFrame and configuration"""
        base_uri = config['base_uri']
        vocab_title = config['vocab_title']
        vocab_description = config['vocab_description']
        field_mapping = config['field_mapping']
        vocab_name = config.get('vocab_name', 'vocabulary')
        language = config.get('language', 'en')
        
        # Construct full base URI
        full_base_uri = self._construct_base_uri(base_uri, vocab_name)
        
        # Get all prefixes
        all_prefixes = self._get_all_prefixes(config.get('additional_prefixes', {}))
        
        # Generate TTL content
        ttl_content = self._generate_header(full_base_uri, all_prefixes)
        ttl_content += self._generate_concept_scheme(vocab_title, vocab_description, language)
        ttl_content += self._generate_concepts(df, field_mapping, full_base_uri, language)
        
        return ttl_content
    
    def _construct_base_uri(self, base_uri: str, vocab_name: str) -> str:
        """Construct full base URI with vocabulary name"""
        if not base_uri.endswith('/'):
            base_uri += '/'
        return f"{base_uri}{vocab_name}/"
    
    def _get_all_prefixes(self, additional_prefixes: Dict[str, str]) -> Dict[str, str]:
        """Combine all prefixes"""
        return {**self.standard_prefixes, **self.esco_prefixes, **additional_prefixes}
    
    def _generate_header(self, base_uri: str, prefixes: Dict[str, str]) -> str:
        """Generate TTL header with base URI and prefixes"""
        content = f"@base <{base_uri}> .\n"
        
        # Add prefix declarations
        for prefix, namespace in prefixes.items():
            content += f"@prefix {prefix}: <{namespace}> .\n"
        
        content += "\n"
        return content
    
    def _generate_concept_scheme(self, title: str, description: str, language: str) -> str:
        """Generate ConceptScheme definition"""
        return f"""<> a skos:ConceptScheme ;
    dct:title "{self._escape_literal(title)}"@{language} ;
    dct:description "{self._escape_literal(description)}"@{language} ;
    dct:created "{datetime.now().strftime('%Y-%m-%d')}"^^xsd:date .

"""
    
    def _generate_concepts(self, df: pd.DataFrame, field_mapping: Dict, base_uri: str, language: str) -> str:
        """Generate all concept definitions"""
        content = ""
        
        for _, row in df.iterrows():
            concept_uri = self._get_concept_uri(row, field_mapping, base_uri)
            content += self._generate_single_concept(row, field_mapping, concept_uri, language)
        
        return content
    
    def _generate_single_concept(self, row: pd.Series, field_mapping: Dict, concept_uri: str, language: str) -> str:
        """Generate a single concept definition"""
        content = f"<{concept_uri}> a skos:Concept"
        
        # Add concept scheme membership
        content += " ;\n    skos:inScheme <>"
        
        # Process all mapped fields (skip 'uri' field as it's already used for the concept URI)
        for skos_prop, field_name in field_mapping.items():
            if skos_prop == 'uri':  # Skip URI field as it's already used for concept URI
                continue
            if field_name and pd.notna(row.get(field_name)):
                value = row[field_name]
                property_ttl = self._generate_property(skos_prop, value, language)
                if property_ttl:
                    content += " ;\n" + property_ttl
        
        content += " .\n\n"
        return content
    
    def _generate_property(self, skos_prop: str, value: Any, language: str) -> str:
        """Generate TTL for a single property"""
        if skos_prop not in self.skos_properties:
            return ""
        
        ttl_prop = self.skos_properties[skos_prop]
        
        # Handle different value types
        if isinstance(value, str):
            if '|' in value:
                # Multiple values separated by pipe
                values = [v.strip() for v in value.split('|') if v.strip()]
                return self._generate_multiple_values(ttl_prop, values, skos_prop, language)
            else:
                return self._generate_single_value(ttl_prop, value, skos_prop, language)
        elif isinstance(value, (list, tuple)):
            return self._generate_multiple_values(ttl_prop, value, skos_prop, language)
        else:
            return self._generate_single_value(ttl_prop, str(value), skos_prop, language)
    
    def _generate_single_value(self, ttl_prop: str, value: str, skos_prop: str, language: str) -> str:
        """Generate TTL for a single property value"""
        value = value.strip()
        if not value:
            return ""
        
        if self._is_uri_property(skos_prop):
            # URI reference
            if value.startswith('http'):
                return f"    {ttl_prop} <{value}>"
            else:
                return f"    {ttl_prop} <{value}>"
        elif self._is_notation_property(skos_prop):
            # Typed literal for notations
            return f'    {ttl_prop} "{self._escape_literal(value)}"^^xsd:string'
        else:
            # Language-tagged literal
            return f'    {ttl_prop} "{self._escape_literal(value)}"@{language}'
    
    def _generate_multiple_values(self, ttl_prop: str, values: List[str], skos_prop: str, language: str) -> str:
        """Generate TTL for multiple property values"""
        if not values:
            return ""
        
        content_parts = []
        for value in values:
            if isinstance(value, str) and value.strip():
                single_value = self._generate_single_value(ttl_prop, value.strip(), skos_prop, language)
                if single_value:
                    content_parts.append(single_value)
        
        return " ;\n".join(content_parts)
    
    def _is_uri_property(self, skos_prop: str) -> bool:
        """Check if property expects URI values"""
        uri_properties = ['broader', 'narrower', 'related', 'broaderTransitive', 'narrowerTransitive',
                         'semanticRelation', 'broadMatch', 'closeMatch', 'exactMatch', 'mappingRelation',
                         'narrowMatch', 'relatedMatch']
        return skos_prop in uri_properties
    
    def _is_notation_property(self, skos_prop: str) -> bool:
        """Check if property is a notation property"""
        return skos_prop == 'notation'
    
    def _get_concept_uri(self, row: pd.Series, field_mapping: Dict, base_uri: str) -> str:
        """Generate or extract concept URI"""
        # Check if URI field is mapped and has value
        uri_field = field_mapping.get('uri')
        if uri_field and pd.notna(row.get(uri_field)):
            uri_value = str(row[uri_field]).strip()
            if uri_value:
                # If it's already a full URI, use it
                if uri_value.startswith('http'):
                    return uri_value
                # Otherwise, make it relative to base URI
                if not base_uri.endswith('/'):
                    base_uri += '/'
                return f"{base_uri}{uri_value}"
        
        # Generate UUID-based URI as fallback
        if not base_uri.endswith('/'):
            base_uri += '/'
        return f"{base_uri}{str(uuid.uuid4())}"
    
    def _escape_literal(self, text: str) -> str:
        """Escape special characters in TTL literals"""
        if not isinstance(text, str):
            text = str(text)
        
        # Escape quotes and backslashes
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        return text
    
    def generate_hierarchical_ttl(self, df: pd.DataFrame, config: Dict, hierarchy_config: Dict) -> str:
        """Generate TTL with hierarchical relationships"""
        # First generate basic TTL
        ttl_content = self.generate_ttl(df, config)
        
        # Add hierarchical relationships
        hierarchy_ttl = self._generate_hierarchy_relationships(df, config, hierarchy_config)
        
        return ttl_content + hierarchy_ttl
    
    def _generate_hierarchy_relationships(self, df: pd.DataFrame, config: Dict, hierarchy_config: Dict) -> str:
        """Generate hierarchical relationships based on level information"""
        content = "# Hierarchical Relationships\n\n"
        
        level_field = hierarchy_config.get('level_field')
        if not level_field or level_field not in df.columns:
            return content
        
        field_mapping = config['field_mapping']
        base_uri = self._construct_base_uri(config['base_uri'], config.get('vocab_name', 'vocabulary'))
        
        # Group by levels
        df_sorted = df.sort_values(level_field)
        
        for _, row in df_sorted.iterrows():
            current_level = row[level_field]
            concept_uri = self._get_concept_uri(row, field_mapping, base_uri)
            
            # Find parent (previous level)
            if isinstance(current_level, str) and '.' in current_level:
                # Handle dotted notation like "1.1.1"
                level_parts = current_level.split('.')
                if len(level_parts) > 1:
                    parent_level = '.'.join(level_parts[:-1])
                    parent_row = df[df[level_field] == parent_level]
                    if not parent_row.empty:
                        parent_uri = self._get_concept_uri(parent_row.iloc[0], field_mapping, base_uri)
                        content += f"<{concept_uri}> skos:broader <{parent_uri}> .\n"
            elif isinstance(current_level, (int, float)):
                # Handle numeric levels
                parent_level = int(current_level) - 1
                if parent_level > 0:
                    parent_row = df[df[level_field] == parent_level]
                    if not parent_row.empty:
                        parent_uri = self._get_concept_uri(parent_row.iloc[0], field_mapping, base_uri)
                        content += f"<{concept_uri}> skos:broader <{parent_uri}> .\n"
        
        return content + "\n"
