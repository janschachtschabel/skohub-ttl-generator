"""
Data enrichment module for SkoHub TTL Generator.
Handles enriching existing TTL files with additional data from CSV/JSON sources.
"""

import pandas as pd
import re
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st
from urllib.parse import urlparse


class DataEnrichment:
    """Handles enriching existing TTL data with additional information."""
    
    def __init__(self):
        self.enrichment_strategies = {
            'merge_by_uri': self._merge_by_uri,
            'merge_by_label': self._merge_by_label,
            'merge_by_mapping': self._merge_by_mapping
        }
    
    def enrich_ttl_data(self, existing_ttl: str, enrichment_data: pd.DataFrame, 
                       strategy: str, config: Dict) -> Dict[str, Any]:
        """Enrich existing TTL with additional data."""
        
        if strategy not in self.enrichment_strategies:
            return {'error': f'Unknown enrichment strategy: {strategy}'}
        
        # Parse existing TTL to extract concepts
        existing_concepts = self._parse_ttl_concepts(existing_ttl)
        
        # Apply enrichment strategy
        enriched_data = self.enrichment_strategies[strategy](
            existing_concepts, enrichment_data, config
        )
        
        return enriched_data
    
    def _parse_ttl_concepts(self, ttl_content: str) -> Dict[str, Dict]:
        """Parse TTL content to extract existing concepts and their properties."""
        concepts = {}
        
        # Pattern to match concept blocks
        concept_pattern = r'<([^>]+)>\s+a\s+skos:Concept\s*;(.*?)(?=<[^>]+>\s+a\s+skos:Concept|$)'
        
        for match in re.finditer(concept_pattern, ttl_content, re.DOTALL):
            uri = match.group(1)
            properties_block = match.group(2)
            
            concept_data = {
                'uri': uri,
                'prefLabel': [],
                'altLabel': [],
                'definition': [],
                'broader': [],
                'narrower': [],
                'other_properties': {}
            }
            
            # Extract prefLabel
            pref_pattern = r'skos:prefLabel\s+"([^"]+)"(?:@(\w+))?'
            for pref_match in re.finditer(pref_pattern, properties_block):
                label = pref_match.group(1)
                lang = pref_match.group(2) or 'de'
                concept_data['prefLabel'].append({'value': label, 'lang': lang})
            
            # Extract altLabel
            alt_pattern = r'skos:altLabel\s+"([^"]+)"(?:@(\w+))?'
            for alt_match in re.finditer(alt_pattern, properties_block):
                label = alt_match.group(1)
                lang = alt_match.group(2) or 'de'
                concept_data['altLabel'].append({'value': label, 'lang': lang})
            
            # Extract definition
            def_pattern = r'skos:definition\s+"([^"]+)"(?:@(\w+))?'
            for def_match in re.finditer(def_pattern, properties_block):
                definition = def_match.group(1)
                lang = def_match.group(2) or 'de'
                concept_data['definition'].append({'value': definition, 'lang': lang})
            
            # Extract broader relationships
            broader_pattern = r'skos:broader\s+<([^>]+)>'
            for broader_match in re.finditer(broader_pattern, properties_block):
                concept_data['broader'].append(broader_match.group(1))
            
            # Extract narrower relationships
            narrower_pattern = r'skos:narrower\s+<([^>]+)>'
            for narrower_match in re.finditer(narrower_pattern, properties_block):
                concept_data['narrower'].append(narrower_match.group(1))
            
            concepts[uri] = concept_data
        
        return concepts
    
    def _merge_by_uri(self, existing_concepts: Dict, enrichment_data: pd.DataFrame, 
                     config: Dict) -> Dict[str, Any]:
        """Merge enrichment data by matching URIs."""
        uri_field = config.get('uri_field')
        base_uri = config.get('base_uri', '')
        
        if not uri_field or uri_field not in enrichment_data.columns:
            return {'error': f'URI field "{uri_field}" not found in enrichment data'}
        
        enriched_concepts = existing_concepts.copy()
        new_concepts = {}
        enrichment_stats = {
            'concepts_enriched': 0,
            'concepts_added': 0,
            'properties_added': 0
        }
        
        for _, row in enrichment_data.iterrows():
            row_uri = str(row[uri_field])
            
            # Normalize URI
            if not row_uri.startswith('http'):
                if base_uri:
                    row_uri = f"{base_uri.rstrip('/')}/{row_uri}"
                else:
                    continue  # Skip if no base URI provided
            
            # Check if concept exists
            if row_uri in enriched_concepts:
                # Enrich existing concept
                concept = enriched_concepts[row_uri]
                self._add_enrichment_properties(concept, row, config)
                enrichment_stats['concepts_enriched'] += 1
            else:
                # Create new concept
                new_concept = self._create_concept_from_row(row_uri, row, config)
                new_concepts[row_uri] = new_concept
                enrichment_stats['concepts_added'] += 1
        
        return {
            'enriched_concepts': enriched_concepts,
            'new_concepts': new_concepts,
            'stats': enrichment_stats
        }
    
    def _merge_by_label(self, existing_concepts: Dict, enrichment_data: pd.DataFrame, 
                       config: Dict) -> Dict[str, Any]:
        """Merge enrichment data by matching labels."""
        label_field = config.get('label_field')
        
        if not label_field or label_field not in enrichment_data.columns:
            return {'error': f'Label field "{label_field}" not found in enrichment data'}
        
        # Create label-to-URI mapping from existing concepts
        label_to_uri = {}
        for uri, concept in existing_concepts.items():
            for pref_label in concept.get('prefLabel', []):
                label_to_uri[pref_label['value'].lower()] = uri
            for alt_label in concept.get('altLabel', []):
                label_to_uri[alt_label['value'].lower()] = uri
        
        enriched_concepts = existing_concepts.copy()
        new_concepts = {}
        enrichment_stats = {
            'concepts_enriched': 0,
            'concepts_added': 0,
            'matches_found': 0,
            'no_matches': 0
        }
        
        for _, row in enrichment_data.iterrows():
            row_label = str(row[label_field]).lower()
            
            if row_label in label_to_uri:
                # Enrich existing concept
                uri = label_to_uri[row_label]
                concept = enriched_concepts[uri]
                self._add_enrichment_properties(concept, row, config)
                enrichment_stats['concepts_enriched'] += 1
                enrichment_stats['matches_found'] += 1
            else:
                # Could create new concept or skip
                enrichment_stats['no_matches'] += 1
        
        return {
            'enriched_concepts': enriched_concepts,
            'new_concepts': new_concepts,
            'stats': enrichment_stats
        }
    
    def _merge_by_mapping(self, existing_concepts: Dict, enrichment_data: pd.DataFrame, 
                         config: Dict) -> Dict[str, Any]:
        """Merge enrichment data using custom field mapping."""
        field_mapping = config.get('field_mapping', {})
        
        if not field_mapping:
            return {'error': 'No field mapping provided for merge strategy'}
        
        enriched_concepts = existing_concepts.copy()
        enrichment_stats = {
            'concepts_enriched': 0,
            'properties_added': 0
        }
        
        # Apply field mapping to all existing concepts
        for uri, concept in enriched_concepts.items():
            # Find matching row in enrichment data based on mapping
            matching_row = self._find_matching_row(concept, enrichment_data, field_mapping)
            
            if matching_row is not None:
                self._add_enrichment_properties(concept, matching_row, config)
                enrichment_stats['concepts_enriched'] += 1
        
        return {
            'enriched_concepts': enriched_concepts,
            'new_concepts': {},
            'stats': enrichment_stats
        }
    
    def _add_enrichment_properties(self, concept: Dict, row: pd.Series, config: Dict):
        """Add enrichment properties to an existing concept."""
        property_mapping = config.get('property_mapping', {})
        
        for column, value in row.items():
            if pd.isna(value) or str(value).strip() == '':
                continue
            
            # Map column to SKOS property
            skos_property = property_mapping.get(column, column)
            
            if skos_property in ['prefLabel', 'altLabel', 'definition']:
                # Handle language-tagged literals
                lang = config.get('language', 'de')
                if skos_property not in concept:
                    concept[skos_property] = []
                
                # Check if value already exists
                existing_values = [item['value'] for item in concept[skos_property]]
                if str(value) not in existing_values:
                    concept[skos_property].append({'value': str(value), 'lang': lang})
            
            elif skos_property in ['broader', 'narrower', 'related']:
                # Handle object properties
                if skos_property not in concept:
                    concept[skos_property] = []
                
                if str(value) not in concept[skos_property]:
                    concept[skos_property].append(str(value))
            
            else:
                # Handle custom properties
                if 'other_properties' not in concept:
                    concept['other_properties'] = {}
                concept['other_properties'][skos_property] = str(value)
    
    def _create_concept_from_row(self, uri: str, row: pd.Series, config: Dict) -> Dict:
        """Create a new concept from enrichment data row."""
        concept = {
            'uri': uri,
            'prefLabel': [],
            'altLabel': [],
            'definition': [],
            'broader': [],
            'narrower': [],
            'other_properties': {}
        }
        
        self._add_enrichment_properties(concept, row, config)
        return concept
    
    def _find_matching_row(self, concept: Dict, enrichment_data: pd.DataFrame, 
                          field_mapping: Dict) -> Optional[pd.Series]:
        """Find matching row in enrichment data based on field mapping."""
        # This is a simplified implementation
        # In practice, you might want more sophisticated matching logic
        
        for _, row in enrichment_data.iterrows():
            match_found = True
            
            for concept_field, data_field in field_mapping.items():
                if data_field not in row.index:
                    continue
                
                concept_values = []
                if concept_field == 'prefLabel':
                    concept_values = [item['value'] for item in concept.get('prefLabel', [])]
                elif concept_field == 'altLabel':
                    concept_values = [item['value'] for item in concept.get('altLabel', [])]
                elif concept_field == 'uri':
                    concept_values = [concept.get('uri', '')]
                
                if str(row[data_field]).lower() not in [v.lower() for v in concept_values]:
                    match_found = False
                    break
            
            if match_found:
                return row
        
        return None
    
    def generate_enriched_ttl(self, enriched_data: Dict, base_uri: str, 
                             namespace_prefix: str = "skills") -> str:
        """Generate TTL content from enriched data."""
        ttl_lines = []
        
        # Add prefixes
        ttl_lines.extend([
            f"@prefix {namespace_prefix}: <{base_uri}> .",
            "@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
            "@prefix dc: <http://purl.org/dc/terms/> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            ""
        ])
        
        # Add enriched concepts
        all_concepts = {}
        all_concepts.update(enriched_data.get('enriched_concepts', {}))
        all_concepts.update(enriched_data.get('new_concepts', {}))
        
        for uri, concept in all_concepts.items():
            ttl_lines.append(f"<{uri}> a skos:Concept ;")
            
            # Add prefLabel
            for pref_label in concept.get('prefLabel', []):
                ttl_lines.append(f'    skos:prefLabel "{pref_label["value"]}"@{pref_label["lang"]} ;')
            
            # Add altLabel
            for alt_label in concept.get('altLabel', []):
                ttl_lines.append(f'    skos:altLabel "{alt_label["value"]}"@{alt_label["lang"]} ;')
            
            # Add definition
            for definition in concept.get('definition', []):
                ttl_lines.append(f'    skos:definition "{definition["value"]}"@{definition["lang"]} ;')
            
            # Add broader relationships
            for broader in concept.get('broader', []):
                ttl_lines.append(f'    skos:broader <{broader}> ;')
            
            # Add narrower relationships
            for narrower in concept.get('narrower', []):
                ttl_lines.append(f'    skos:narrower <{narrower}> ;')
            
            # Add other properties
            for prop, value in concept.get('other_properties', {}).items():
                ttl_lines.append(f'    {prop} "{value}" ;')
            
            # Remove last semicolon and add period
            if ttl_lines[-1].endswith(' ;'):
                ttl_lines[-1] = ttl_lines[-1][:-2] + ' .'
            
            ttl_lines.append("")
        
        return '\n'.join(ttl_lines)
    
    def get_enrichment_config_ui(self, strategy: str, enrichment_data: pd.DataFrame) -> Dict:
        """Generate Streamlit UI for enrichment configuration."""
        config = {}
        
        if strategy == 'merge_by_uri':
            # URI field selection
            uri_options = [col for col in enrichment_data.columns 
                          if any(pattern in col.lower() for pattern in ['uri', 'id', 'identifier', 'code'])]
            
            config['uri_field'] = st.selectbox(
                "URI/ID Field in Enrichment Data",
                options=enrichment_data.columns.tolist(),
                index=0 if not uri_options else enrichment_data.columns.tolist().index(uri_options[0]),
                help="Field containing URIs or IDs to match with existing TTL concepts"
            )
            
            config['base_uri'] = st.text_input(
                "Base URI (if enrichment data has relative URIs)",
                value="http://w3id.org/openeduhub/vocabs/skills/",
                help="Base URI to prepend to relative URIs in enrichment data"
            )
        
        elif strategy == 'merge_by_label':
            # Label field selection
            label_options = [col for col in enrichment_data.columns 
                           if any(pattern in col.lower() for pattern in ['label', 'name', 'title', 'term'])]
            
            config['label_field'] = st.selectbox(
                "Label Field in Enrichment Data",
                options=enrichment_data.columns.tolist(),
                index=0 if not label_options else enrichment_data.columns.tolist().index(label_options[0]),
                help="Field containing labels to match with existing TTL concepts"
            )
        
        # Property mapping for all strategies
        st.subheader("Property Mapping")
        st.write("Map enrichment data fields to SKOS properties:")
        
        property_mapping = {}
        skos_properties = ['prefLabel', 'altLabel', 'definition', 'broader', 'narrower', 'related', 'note']
        
        for column in enrichment_data.columns:
            if column == config.get('uri_field') or column == config.get('label_field'):
                continue  # Skip fields used for matching
            
            mapped_property = st.selectbox(
                f"Map '{column}' to:",
                options=['skip'] + skos_properties + ['custom'],
                key=f"mapping_{column}",
                help=f"Choose SKOS property for field '{column}'"
            )
            
            if mapped_property == 'custom':
                custom_property = st.text_input(
                    f"Custom property for '{column}':",
                    key=f"custom_{column}",
                    help="Enter custom property name (e.g., 'dc:description')"
                )
                if custom_property:
                    property_mapping[column] = custom_property
            elif mapped_property != 'skip':
                property_mapping[column] = mapped_property
        
        config['property_mapping'] = property_mapping
        config['language'] = st.selectbox(
            "Language for text properties",
            options=['de', 'en', 'fr', 'es'],
            index=0,
            help="Language tag for text properties"
        )
        
        return config
