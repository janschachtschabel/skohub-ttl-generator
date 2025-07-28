"""
Enhanced SKOS Documentation Properties Handler

Implements W3C SKOS documentation properties according to Section 7:
- skos:note, skos:definition, skos:example, skos:scopeNote
- skos:changeNote, skos:editorialNote, skos:historyNote
- Multi-language support and validation

Author: SkoHub TTL Generator
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentationType(Enum):
    """SKOS documentation property types according to W3C SKOS Reference Section 7"""
    NOTE = "note"
    DEFINITION = "definition"
    EXAMPLE = "example"
    SCOPE_NOTE = "scopeNote"
    CHANGE_NOTE = "changeNote"
    EDITORIAL_NOTE = "editorialNote"
    HISTORY_NOTE = "historyNote"

class DocumentationHandler:
    """
    Handles SKOS documentation properties for concept annotation.
    
    Implements W3C SKOS Reference Section 7 requirements:
    - All documentation properties are sub-properties of skos:note
    - Support for multiple languages and formats
    - Validation and quality checking
    """
    
    def __init__(self):
        self.documentation_patterns = {
            'note_fields': ['note', 'notes', 'comment', 'comments', 'remark'],
            'definition_fields': ['definition', 'def', 'meaning', 'explanation'],
            'example_fields': ['example', 'examples', 'sample', 'instance'],
            'scope_fields': ['scope', 'scopenote', 'scope_note', 'application', 'usage'],
            'change_fields': ['change', 'changenote', 'change_note', 'modification', 'update'],
            'editorial_fields': ['editorial', 'editorialnote', 'editorial_note', 'editor_note'],
            'history_fields': ['history', 'historynote', 'history_note', 'background']
        }
        
        # Language detection patterns
        self.language_patterns = {
            'de': ['deutsch', 'german', 'de', 'ger'],
            'en': ['english', 'eng', 'en'],
            'fr': ['french', 'français', 'fr', 'fra'],
            'es': ['spanish', 'español', 'es', 'spa'],
            'it': ['italian', 'italiano', 'it', 'ita']
        }
    
    def detect_documentation_fields(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect documentation-related fields in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping documentation types to detected column names
        """
        detected_fields = {doc_type.value: [] for doc_type in DocumentationType}
        
        columns_lower = [col.lower() for col in df.columns]
        
        # Check each documentation type
        for doc_type in DocumentationType:
            patterns = self.documentation_patterns.get(f'{doc_type.value}_fields', [])
            
            for pattern in patterns:
                matching_cols = [
                    df.columns[i] for i, col in enumerate(columns_lower)
                    if pattern in col
                ]
                detected_fields[doc_type.value].extend(matching_cols)
        
        # Remove duplicates while preserving order
        for doc_type in detected_fields:
            detected_fields[doc_type] = list(dict.fromkeys(detected_fields[doc_type]))
        
        return detected_fields
    
    def detect_language_variants(self, df: pd.DataFrame, field_name: str) -> Dict[str, str]:
        """
        Detect language variants of a documentation field.
        
        Args:
            df: Input DataFrame
            field_name: Base field name
            
        Returns:
            Dictionary mapping language codes to column names
        """
        language_variants = {}
        
        # Look for language suffixes in column names
        for col in df.columns:
            col_lower = col.lower()
            base_field = field_name.lower()
            
            if base_field in col_lower:
                # Check for language indicators
                for lang_code, indicators in self.language_patterns.items():
                    for indicator in indicators:
                        if indicator in col_lower and indicator != base_field:
                            language_variants[lang_code] = col
                            break
                
                # If no specific language found, assume default
                if col not in language_variants.values():
                    if col_lower == base_field or col_lower == f"{base_field}_en":
                        language_variants['en'] = col
        
        return language_variants
    
    def process_documentation_data(self, df: pd.DataFrame, 
                                 field_mapping: Dict[str, Any],
                                 concept_uri_field: str) -> Dict[str, Any]:
        """
        Process documentation data and organize by concept.
        
        Args:
            df: Input DataFrame
            field_mapping: Mapping of documentation types to fields
            concept_uri_field: Field containing concept URIs
            
        Returns:
            Processed documentation data organized by concept
        """
        documentation_data = {}
        
        for _, row in df.iterrows():
            concept_uri = row.get(concept_uri_field)
            if pd.isna(concept_uri):
                continue
            
            if concept_uri not in documentation_data:
                documentation_data[concept_uri] = {}
            
            # Process each documentation type
            for doc_type, fields in field_mapping.items():
                if not fields:
                    continue
                
                doc_entries = []
                
                if isinstance(fields, dict):
                    # Multi-language fields
                    for lang_code, field_name in fields.items():
                        if field_name in row and pd.notna(row[field_name]):
                            content = str(row[field_name]).strip()
                            if content:
                                doc_entries.append({
                                    'content': content,
                                    'language': lang_code,
                                    'type': doc_type
                                })
                elif isinstance(fields, list):
                    # Multiple fields of same type
                    for field_name in fields:
                        if field_name in row and pd.notna(row[field_name]):
                            content = str(row[field_name]).strip()
                            if content:
                                doc_entries.append({
                                    'content': content,
                                    'language': 'en',  # Default language
                                    'type': doc_type
                                })
                else:
                    # Single field
                    if fields in row and pd.notna(row[fields]):
                        content = str(row[fields]).strip()
                        if content:
                            doc_entries.append({
                                'content': content,
                                'language': 'en',  # Default language
                                'type': doc_type
                            })
                
                if doc_entries:
                    if doc_type not in documentation_data[concept_uri]:
                        documentation_data[concept_uri][doc_type] = []
                    documentation_data[concept_uri][doc_type].extend(doc_entries)
        
        return documentation_data
    
    def validate_documentation(self, documentation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate documentation according to SKOS standards and best practices.
        
        Args:
            documentation_data: Documentation data to validate
            
        Returns:
            Validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'quality_metrics': {}
        }
        
        total_concepts = len(documentation_data)
        concepts_with_docs = 0
        doc_type_counts = {doc_type.value: 0 for doc_type in DocumentationType}
        language_distribution = {}
        
        for concept_uri, concept_docs in documentation_data.items():
            if concept_docs:
                concepts_with_docs += 1
            
            # Check for documentation quality issues
            for doc_type, entries in concept_docs.items():
                doc_type_counts[doc_type] += len(entries)
                
                for entry in entries:
                    content = entry['content']
                    language = entry['language']
                    
                    # Track language distribution
                    if language not in language_distribution:
                        language_distribution[language] = 0
                    language_distribution[language] += 1
                    
                    # Quality checks
                    if len(content) < 10:
                        validation_result['warnings'].append(
                            f"Very short {doc_type} for concept {concept_uri}: '{content[:50]}...'"
                        )
                    
                    if len(content) > 1000:
                        validation_result['warnings'].append(
                            f"Very long {doc_type} for concept {concept_uri} ({len(content)} chars)"
                        )
                    
                    # Check for potential HTML/markup
                    if re.search(r'<[^>]+>', content):
                        validation_result['warnings'].append(
                            f"Potential HTML markup in {doc_type} for concept {concept_uri}"
                        )
        
        # Generate statistics
        validation_result['statistics'] = {
            'total_concepts': total_concepts,
            'concepts_with_documentation': concepts_with_docs,
            'documentation_coverage': concepts_with_docs / total_concepts if total_concepts > 0 else 0,
            'documentation_type_counts': doc_type_counts,
            'language_distribution': language_distribution,
            'total_documentation_entries': sum(doc_type_counts.values())
        }
        
        # Quality metrics
        validation_result['quality_metrics'] = {
            'average_docs_per_concept': sum(doc_type_counts.values()) / total_concepts if total_concepts > 0 else 0,
            'concepts_with_definitions': doc_type_counts.get('definition', 0),
            'concepts_with_examples': doc_type_counts.get('example', 0),
            'multilingual_coverage': len(language_distribution)
        }
        
        return validation_result
    
    def generate_documentation_ttl(self, documentation_data: Dict[str, Any],
                                 include_timestamps: bool = False) -> str:
        """
        Generate TTL representation of SKOS documentation properties.
        
        Args:
            documentation_data: Documentation data
            include_timestamps: Whether to include timestamps for change notes
            
        Returns:
            TTL string with SKOS documentation properties
        """
        ttl_lines = []
        
        for concept_uri, concept_docs in documentation_data.items():
            if not concept_docs:
                continue
            
            ttl_lines.append(f"<{concept_uri}> rdf:type skos:Concept ;")
            
            # Add documentation properties
            for doc_type, entries in concept_docs.items():
                for entry in entries:
                    content = entry['content']
                    language = entry['language']
                    
                    # Escape quotes and special characters
                    escaped_content = self._escape_ttl_literal(content)
                    
                    # Add language tag
                    if language and language != 'none':
                        ttl_lines.append(f'    skos:{doc_type} """{escaped_content}"""@{language} ;')
                    else:
                        ttl_lines.append(f'    skos:{doc_type} """{escaped_content}""" ;')
            
            # Add timestamp for change notes if requested
            if include_timestamps and 'changeNote' in concept_docs:
                timestamp = datetime.now().isoformat()
                ttl_lines.append(f'    dct:modified "{timestamp}"^^xsd:dateTime ;')
            
            # Remove last semicolon and add period
            if ttl_lines and ttl_lines[-1].endswith(' ;'):
                ttl_lines[-1] = ttl_lines[-1][:-2] + ' .'
            
            ttl_lines.append("")  # Empty line between concepts
        
        return "\n".join(ttl_lines)
    
    def _escape_ttl_literal(self, text: str) -> str:
        """Escape special characters for TTL literals"""
        if not text:
            return ""
        
        # Replace problematic characters
        text = text.replace('\\', '\\\\')  # Escape backslashes
        text = text.replace('"', '\\"')    # Escape quotes
        text = text.replace('\n', '\\n')   # Escape newlines
        text = text.replace('\r', '\\r')   # Escape carriage returns
        text = text.replace('\t', '\\t')   # Escape tabs
        
        return text
    
    def suggest_documentation_improvements(self, documentation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Suggest improvements for documentation quality.
        
        Args:
            documentation_data: Documentation data to analyze
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        concepts_without_definitions = []
        concepts_without_examples = []
        concepts_with_short_docs = []
        
        for concept_uri, concept_docs in documentation_data.items():
            has_definition = 'definition' in concept_docs and concept_docs['definition']
            has_example = 'example' in concept_docs and concept_docs['example']
            
            if not has_definition:
                concepts_without_definitions.append(concept_uri)
            
            if not has_example:
                concepts_without_examples.append(concept_uri)
            
            # Check for short documentation
            for doc_type, entries in concept_docs.items():
                for entry in entries:
                    if len(entry['content']) < 20:
                        concepts_with_short_docs.append({
                            'concept': concept_uri,
                            'doc_type': doc_type,
                            'content': entry['content']
                        })
        
        # Generate suggestions
        if concepts_without_definitions:
            suggestions.append({
                'type': 'missing_definitions',
                'priority': 'high',
                'description': f"{len(concepts_without_definitions)} concepts lack definitions",
                'concepts': concepts_without_definitions[:10],  # Show first 10
                'action': 'Add skos:definition properties for better concept understanding'
            })
        
        if concepts_without_examples:
            suggestions.append({
                'type': 'missing_examples',
                'priority': 'medium',
                'description': f"{len(concepts_without_examples)} concepts lack examples",
                'concepts': concepts_without_examples[:10],
                'action': 'Add skos:example properties to illustrate concept usage'
            })
        
        if concepts_with_short_docs:
            suggestions.append({
                'type': 'short_documentation',
                'priority': 'low',
                'description': f"{len(concepts_with_short_docs)} documentation entries are very short",
                'examples': concepts_with_short_docs[:5],
                'action': 'Expand documentation entries for better clarity'
            })
        
        return suggestions
    
    def merge_documentation(self, doc_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple documentation datasets.
        
        Args:
            doc_data_list: List of documentation data dictionaries
            
        Returns:
            Merged documentation data
        """
        merged_data = {}
        
        for doc_data in doc_data_list:
            for concept_uri, concept_docs in doc_data.items():
                if concept_uri not in merged_data:
                    merged_data[concept_uri] = {}
                
                for doc_type, entries in concept_docs.items():
                    if doc_type not in merged_data[concept_uri]:
                        merged_data[concept_uri][doc_type] = []
                    
                    # Avoid duplicates
                    existing_contents = {entry['content'] for entry in merged_data[concept_uri][doc_type]}
                    
                    for entry in entries:
                        if entry['content'] not in existing_contents:
                            merged_data[concept_uri][doc_type].append(entry)
                            existing_contents.add(entry['content'])
        
        return merged_data
