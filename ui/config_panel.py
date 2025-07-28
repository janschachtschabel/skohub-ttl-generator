"""
Configuration Panel UI module for SkoHub TTL Generator.
Handles all configuration settings including AI, OpenAI, and vocabulary settings.
"""

import streamlit as st
import os
from typing import Dict, Any


class ConfigPanel:
    """Handles configuration UI for the SkoHub TTL Generator."""
    
    def __init__(self, ai_assistant):
        self.ai_assistant = ai_assistant
        self.base_uri_suggestions = [
            "http://w3id.org/openeduhub/vocabs/",
            "http://w3id.org/openeduhub/vocabs/skills/",
            "http://w3id.org/openeduhub/vocabs/occupations/",
            "http://w3id.org/openeduhub/vocabs/competencies/",
            "http://w3id.org/openeduhub/vocabs/topics/",
            "http://example.org/vocabs/",
            "https://vocab.example.org/"
        ]
    
    def render_ai_config_sidebar(self):
        """Render AI configuration section for sidebar"""
        st.markdown("### 🤖 AI Configuration")
        
        config = {}
        
        # Local AI Configuration (MiniLM only)
        config.update(self._render_local_ai_config())
        
        return config
    
    def render_vocab_config_section(self):
        """Render vocabulary configuration section for main area"""
        st.markdown("### 📚 Vocabulary Configuration")
        
        config = {}
        
        # Base URI Configuration
        config.update(self._render_base_uri_config())
        
        # Vocabulary Metadata
        config.update(self._render_vocab_metadata_config())
        
        # Additional Prefixes
        config.update(self._render_prefixes_config())
        
        # Export Settings
        config.update(self._render_export_config())
        
        return config
    
    def _render_local_ai_config(self) -> Dict[str, Any]:
        """Render local AI configuration section"""
        st.subheader("🤖 Local AI Configuration")
        
        config = {}
        
        if self.ai_assistant.is_local_ai_available():
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
            
            config['similarity_threshold'] = similarity_threshold
        else:
            st.error("❌ Install `sentence-transformers` package for AI suggestions")
            st.code("pip install sentence-transformers torch")
            config['similarity_threshold'] = 0.3
        
        return config
    

    
    def _render_base_uri_config(self) -> Dict[str, Any]:
        """Render base URI configuration section"""
        st.subheader("🌐 Base URI")
        
        # Base URI input with suggestions
        base_uri = st.selectbox(
            "Base URI",
            options=self.base_uri_suggestions,
            index=0,
            help="Base URI for the vocabulary namespace",
            key="config_base_uri_select"
        )
        
        # Custom base URI option
        if st.checkbox("Use custom Base URI"):
            base_uri = st.text_input(
                "Custom Base URI",
                value=base_uri,
                help="Enter a custom base URI"
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
        
        return {
            'base_uri': base_uri,
            'vocab_name': vocab_name,
            'full_base_uri': preview_base
        }
    
    def _render_vocab_metadata_config(self) -> Dict[str, Any]:
        """Render vocabulary metadata configuration section"""
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
        
        # Language selection
        language = st.selectbox(
            "Primary Language",
            options=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'sv', 'da', 'no'],
            index=1,  # Default to German
            help="Primary language for labels and descriptions",
            key="config_language_select"
        )
        
        # Creator/Publisher
        creator = st.text_input(
            "Creator/Publisher",
            help="Person or organization creating this vocabulary"
        )
        
        return {
            'vocab_title': vocab_title,
            'vocab_description': vocab_description,
            'language': language,
            'creator': creator
        }
    
    def _render_prefixes_config(self) -> Dict[str, Any]:
        """Render additional prefixes configuration section"""
        st.subheader("🏷️ Additional Prefixes")
        
        # Show standard prefixes
        with st.expander("📋 Standard Prefixes (included)"):
            st.write("• `skos:` → http://www.w3.org/2004/02/skos/core#")
            st.write("• `dct:` → http://purl.org/dc/terms/")
            st.write("• `xsd:` → http://www.w3.org/2001/XMLSchema#")
            st.write("• `rdf:` → http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            st.write("• `rdfs:` → http://www.w3.org/2000/01/rdf-schema#")
            st.write("• `esco:` → http://data.europa.eu/esco/skill/")
            st.write("• `isced:` → http://data.europa.eu/esco/isced-f/")
        
        # Custom prefixes
        additional_prefixes = {}
        
        if st.checkbox("Add custom prefixes"):
            num_prefixes = st.number_input("Number of custom prefixes", min_value=1, max_value=10, value=1)
            
            for i in range(num_prefixes):
                col1, col2 = st.columns(2)
                with col1:
                    prefix = st.text_input(f"Prefix {i+1}", key=f"prefix_{i}")
                with col2:
                    namespace = st.text_input(f"Namespace {i+1}", key=f"namespace_{i}")
                
                if prefix and namespace:
                    additional_prefixes[prefix] = namespace
        
        return {'additional_prefixes': additional_prefixes}
    
    def _render_export_config(self) -> Dict[str, Any]:
        """Render export configuration section"""
        st.subheader("📤 Export Settings")
        
        # File format options
        export_format = st.selectbox(
            "Export Format",
            options=["TTL (Turtle)", "RDF/XML", "N-Triples", "JSON-LD"],
            index=0,
            help="Choose the output format for the vocabulary",
            key="config_export_format_select"
        )
        
        # Validation options
        validate_skos = st.checkbox(
            "Validate SKOS compliance",
            value=True,
            help="Check generated TTL for SKOS compliance"
        )
        
        # Include statistics
        include_stats = st.checkbox(
            "Include generation statistics",
            value=True,
            help="Include comments with generation statistics"
        )
        
        return {
            'export_format': export_format,
            'validate_skos': validate_skos,
            'include_stats': include_stats
        }
    
    def render_field_mapping_ui(self, df, ai_suggestions: list = None, available_fields_dict: Dict[str, list] = None) -> Dict[str, str]:
        """Render field mapping interface with multi-source support"""
        st.subheader("🔗 Field Mapping")
        
        if df is None or df.empty:
            st.warning("No data available for field mapping")
            return {}
        
        # Show AI suggestions if available
        if ai_suggestions:
            with st.expander("🤖 AI Suggestions (MiniLM)", expanded=True):
                st.info("🌍 Using all-MiniLM-L12-v2 for intelligent field matching")
                for suggestion in ai_suggestions:
                    st.write(f"• {suggestion}")
        
        # Prepare available fields from all sources
        if available_fields_dict and len(available_fields_dict) > 1:
            # Multiple sources available
            st.info(f"📁 {len(available_fields_dict)} file sources detected. You can map fields from any source.")
            
            # Join Key Selection for merging multiple sources
            st.markdown("#### 🔗 Join Key Configuration")
            st.warning("⚠️ **Important**: Select matching fields to properly merge data from different sources!")
            
            join_keys = {}
            for source, fields in available_fields_dict.items():
                join_key = st.selectbox(
                    f"Join Key for {source}",
                    options=[''] + fields,
                    help=f"Select the field in {source} that contains matching identifiers (e.g., URI, ID, Code)",
                    key=f"join_key_{source.replace(' ', '_').replace('(', '').replace(')', '')}"
                )
                if join_key:
                    join_keys[source] = join_key
            
            # Store join keys in session state for later use
            st.session_state.join_keys = join_keys
            
            if len(join_keys) >= 2:
                st.success(f"✅ Join keys configured: {', '.join([f'{s}: {k}' for s, k in join_keys.items()])}")
            else:
                st.error("❌ Please select join keys for at least 2 sources to enable data merging!")
            
            st.markdown("---")
            
            # Create combined field list with source indicators
            available_fields = ['']
            for source, fields in available_fields_dict.items():
                for field in fields:
                    available_fields.append(f"{field} ({source})")
        else:
            # Single source (main data)
            available_fields = [''] + list(df.columns)
            st.session_state.join_keys = {}
        
        # Field mapping interface
        field_mapping = {}
        
        # Core SKOS properties
        core_properties = [
            ('uri', 'URI/Identifier'),
            ('prefLabel', 'Preferred Label'),
            ('altLabel', 'Alternative Labels'),
            ('definition', 'Definition'),
            ('scopeNote', 'Scope Note'),
            ('note', 'General Note'),
            ('example', 'Example'),
            ('broader', 'Broader Concept'),
            ('narrower', 'Narrower Concept'),
            ('related', 'Related Concept'),
            ('notation', 'Notation/Code')
        ]
        
        # Additional SKOS properties
        additional_properties = [
            ('editorialNote', 'Editorial Note'),
            ('historyNote', 'History Note'),
            ('changeNote', 'Change Note'),
            ('hiddenLabel', 'Hidden Label'),
            ('exactMatch', 'Exact Match'),
            ('closeMatch', 'Close Match'),
            ('broadMatch', 'Broad Match'),
            ('narrowMatch', 'Narrow Match'),
            ('relatedMatch', 'Related Match')
        ]
        
        # Core properties mapping
        st.markdown("#### 🎯 Core Properties")
        for prop_key, prop_label in core_properties:
            field_mapping[prop_key] = st.selectbox(
                prop_label,
                options=available_fields,
                key=f"mapping_{prop_key}",
                help=f"Map to field containing {prop_label.lower()}"
            )
        
        # Additional properties mapping
        with st.expander("📝 Additional SKOS Properties"):
            for prop_key, prop_label in additional_properties:
                field_mapping[prop_key] = st.selectbox(
                    prop_label,
                    options=available_fields,
                    key=f"mapping_{prop_key}",
                    help=f"Map to field containing {prop_label.lower()}"
                )
        
        # Remove empty mappings
        field_mapping = {k: v for k, v in field_mapping.items() if v}
        
        return field_mapping
