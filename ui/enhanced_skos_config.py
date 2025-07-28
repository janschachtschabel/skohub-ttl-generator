"""
Enhanced SKOS Configuration UI

Optimized Streamlit components for new W3C SKOS features:
- Mapping properties configuration
- Collections management
- Enhanced documentation properties
- Performance optimized with caching and lazy loading

Author: SkoHub TTL Generator
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from core.mapping_handler import MappingHandler, MappingType
from core.collections_handler import CollectionsHandler, CollectionType
from core.documentation_handler import DocumentationHandler, DocumentationType

logger = logging.getLogger(__name__)

class EnhancedSKOSConfigUI:
    """
    Streamlit UI components for enhanced SKOS features with performance optimization.
    
    Features:
    - Fast-loading configuration panels
    - Cached computations for responsiveness
    - Lazy loading of complex operations
    - Real-time validation feedback
    """
    
    def __init__(self):
        self.mapping_handler = MappingHandler()
        self.collections_handler = CollectionsHandler()
        self.documentation_handler = DocumentationHandler()
        
        # Initialize session state for performance
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables for caching"""
        if 'mapping_config_cache' not in st.session_state:
            st.session_state.mapping_config_cache = {}
        if 'collections_config_cache' not in st.session_state:
            st.session_state.collections_config_cache = {}
        if 'documentation_config_cache' not in st.session_state:
            st.session_state.documentation_config_cache = {}
    
    @st.cache_data
    def _detect_mapping_fields_cached(_self, df_hash: str, columns: List[str]) -> Dict[str, Optional[str]]:
        """Cached mapping field detection"""
        # Recreate minimal df for detection
        df_mock = pd.DataFrame(columns=columns)
        return _self.mapping_handler.detect_mapping_fields(df_mock)
    
    @st.cache_data
    def _detect_collection_structure_cached(_self, df_hash: str, columns: List[str]) -> Dict[str, Any]:
        """Cached collection structure detection"""
        df_mock = pd.DataFrame(columns=columns)
        return _self.collections_handler.detect_collection_structure(df_mock)
    
    @st.cache_data
    def _detect_documentation_fields_cached(_self, df_hash: str, columns: List[str]) -> Dict[str, List[str]]:
        """Cached documentation field detection"""
        df_mock = pd.DataFrame(columns=columns)
        return _self.documentation_handler.detect_documentation_fields(df_mock)
    
    def render_mapping_configuration(self, file_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Render SKOS mapping properties configuration UI.
        
        Args:
            file_data: List of uploaded file data
            
        Returns:
            Mapping configuration or None
        """
        st.subheader("🔗 SKOS Mapping Properties")
        
        if not file_data:
            st.info("Upload files to configure mapping properties")
            return None
        
        # Quick toggle for mapping features
        enable_mappings = st.checkbox(
            "Enable SKOS Mapping Properties",
            help="Configure cross-vocabulary concept mappings (exactMatch, closeMatch, etc.)"
        )
        
        if not enable_mappings:
            return None
        
        mapping_config = {}
        
        # File selection for mappings (optimized)
        mapping_file_options = {f"File {i+1}: {data['name']}": i 
                              for i, data in enumerate(file_data)}
        
        selected_file_idx = st.selectbox(
            "Select file containing mapping data",
            options=list(mapping_file_options.keys()),
            key="mapping_file_select"
        )
        
        if selected_file_idx:
            file_idx = mapping_file_options[selected_file_idx]
            selected_data = file_data[file_idx]
            df = selected_data['data']
            
            # Use cached field detection for performance
            df_hash = str(hash(tuple(df.columns)))
            detected_fields = self._detect_mapping_fields_cached(df_hash, list(df.columns))
            
            st.write("**Detected Mapping Fields:**")
            
            # Field mapping with auto-detection
            col1, col2 = st.columns(2)
            
            with col1:
                source_uri_field = st.selectbox(
                    "Source Concept URI",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_fields['source_uri']) + 1 
                    if detected_fields['source_uri'] else 0,
                    key="mapping_source_uri"
                )
                
                mapping_type_field = st.selectbox(
                    "Mapping Type Field",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_fields['mapping_type']) + 1 
                    if detected_fields['mapping_type'] else 0,
                    key="mapping_type_field"
                )
            
            with col2:
                target_uri_field = st.selectbox(
                    "Target Concept URI",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_fields['target_uri']) + 1 
                    if detected_fields['target_uri'] else 0,
                    key="mapping_target_uri"
                )
                
                confidence_field = st.selectbox(
                    "Confidence Score (optional)",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_fields['confidence']) + 1 
                    if detected_fields['confidence'] else 0,
                    key="mapping_confidence"
                )
            
            # Quick validation (only if fields are selected)
            if source_uri_field and target_uri_field:
                with st.expander("🔍 Quick Validation", expanded=False):
                    field_mapping = {
                        'source_uri': source_uri_field,
                        'target_uri': target_uri_field,
                        'mapping_type': mapping_type_field if mapping_type_field else None,
                        'confidence': confidence_field if confidence_field else None
                    }
                    
                    # Fast validation (sample only for performance)
                    sample_size = min(1000, len(df))
                    df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
                    
                    validation_result = self.mapping_handler.validate_mapping_data(df_sample, field_mapping)
                    
                    if validation_result['is_valid']:
                        st.success(f"✅ Validation passed ({sample_size} rows checked)")
                    else:
                        st.error("❌ Validation issues found:")
                        for issue in validation_result['issues']:
                            st.error(f"• {issue}")
                    
                    # Show quick stats
                    stats = validation_result['statistics']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Mappings", stats.get('total_mappings', 0))
                    with col2:
                        st.metric("Source Concepts", stats.get('unique_source_concepts', 0))
                    with col3:
                        st.metric("Target Concepts", stats.get('unique_target_concepts', 0))
                
                mapping_config = {
                    'enabled': True,
                    'file_index': file_idx,
                    'field_mapping': field_mapping,
                    'validation_passed': validation_result['is_valid']
                }
        
        return mapping_config if mapping_config else None
    
    def render_collections_configuration(self, file_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Render SKOS collections configuration UI.
        
        Args:
            file_data: List of uploaded file data
            
        Returns:
            Collections configuration or None
        """
        st.subheader("📚 SKOS Collections")
        
        if not file_data:
            st.info("Upload files to configure collections")
            return None
        
        enable_collections = st.checkbox(
            "Enable SKOS Collections",
            help="Group concepts into labeled collections"
        )
        
        if not enable_collections:
            return None
        
        collections_config = {}
        
        # File selection
        collection_file_options = {f"File {i+1}: {data['name']}": i 
                                 for i, data in enumerate(file_data)}
        
        selected_file_idx = st.selectbox(
            "Select file for collection data",
            options=list(collection_file_options.keys()),
            key="collection_file_select"
        )
        
        if selected_file_idx:
            file_idx = collection_file_options[selected_file_idx]
            selected_data = file_data[file_idx]
            df = selected_data['data']
            
            # Cached detection
            df_hash = str(hash(tuple(df.columns)))
            detected_structure = self._detect_collection_structure_cached(df_hash, list(df.columns))
            
            # Configuration options
            col1, col2 = st.columns(2)
            
            with col1:
                collection_field = st.selectbox(
                    "Collection Grouping Field",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_structure['collection_field']) + 1 
                    if detected_structure['collection_field'] else 0,
                    key="collection_grouping_field"
                )
                
                collection_type = st.selectbox(
                    "Collection Type",
                    options=[CollectionType.COLLECTION.value, CollectionType.ORDERED_COLLECTION.value],
                    index=0 if detected_structure['collection_type'] == CollectionType.COLLECTION else 1,
                    key="collection_type_select"
                )
            
            with col2:
                member_field = st.selectbox(
                    "Member Concept URI Field",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_structure['member_field']) + 1 
                    if detected_structure['member_field'] else 0,
                    key="collection_member_field"
                )
                
                order_field = st.selectbox(
                    "Order Field (for ordered collections)",
                    options=[''] + list(df.columns),
                    index=list(df.columns).index(detected_structure['order_field']) + 1 
                    if detected_structure['order_field'] else 0,
                    key="collection_order_field",
                    disabled=collection_type != CollectionType.ORDERED_COLLECTION.value
                )
            
            # Quick preview
            if collection_field and member_field:
                with st.expander("📋 Collection Preview", expanded=False):
                    # Show sample collections (performance optimized)
                    sample_collections = df[collection_field].value_counts().head(5)
                    
                    st.write("**Top 5 Collections:**")
                    for collection_name, count in sample_collections.items():
                        st.write(f"• {collection_name}: {count} members")
                    
                    total_collections = df[collection_field].nunique()
                    total_members = len(df)
                    avg_size = total_members / total_collections if total_collections > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Collections", total_collections)
                    with col2:
                        st.metric("Total Members", total_members)
                    with col3:
                        st.metric("Avg Collection Size", f"{avg_size:.1f}")
                
                collections_config = {
                    'enabled': True,
                    'file_index': file_idx,
                    'collection_field': collection_field,
                    'member_field': member_field,
                    'order_field': order_field if order_field else None,
                    'collection_type': CollectionType.ORDERED_COLLECTION if collection_type == CollectionType.ORDERED_COLLECTION.value else CollectionType.COLLECTION
                }
        
        return collections_config if collections_config else None
    
    def render_documentation_configuration(self, file_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Render enhanced documentation properties configuration UI.
        
        Args:
            file_data: List of uploaded file data
            
        Returns:
            Documentation configuration or None
        """
        st.subheader("📝 Enhanced Documentation Properties")
        
        if not file_data:
            st.info("Upload files to configure documentation")
            return None
        
        enable_enhanced_docs = st.checkbox(
            "Enable Enhanced Documentation",
            help="Configure additional SKOS documentation properties"
        )
        
        if not enable_enhanced_docs:
            return None
        
        documentation_config = {}
        
        # File selection
        doc_file_options = {f"File {i+1}: {data['name']}": i 
                          for i, data in enumerate(file_data)}
        
        selected_file_idx = st.selectbox(
            "Select file for documentation data",
            options=list(doc_file_options.keys()),
            key="documentation_file_select"
        )
        
        if selected_file_idx:
            file_idx = doc_file_options[selected_file_idx]
            selected_data = file_data[file_idx]
            df = selected_data['data']
            
            # Cached detection
            df_hash = str(hash(tuple(df.columns)))
            detected_fields = self._detect_documentation_fields_cached(df_hash, list(df.columns))
            
            st.write("**Map Documentation Fields:**")
            
            # Create tabs for different documentation types
            doc_tabs = st.tabs([
                "📖 Core Docs", 
                "🔍 Scope & Usage", 
                "📝 Editorial", 
                "🕒 History"
            ])
            
            field_mapping = {}
            
            with doc_tabs[0]:  # Core documentation
                col1, col2 = st.columns(2)
                with col1:
                    definition_field = st.selectbox(
                        "Definition Field",
                        options=[''] + list(df.columns),
                        index=self._get_field_index(df.columns, detected_fields.get('definition', [])),
                        key="doc_definition_field"
                    )
                    if definition_field:
                        field_mapping['definition'] = definition_field
                
                with col2:
                    example_field = st.selectbox(
                        "Example Field",
                        options=[''] + list(df.columns),
                        index=self._get_field_index(df.columns, detected_fields.get('example', [])),
                        key="doc_example_field"
                    )
                    if example_field:
                        field_mapping['example'] = example_field
            
            with doc_tabs[1]:  # Scope & Usage
                scope_field = st.selectbox(
                    "Scope Note Field",
                    options=[''] + list(df.columns),
                    index=self._get_field_index(df.columns, detected_fields.get('scopeNote', [])),
                    key="doc_scope_field"
                )
                if scope_field:
                    field_mapping['scopeNote'] = scope_field
                
                note_field = st.selectbox(
                    "General Note Field",
                    options=[''] + list(df.columns),
                    index=self._get_field_index(df.columns, detected_fields.get('note', [])),
                    key="doc_note_field"
                )
                if note_field:
                    field_mapping['note'] = note_field
            
            with doc_tabs[2]:  # Editorial
                editorial_field = st.selectbox(
                    "Editorial Note Field",
                    options=[''] + list(df.columns),
                    index=self._get_field_index(df.columns, detected_fields.get('editorialNote', [])),
                    key="doc_editorial_field"
                )
                if editorial_field:
                    field_mapping['editorialNote'] = editorial_field
                
                change_field = st.selectbox(
                    "Change Note Field",
                    options=[''] + list(df.columns),
                    index=self._get_field_index(df.columns, detected_fields.get('changeNote', [])),
                    key="doc_change_field"
                )
                if change_field:
                    field_mapping['changeNote'] = change_field
            
            with doc_tabs[3]:  # History
                history_field = st.selectbox(
                    "History Note Field",
                    options=[''] + list(df.columns),
                    index=self._get_field_index(df.columns, detected_fields.get('historyNote', [])),
                    key="doc_history_field"
                )
                if history_field:
                    field_mapping['historyNote'] = history_field
            
            # Quick statistics
            if field_mapping:
                with st.expander("📊 Documentation Statistics", expanded=False):
                    total_concepts = len(df)
                    
                    stats_cols = st.columns(len(field_mapping))
                    for i, (doc_type, field_name) in enumerate(field_mapping.items()):
                        with stats_cols[i]:
                            non_empty = df[field_name].notna().sum()
                            coverage = (non_empty / total_concepts) * 100
                            st.metric(
                                f"{doc_type.title()}",
                                f"{non_empty}",
                                f"{coverage:.1f}% coverage"
                            )
                
                documentation_config = {
                    'enabled': True,
                    'file_index': file_idx,
                    'field_mapping': field_mapping
                }
        
        return documentation_config if documentation_config else None
    
    def _get_field_index(self, columns: List[str], detected_fields: List[str]) -> int:
        """Get index of first detected field, or 0 if none found"""
        if detected_fields:
            try:
                return list(columns).index(detected_fields[0]) + 1
            except ValueError:
                pass
        return 0
    
    def render_enhanced_skos_summary(self, mapping_config: Optional[Dict],
                                   collections_config: Optional[Dict],
                                   documentation_config: Optional[Dict]) -> Dict[str, Any]:
        """
        Render summary of enhanced SKOS configuration.
        
        Args:
            mapping_config: Mapping configuration
            collections_config: Collections configuration
            documentation_config: Documentation configuration
            
        Returns:
            Combined configuration summary
        """
        st.subheader("🎯 Enhanced SKOS Features Summary")
        
        enhanced_config = {
            'mapping': mapping_config,
            'collections': collections_config,
            'documentation': documentation_config
        }
        
        # Feature status overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mapping_status = "✅ Enabled" if mapping_config and mapping_config.get('enabled') else "❌ Disabled"
            st.metric("Mapping Properties", mapping_status)
        
        with col2:
            collections_status = "✅ Enabled" if collections_config and collections_config.get('enabled') else "❌ Disabled"
            st.metric("Collections", collections_status)
        
        with col3:
            docs_status = "✅ Enabled" if documentation_config and documentation_config.get('enabled') else "❌ Disabled"
            st.metric("Enhanced Docs", docs_status)
        
        # Detailed configuration info
        if any([mapping_config, collections_config, documentation_config]):
            with st.expander("📋 Configuration Details", expanded=False):
                
                if mapping_config and mapping_config.get('enabled'):
                    st.write("**🔗 Mapping Properties:**")
                    field_mapping = mapping_config.get('field_mapping', {})
                    for field_type, field_name in field_mapping.items():
                        if field_name:
                            st.write(f"• {field_type}: `{field_name}`")
                
                if collections_config and collections_config.get('enabled'):
                    st.write("**📚 Collections:**")
                    st.write(f"• Collection Field: `{collections_config.get('collection_field')}`")
                    st.write(f"• Member Field: `{collections_config.get('member_field')}`")
                    st.write(f"• Type: {collections_config.get('collection_type', CollectionType.COLLECTION).value}")
                
                if documentation_config and documentation_config.get('enabled'):
                    st.write("**📝 Enhanced Documentation:**")
                    field_mapping = documentation_config.get('field_mapping', {})
                    for doc_type, field_name in field_mapping.items():
                        st.write(f"• {doc_type}: `{field_name}`")
        
        return enhanced_config
