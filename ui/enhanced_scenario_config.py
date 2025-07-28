"""
Enhanced scenario configuration UI for SkoHub TTL Generator.
Supports data enrichment, combination, and flexible hierarchy extraction.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
from core.multi_file_handler import ScenarioType, FileRole, MultiFileHandler
from core.hierarchy_extractor import HierarchySource, HierarchyExtractor
from core.data_enrichment import DataEnrichment


class EnhancedScenarioConfigUI:
    """Enhanced UI for configuring multi-file scenarios with flexible options."""
    
    def __init__(self):
        self.multi_file_handler = MultiFileHandler()
        self.hierarchy_extractor = HierarchyExtractor()
        self.data_enrichment = DataEnrichment()
    
    def render_scenario_selection(self) -> Optional[ScenarioType]:
        """Render scenario type selection UI."""
        st.subheader("📋 Multi-File Scenario Selection")
        
        # Scenario descriptions
        scenario_options = {}
        for scenario_type in ScenarioType:
            config = self.multi_file_handler.get_scenario_config(scenario_type)
            if config:
                scenario_options[config['name']] = scenario_type
        
        selected_name = st.selectbox(
            "Choose your multi-file scenario:",
            options=list(scenario_options.keys()),
            help="Select the type of multi-file processing you need"
        )
        
        if selected_name:
            selected_scenario = scenario_options[selected_name]
            config = self.multi_file_handler.get_scenario_config(selected_scenario)
            
            # Show scenario description
            st.info(f"**{config['name']}**: {config['description']}")
            st.caption(f"Example: {config['example']}")
            
            return selected_scenario
        
        return None
    
    def render_file_upload_and_roles(self, scenario_type: ScenarioType) -> Dict[str, Any]:
        """Render file upload and role assignment UI."""
        st.subheader("📁 File Upload & Role Assignment")
        
        config = self.multi_file_handler.get_scenario_config(scenario_type)
        required_roles = config.get('required_roles', [])
        
        uploaded_files = {}
        file_roles = {}
        
        # File upload
        files = st.file_uploader(
            "Upload your files",
            accept_multiple_files=True,
            type=['csv', 'json', 'ttl', 'txt'],
            help="Upload all files needed for your scenario"
        )
        
        if files:
            st.write("**Assign roles to your files:**")
            
            # Role assignment for each file
            for i, file in enumerate(files):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"📄 **{file.name}**")
                    st.caption(f"Size: {len(file.getvalue())} bytes")
                
                with col2:
                    # Available roles based on scenario
                    available_roles = self._get_available_roles(scenario_type)
                    
                    role = st.selectbox(
                        "Role:",
                        options=[role.value for role in available_roles],
                        key=f"role_{i}",
                        help=f"Assign a role to {file.name}"
                    )
                    
                    file_roles[file.name] = FileRole(role)
                
                uploaded_files[file.name] = file
                st.divider()
            
            # Validate required roles
            assigned_roles = set(file_roles.values())
            missing_roles = [role for role in required_roles if role not in assigned_roles]
            
            if missing_roles:
                st.error(f"Missing required roles: {[role.value for role in missing_roles]}")
                return {}
            
            return {
                'files': uploaded_files,
                'roles': file_roles,
                'scenario_type': scenario_type
            }
        
        return {}
    
    def render_hierarchy_extraction_config(self, datasets: List[Dict]) -> Dict[str, Any]:
        """Render hierarchy extraction configuration UI."""
        st.subheader("🌳 Hierarchy Extraction Configuration")
        
        # Get available hierarchy sources
        available_sources = self.hierarchy_extractor.get_available_sources(datasets)
        
        if not available_sources:
            st.warning("No hierarchy sources detected in your data.")
            return {}
        
        # Source selection
        source_names = {
            HierarchySource.AUTO_DETECT: "🤖 Auto-Detect (Recommended)",
            HierarchySource.TTL_BROADER_NARROWER: "📄 TTL SKOS Hierarchy",
            HierarchySource.CSV_LEVEL_FIELD: "📊 CSV Level/Depth Field",
            HierarchySource.CSV_PARENT_FIELD: "👥 CSV Parent ID Field",
            HierarchySource.CSV_PATH_FIELD: "🗂️ CSV Hierarchical Path",
            HierarchySource.HIERARCHY_FILE: "📋 Separate Hierarchy File"
        }
        
        available_names = [source_names.get(source, source.value) for source in available_sources]
        
        selected_name = st.selectbox(
            "Choose hierarchy extraction source:",
            options=available_names,
            help="Select how to extract hierarchy information"
        )
        
        # Find selected source
        selected_source = None
        for source in available_sources:
            if source_names.get(source, source.value) == selected_name:
                selected_source = source
                break
        
        if not selected_source:
            return {}
        
        # Source-specific configuration
        hierarchy_config = {}
        
        if selected_source != HierarchySource.AUTO_DETECT:
            st.write(f"**Configuration for {selected_name}:**")
            hierarchy_config = self.hierarchy_extractor.get_hierarchy_config_ui(
                selected_source, datasets
            )
        
        return {
            'source': selected_source,
            'config': hierarchy_config
        }
    
    def render_data_enrichment_config(self, existing_ttl: str, enrichment_data: pd.DataFrame) -> Dict[str, Any]:
        """Render data enrichment configuration UI."""
        st.subheader("🔧 Data Enrichment Configuration")
        
        # Enrichment strategy selection
        strategies = {
            'merge_by_uri': "🔗 Merge by URI/ID",
            'merge_by_label': "🏷️ Merge by Label",
            'merge_by_mapping': "🗺️ Custom Field Mapping"
        }
        
        selected_strategy_name = st.selectbox(
            "Choose enrichment strategy:",
            options=list(strategies.values()),
            help="Select how to match enrichment data with existing TTL concepts"
        )
        
        # Find selected strategy
        selected_strategy = None
        for strategy, name in strategies.items():
            if name == selected_strategy_name:
                selected_strategy = strategy
                break
        
        if not selected_strategy:
            return {}
        
        # Strategy-specific configuration
        st.write(f"**Configuration for {selected_strategy_name}:**")
        enrichment_config = self.data_enrichment.get_enrichment_config_ui(
            selected_strategy, enrichment_data
        )
        
        return {
            'strategy': selected_strategy,
            'config': enrichment_config
        }
    
    def render_data_combination_config(self, datasets: List[Dict]) -> Dict[str, Any]:
        """Render data combination configuration UI."""
        st.subheader("🔄 Data Combination Configuration")
        
        # Combination strategies
        strategies = {
            'simple_concat': "📋 Simple Concatenation",
            'merge_duplicates': "🔄 Merge Duplicates",
            'priority_merge': "⭐ Priority-based Merge"
        }
        
        selected_strategy = st.selectbox(
            "Choose combination strategy:",
            options=list(strategies.values()),
            help="Select how to combine multiple datasets"
        )
        
        config = {}
        
        if selected_strategy == "🔄 Merge Duplicates":
            # Duplicate detection configuration
            st.write("**Duplicate Detection:**")
            
            # Get common fields across datasets
            common_fields = self._get_common_fields(datasets)
            
            if common_fields:
                config['merge_key'] = st.selectbox(
                    "Field for duplicate detection:",
                    options=common_fields,
                    help="Field used to identify duplicate records"
                )
                
                config['merge_strategy'] = st.selectbox(
                    "When duplicates found:",
                    options=['keep_first', 'keep_last', 'merge_properties'],
                    help="How to handle duplicate records"
                )
        
        elif selected_strategy == "⭐ Priority-based Merge":
            # Priority configuration
            st.write("**Dataset Priorities:**")
            
            for i, dataset in enumerate(datasets):
                dataset_name = dataset.get('name', f'Dataset {i+1}')
                config[f'priority_{i}'] = st.number_input(
                    f"Priority for {dataset_name}:",
                    min_value=1,
                    max_value=len(datasets),
                    value=i+1,
                    help="Higher numbers = higher priority"
                )
        
        return {
            'strategy': selected_strategy,
            'config': config
        }
    
    def render_processing_preview(self, scenario_config: Dict) -> None:
        """Render processing preview and statistics."""
        st.subheader("👀 Processing Preview")
        
        scenario_type = scenario_config.get('scenario_type')
        
        if scenario_type == ScenarioType.DATA_ENRICHMENT:
            self._render_enrichment_preview(scenario_config)
        elif scenario_type == ScenarioType.DATA_COMBINATION:
            self._render_combination_preview(scenario_config)
        elif scenario_type in [ScenarioType.DISTRIBUTED_DATA, ScenarioType.COMPLETE_RECORDS]:
            self._render_standard_preview(scenario_config)
    
    def _render_enrichment_preview(self, config: Dict) -> None:
        """Render enrichment processing preview."""
        existing_ttl = config.get('existing_ttl', '')
        enrichment_data = config.get('enrichment_data')
        
        if existing_ttl and enrichment_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Existing TTL Concepts", self._count_ttl_concepts(existing_ttl))
                st.metric("TTL File Size", f"{len(existing_ttl)} chars")
            
            with col2:
                st.metric("Enrichment Records", len(enrichment_data))
                st.metric("Enrichment Fields", len(enrichment_data.columns))
            
            # Show sample enrichment data
            st.write("**Sample Enrichment Data:**")
            st.dataframe(enrichment_data.head(3), use_container_width=True)
    
    def _render_combination_preview(self, config: Dict) -> None:
        """Render combination processing preview."""
        datasets = config.get('datasets', [])
        
        if datasets:
            total_records = sum(len(d.get('data', [])) for d in datasets if hasattr(d.get('data'), '__len__'))
            total_files = len(datasets)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Files", total_files)
                st.metric("Total Records", total_records)
            
            with col2:
                avg_records = total_records // total_files if total_files > 0 else 0
                st.metric("Avg Records/File", avg_records)
            
            # Show dataset breakdown
            st.write("**Dataset Breakdown:**")
            for i, dataset in enumerate(datasets):
                data = dataset.get('data')
                name = dataset.get('name', f'Dataset {i+1}')
                
                if hasattr(data, '__len__'):
                    st.write(f"- **{name}**: {len(data)} records")
    
    def _render_standard_preview(self, config: Dict) -> None:
        """Render standard multi-file processing preview."""
        files = config.get('files', {})
        roles = config.get('roles', {})
        
        if files:
            st.write("**File Summary:**")
            
            for filename, file in files.items():
                role = roles.get(filename, 'Unknown')
                size = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"📄 **{filename}**")
                
                with col2:
                    st.write(f"Role: {role.value if hasattr(role, 'value') else role}")
                
                with col3:
                    st.write(f"Size: {size} bytes")
    
    def _get_available_roles(self, scenario_type: ScenarioType) -> List[FileRole]:
        """Get available file roles for a scenario type."""
        if scenario_type == ScenarioType.DATA_ENRICHMENT:
            return [FileRole.EXISTING_TTL, FileRole.ENRICHMENT_SOURCE]
        elif scenario_type == ScenarioType.DATA_COMBINATION:
            return [FileRole.COMBINATION_SOURCE]
        elif scenario_type == ScenarioType.DISTRIBUTED_DATA:
            return [FileRole.PRIMARY, FileRole.SECONDARY, FileRole.HIERARCHY, FileRole.METADATA]
        elif scenario_type == ScenarioType.COMPLETE_RECORDS:
            return [FileRole.PRIMARY, FileRole.SECONDARY]
        else:
            return list(FileRole)
    
    def _get_common_fields(self, datasets: List[Dict]) -> List[str]:
        """Get common fields across all datasets."""
        if not datasets:
            return []
        
        # Get fields from first dataset
        first_data = datasets[0].get('data')
        if isinstance(first_data, pd.DataFrame):
            common_fields = set(first_data.columns)
        else:
            return []
        
        # Intersect with other datasets
        for dataset in datasets[1:]:
            data = dataset.get('data')
            if isinstance(data, pd.DataFrame):
                common_fields &= set(data.columns)
        
        return list(common_fields)
    
    def _count_ttl_concepts(self, ttl_content: str) -> int:
        """Count concepts in TTL content."""
        import re
        concept_pattern = r'<[^>]+>\s+a\s+skos:Concept'
        matches = re.findall(concept_pattern, ttl_content)
        return len(matches)
    
    def render_processing_controls(self) -> Dict[str, Any]:
        """Render processing control buttons and options."""
        st.subheader("🚀 Processing Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            preview_mode = st.checkbox(
                "Preview Mode",
                value=True,
                help="Generate preview without full processing"
            )
        
        with col2:
            validate_output = st.checkbox(
                "Validate Output",
                value=True,
                help="Validate generated TTL against SKOS standards"
            )
        
        with col3:
            include_stats = st.checkbox(
                "Include Statistics",
                value=True,
                help="Generate processing statistics"
            )
        
        # Processing button
        process_button = st.button(
            "🔄 Process Multi-File Data",
            type="primary",
            help="Start processing with current configuration"
        )
        
        return {
            'preview_mode': preview_mode,
            'validate_output': validate_output,
            'include_stats': include_stats,
            'process_button': process_button
        }
