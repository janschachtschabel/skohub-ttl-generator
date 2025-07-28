"""
Scenario configuration UI components for SkoHub TTL Generator.
Provides Streamlit UI for multi-file scenario configuration.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Tuple
from core.multi_file_handler import ScenarioType, FileRole, MultiFileHandler
from core.hierarchy_handler import HierarchyType, HierarchyHandler


class ScenarioConfigUI:
    """UI components for configuring multi-file scenarios."""
    
    def __init__(self):
        self.multi_file_handler = MultiFileHandler()
        self.hierarchy_handler = HierarchyHandler()
    
    def render_scenario_selection(self, file_data: List[Dict]) -> Tuple[ScenarioType, Dict]:
        """Render scenario selection UI"""
        st.subheader("🎯 Multi-File Scenario Configuration")
        
        # Add helpful guidance
        with st.expander("ℹ️ Scenario Selection Help", expanded=True):
            st.markdown("""
            **Choose the right scenario for your data:**
            
            🔗 **Distributed Data (Kldb-Style)**
            - Data for each concept is split across multiple files
            - Example: URIs and labels in file 1, descriptions in file 2
            - Files need to be joined using a common key field
            - Use when: Different files contain different properties for the same concepts
            
            📋 **Complete Records (Esco-Style)**
            - Each file contains complete records for different concept types
            - Example: Skills in file 1, Language skills in file 2
            - Files are combined/concatenated rather than joined
            - Use when: Each file represents a different category of concepts
            """)
        
        if len(file_data) < 2:
            st.info("📄 Single file detected - using complete records scenario")
            return ScenarioType.COMPLETE_RECORDS, {}
        
        # Show file overview
        st.write("**Your uploaded files:**")
        for i, data in enumerate(file_data):
            st.write(f"📄 **{data['filename']}** - {len(data['columns'])} columns, {len(data['data'])} rows")
            with st.expander(f"Preview columns from {data['filename']}"):
                st.write(f"Columns: {', '.join(data['columns'][:10])}{'...' if len(data['columns']) > 10 else ''}")
        
        # Auto-detect scenario
        detected_scenario = self.multi_file_handler.detect_scenario_type(file_data)
        
        # Scenario options with detailed descriptions
        scenario_options = {
            "🔗 Distributed Data (Kldb-Style) - Join files on common key": ScenarioType.DISTRIBUTED_DATA,
            "📋 Complete Records (Esco-Style) - Combine separate concept types": ScenarioType.COMPLETE_RECORDS
        }
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_scenario_name = st.selectbox(
                "Select your data scenario",
                options=list(scenario_options.keys()),
                index=0 if detected_scenario == ScenarioType.DISTRIBUTED_DATA else 1,
                help="Choose based on how your data is structured across files"
            )
            
            selected_scenario = scenario_options[selected_scenario_name]
        
        with col2:
            if detected_scenario == selected_scenario:
                st.success("✅ Auto-detected")
            else:
                st.info("🔄 Manual override")
        
        # Show scenario description
        config = self.multi_file_handler.get_scenario_config(selected_scenario)
        with st.expander("ℹ️ Scenario Details"):
            st.write(f"**Description:** {config.get('description', 'N/A')}")
            st.write(f"**Example:** {config.get('example', 'N/A')}")
        
        return selected_scenario, config
    
    def render_file_role_assignment(self, file_data: List[Dict], scenario_type: ScenarioType) -> Dict[str, FileRole]:
        """Render file role assignment UI"""
        st.subheader("📁 File Role Assignment")
        
        # Add helpful guidance
        with st.expander("ℹ️ File Role Guidelines", expanded=False):
            st.markdown("""
            **Choose the right role for each file:**
            
            🎯 **PRIMARY**: Main file containing URIs/IDs and basic labels
            - Should have: URI, ID, or identifier fields
            - Should have: Labels, names, or titles
            - Example: Main concept definitions with URIs
            
            📋 **SECONDARY**: Additional data to be joined with primary
            - Contains: Extra properties for existing concepts
            - Joined using: Common key fields
            - Example: Additional descriptions or properties
            
            🌳 **HIERARCHY**: File containing parent/child relationships
            - Contains: broader/narrower, parent/child fields
            - Defines: Concept hierarchies and relationships
            - Example: SKOS broader/narrower relationships
            
            📝 **METADATA**: Descriptive information (definitions, notes)
            - Contains: Definitions, descriptions, notes, examples
            - Provides: Rich textual information about concepts
            - Example: Detailed explanations and usage notes
            """)
        
        # Auto-assign roles
        suggested_roles = self.multi_file_handler.configure_file_roles(file_data, scenario_type)
        
        file_roles = {}
        
        st.write("**Assign roles to your files:**")
        
        for i, data in enumerate(file_data):
            filename = data['filename']
            suggested_role = suggested_roles.get(filename, FileRole.SECONDARY)
            
            # Show file info in expandable section
            with st.expander(f"📄 {filename} - {len(data['columns'])} columns, {len(data['data'])} rows"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Sample columns:**")
                    st.write(f"{', '.join(data['columns'][:8])}{'...' if len(data['columns']) > 8 else ''}")
                    
                    # Show sample data
                    if st.button(f"Show sample data", key=f"sample_{i}"):
                        st.dataframe(data['data'].head(3))
                
                with col2:
                    # Role selection with better descriptions
                    role_options = {
                        FileRole.PRIMARY: "🎯 PRIMARY - Main URIs & Labels",
                        FileRole.SECONDARY: "📋 SECONDARY - Additional Data", 
                        FileRole.HIERARCHY: "🌳 HIERARCHY - Parent/Child Relations",
                        FileRole.METADATA: "📝 METADATA - Descriptions & Notes"
                    }
                    
                    selected_role = st.selectbox(
                        f"Role for {filename}",
                        options=list(role_options.keys()),
                        format_func=lambda x: role_options[x],
                        index=list(role_options.keys()).index(suggested_role),
                        key=f"role_{i}",
                        help=f"What type of data does {filename} contain?"
                    )
                    
                    file_roles[filename] = selected_role
                    
                    # Show suggestion status
                    if suggested_role == selected_role:
                        st.success("✅ Auto-suggested")
                    else:
                        st.info("🔄 Manual selection")
        
        return file_roles
    
    def render_join_configuration(self, file_data: List[Dict], file_roles: Dict[str, FileRole], 
                                 scenario_type: ScenarioType) -> Dict:
        """Render join configuration UI"""
        st.subheader("🔗 Join Configuration")
        
        # Add helpful guidance
        with st.expander("ℹ️ Join Configuration Help", expanded=False):
            st.markdown("""
            **When to use different join types:**
            - **Inner Join**: Only keep records that exist in both files
            - **Left Join**: Keep all records from the first file, add matching data from second
            - **Outer Join**: Keep all records from both files
            
            **File Role Guidelines:**
            - **PRIMARY**: Main file with URIs/IDs and basic labels
            - **SECONDARY**: Additional data to be joined
            - **HIERARCHY**: File containing parent/child relationships
            - **METADATA**: Descriptive information (definitions, notes)
            """)
        
        join_config = {}
        
        if scenario_type == ScenarioType.DISTRIBUTED_DATA:
            # Get all available fields from each file
            file_fields = self.multi_file_handler.get_join_field_suggestions(file_data, file_roles)
            
            st.write("**Select join fields from each file:**")
            
            # Show fields from each file
            for i, (filename, fields) in enumerate(file_fields.items()):
                role = file_roles.get(filename, FileRole.SECONDARY)
                role_emoji = {
                    FileRole.PRIMARY: "🎯",
                    FileRole.SECONDARY: "📋", 
                    FileRole.HIERARCHY: "🌳",
                    FileRole.METADATA: "📝"
                }.get(role, "📄")
                
                st.write(f"**{role_emoji} {filename}** ({role.value})")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    selected_field = st.selectbox(
                        f"Join field from {filename}",
                        options=fields,
                        key=f"join_field_{i}",
                        help=f"Select the field to use for joining from {filename}"
                    )
                    join_config[f'join_field_{filename}'] = selected_field
                
                with col2:
                    # Show field preview
                    if st.button(f"Preview", key=f"preview_{i}"):
                        file_data_item = next(d for d in file_data if d['filename'] == filename)
                        sample_values = file_data_item['data'][selected_field].head(3).tolist()
                        st.write(f"Sample: {sample_values}")
            
            # Join strategy selection
            st.write("**Join Strategy:**")
            col1, col2 = st.columns(2)
            
            with col2:
                # Join type
                join_type = st.selectbox(
                    "Join type",
                    options=['outer', 'inner', 'left'],
                    index=0,
                    help="How to handle missing values in joins"
                )
                join_config['how'] = join_type
            
            # URI normalization options
            st.write("**URI Normalization Options:**")
            col1, col2 = st.columns(2)
            
            with col1:
                normalize_uris = st.checkbox(
                    "Normalize URIs for joining",
                    value=True,
                    help="Remove base URI from URIs before joining"
                )
                join_config['normalize_uris'] = normalize_uris
            
            with col2:
                if normalize_uris:
                    base_uri = st.text_input(
                        "Base URI to remove",
                        value="http://",
                        help="Base URI to strip from URIs before joining"
                    )
                    join_config['base_uri'] = base_uri
        
        else:  # COMPLETE_RECORDS
            st.write("Configuration for combining complete record files:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                remove_duplicates = st.checkbox(
                    "Remove duplicates",
                    value=True,
                    help="Remove duplicate records after combining"
                )
                join_config['remove_duplicates'] = remove_duplicates
            
            with col2:
                add_source_info = st.checkbox(
                    "Add source file information",
                    value=True,
                    help="Add a field indicating which file each record came from"
                )
                join_config['add_source_info'] = add_source_info
        
        return join_config
    
    def render_hierarchy_configuration(self, combined_df: pd.DataFrame, 
                                     additional_files: List[Dict] = None, 
                                     file_data: List[Dict] = None) -> Dict:
        """Render hierarchy configuration UI"""
        st.subheader("🌳 Hierarchy Configuration")
        
        # Add helpful guidance
        with st.expander("ℹ️ Hierarchy Configuration Help", expanded=False):
            st.markdown("""
            **Choose the right hierarchy type for your data:**
            
            👨‍👩‍👧‍👦 **Parent Field**: Each record has a field pointing to its parent
            - Example: parent_id, broader_concept, parent_uri
            - Use when: Direct parent-child relationships are explicit
            
            📊 **Level/Depth Field**: Each record has a numeric level indicator
            - Example: level=1,2,3 or depth=0,1,2
            - Use when: Hierarchy depth is explicitly numbered
            - **Perfect for KldB**: Use "Ebene" field (1,2,3,4,5)
            
            🗋️ **Hierarchical Path**: IDs encode the hierarchy structure
            - Example: 1.2.3, /root/child/grandchild, 0-1-11-110
            - Use when: The ID itself shows the hierarchy
            - **Good for KldB**: Schlüssel shows hierarchy (0 → 1 → 11 → 110)
            
            🎯 **SKOS Broader/Narrower**: Explicit SKOS relationships
            - Use when: Data already contains skos:broader/skos:narrower
            """)
        
        # Aggregate all available fields from all sources - USE ORIGINAL COLUMNS!
        all_available_fields = []
        if combined_df is not None and not combined_df.empty:
            all_available_fields.extend(combined_df.columns.tolist())
        
        if file_data:
            for data in file_data:
                # Use ORIGINAL columns from CSV, not processed columns
                if 'original_data' in data and data['original_data'] is not None:
                    all_available_fields.extend(data['original_data'].columns.tolist())
                else:
                    all_available_fields.extend(data['columns'])
        
        # Remove duplicates while preserving order
        all_available_fields = list(dict.fromkeys(all_available_fields))
        
        # Detect hierarchy type
        detected_type = self.hierarchy_handler.detect_hierarchy_type(combined_df, additional_files)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Hierarchy type selection
            hierarchy_options = {
                HierarchyType.PARENT_FIELD: "👨‍👩‍👧‍👦 Parent Field",
                HierarchyType.LEVEL_FIELD: "📊 Level/Depth Field", 
                HierarchyType.PATH_BASED: "🗂️ Hierarchical Path",
                HierarchyType.BROADER_NARROWER: "🎯 SKOS Broader/Narrower",
                HierarchyType.HIERARCHY_FILE: "📄 Separate Hierarchy File"
            }
            
            selected_hierarchy_type = st.selectbox(
                "Hierarchy type",
                options=list(hierarchy_options.keys()),
                format_func=lambda x: hierarchy_options[x],
                index=list(hierarchy_options.keys()).index(detected_type),
                help="How is the hierarchy defined in your data?"
            )
        
        with col2:
            if detected_type == selected_hierarchy_type:
                st.success("✅ Auto-detected")
            else:
                st.info("🔄 Manual override")
        
        # Field suggestions based on hierarchy type
        field_suggestions = self.hierarchy_handler.get_hierarchy_field_suggestions(
            combined_df, selected_hierarchy_type
        )
        
        hierarchy_config = {'type': selected_hierarchy_type.value}
        
        # Configure fields based on hierarchy type
        if selected_hierarchy_type == HierarchyType.PARENT_FIELD:
            st.write("**Select the field containing parent IDs:**")
            
            # Show field suggestions
            parent_candidates = [f for f in all_available_fields if any(keyword in f.lower() for keyword in ['parent', 'broader', 'super', 'up'])]
            
            if parent_candidates:
                st.write(f"💡 **Suggested fields**: {', '.join(parent_candidates)}")
            
            parent_field = st.selectbox(
                "Parent field",
                options=all_available_fields,
                index=all_available_fields.index(parent_candidates[0]) if parent_candidates else 0,
                help="Field containing parent concept ID"
            )
            hierarchy_config['parent_field'] = parent_field
        
        elif selected_hierarchy_type == HierarchyType.LEVEL_FIELD:
            st.write("**Select the field containing hierarchy levels (e.g., 1,2,3,4,5):**")
            
            # Show field suggestions with preview
            level_candidates = [f for f in all_available_fields if any(keyword in f.lower() for keyword in ['level', 'ebene', 'depth', 'tier'])]
            
            if level_candidates:
                st.write(f"💡 **Suggested fields**: {', '.join(level_candidates)}")
            
            level_field = st.selectbox(
                "Level field",
                options=all_available_fields,
                index=all_available_fields.index(level_candidates[0]) if level_candidates else 0,
                help="Field containing hierarchy level/depth (e.g., Ebene with values 1,2,3,4,5)"
            )
            hierarchy_config['level_field'] = level_field
            
            # Show preview if possible
            if file_data:
                for data in file_data:
                    if level_field in data['columns']:
                        sample_values = data['data'][level_field].unique()[:5]
                        st.write(f"📊 **Sample values**: {list(sample_values)}")
                        break
        
        elif selected_hierarchy_type == HierarchyType.PATH_BASED:
            st.write("**Select the field containing hierarchical paths:**")
            
            # Show field suggestions
            path_candidates = [f for f in all_available_fields if any(keyword in f.lower() for keyword in ['path', 'id', 'code', 'schlüssel', 'key', 'identifier'])]
            
            if path_candidates:
                st.write(f"💡 **Suggested fields**: {', '.join(path_candidates)}")
            
            path_field = st.selectbox(
                "Path field",
                options=all_available_fields,
                index=all_available_fields.index(path_candidates[0]) if path_candidates else 0,
                help="Field containing hierarchical path (e.g., Schlüssel KldB with values like 0, 1, 11, 110)"
            )
            hierarchy_config['path_field'] = path_field
            
            path_separator = st.text_input(
                "Path separator",
                value="/",
                help="Character used to separate path levels"
            )
            hierarchy_config['path_separator'] = path_separator
        
        elif selected_hierarchy_type == HierarchyType.BROADER_NARROWER:
            broader_options = [col for col in combined_df.columns if 'broader' in col.lower()]
            narrower_options = [col for col in combined_df.columns if 'narrower' in col.lower()]
            
            col1, col2 = st.columns(2)
            with col1:
                if broader_options:
                    broader_field = st.selectbox("Broader field", options=broader_options)
                    hierarchy_config['broader_field'] = broader_field
            
            with col2:
                if narrower_options:
                    narrower_field = st.selectbox("Narrower field", options=narrower_options)
                    hierarchy_config['narrower_field'] = narrower_field
        
        # Common configuration
        id_field_options = [col for col in combined_df.columns 
                           if col.lower() in ['id', 'uri', 'concepturi', 'identifier']]
        
        if id_field_options:
            id_field = st.selectbox(
                "ID field",
                options=id_field_options,
                help="Field containing concept identifiers"
            )
        else:
            id_field = st.selectbox(
                "ID field",
                options=combined_df.columns.tolist(),
                help="Field containing concept identifiers"
            )
        
        hierarchy_config['id_field'] = id_field
        
        return hierarchy_config
    
    def render_preview_and_validation(self, df: pd.DataFrame, hierarchy_config: Dict) -> None:
        """Render preview and validation UI"""
        st.subheader("👀 Preview & Validation")
        
        # Process hierarchy for preview
        processed_df = self.hierarchy_handler.process_hierarchy(df, hierarchy_config)
        
        # Show preview
        with st.expander("📊 Data Preview"):
            st.dataframe(processed_df.head(10))
        
        # Validate hierarchy
        validation_issues = self.hierarchy_handler.validate_hierarchy(processed_df)
        
        # Show validation results
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Validation Results:**")
            
            total_issues = sum(len(issues) for issues in validation_issues.values())
            if total_issues == 0:
                st.success("✅ No hierarchy issues found")
            else:
                st.warning(f"⚠️ {total_issues} issues found")
                
                for issue_type, issues in validation_issues.items():
                    if issues:
                        st.write(f"- {issue_type.replace('_', ' ').title()}: {len(issues)}")
        
        with col2:
            # Show hierarchy statistics
            stats = self.hierarchy_handler.generate_hierarchy_statistics(processed_df)
            
            st.write("**Hierarchy Statistics:**")
            st.write(f"- Total concepts: {stats['total_concepts']}")
            st.write(f"- Top-level concepts: {stats['top_level_concepts']}")
            st.write(f"- Concepts with children: {stats['concepts_with_children']}")
            if stats['max_depth'] > 0:
                st.write(f"- Maximum depth: {stats['max_depth']}")
        
        # Show detailed issues if any
        if total_issues > 0:
            with st.expander("🔍 Detailed Issues"):
                for issue_type, issues in validation_issues.items():
                    if issues:
                        st.write(f"**{issue_type.replace('_', ' ').title()}:**")
                        for issue in issues[:5]:  # Show first 5 issues
                            st.write(f"- {issue}")
                        if len(issues) > 5:
                            st.write(f"... and {len(issues) - 5} more")
        
        return processed_df
