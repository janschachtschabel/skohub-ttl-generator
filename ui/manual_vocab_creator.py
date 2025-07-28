"""
Manual Vocabulary Creator UI module for SkoHub TTL Generator.
Provides step-by-step graphical interface for manual vocabulary creation.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
import uuid
from datetime import datetime


class ManualVocabCreator:
    """Handles manual vocabulary creation with step-by-step UI."""
    
    def __init__(self):
        self.steps = [
            "Vocabulary Metadata",
            "Concept Creation", 
            "Hierarchical Structure",
            "Relationships & Properties",
            "Review & Generate"
        ]
        
        # Initialize session state
        if 'manual_vocab_step' not in st.session_state:
            st.session_state.manual_vocab_step = 0
        if 'manual_vocab_data' not in st.session_state:
            st.session_state.manual_vocab_data = {
                'metadata': {},
                'concepts': [],
                'hierarchies': [],
                'relationships': []
            }
    
    def render_manual_creator(self):
        """Render the complete manual vocabulary creator interface"""
        st.header("🎯 Manual Vocabulary Creator")
        st.markdown("Create SKOS vocabularies step-by-step with a guided interface")
        
        # Progress indicator
        self._render_progress_indicator()
        
        # Current step content
        current_step = st.session_state.manual_vocab_step
        
        if current_step == 0:
            self._render_metadata_step()
        elif current_step == 1:
            self._render_concept_creation_step()
        elif current_step == 2:
            self._render_hierarchy_step()
        elif current_step == 3:
            self._render_relationships_step()
        elif current_step == 4:
            self._render_review_step()
        
        # Navigation buttons
        should_generate = self._render_navigation_buttons()
        
        return should_generate
    
    def _render_progress_indicator(self):
        """Render progress indicator showing current step"""
        current_step = st.session_state.manual_vocab_step
        
        cols = st.columns(len(self.steps))
        for i, step_name in enumerate(self.steps):
            with cols[i]:
                if i < current_step:
                    st.success(f"✅ {step_name}")
                elif i == current_step:
                    st.info(f"🔄 {step_name}")
                else:
                    st.write(f"⏳ {step_name}")
        
        st.markdown("---")
    
    def _render_metadata_step(self):
        """Step 1: Vocabulary Metadata"""
        st.subheader("📚 Step 1: Vocabulary Metadata")
        
        col1, col2 = st.columns(2)
        
        with col1:
            title = st.text_input(
                "Vocabulary Title *",
                value=st.session_state.manual_vocab_data['metadata'].get('title', ''),
                help="The main title of your vocabulary"
            )
            
            description = st.text_area(
                "Description *",
                value=st.session_state.manual_vocab_data['metadata'].get('description', ''),
                help="Detailed description of the vocabulary's purpose and scope"
            )
            
            domain = st.text_input(
                "Domain/Subject Area",
                value=st.session_state.manual_vocab_data['metadata'].get('domain', ''),
                help="The subject domain this vocabulary covers"
            )
        
        with col2:
            base_uri = st.text_input(
                "Base URI *",
                value=st.session_state.manual_vocab_data['metadata'].get('base_uri', 'http://w3id.org/openeduhub/vocabs/'),
                help="Base URI for the vocabulary namespace"
            )
            
            vocab_name = st.text_input(
                "Vocabulary Name (for URI) *",
                value=st.session_state.manual_vocab_data['metadata'].get('vocab_name', ''),
                help="Short name used in URI construction"
            )
            
            language = st.selectbox(
                "Primary Language",
                options=['en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'sv', 'da', 'no'],
                index=0 if st.session_state.manual_vocab_data['metadata'].get('language') == 'en' else 1,
                help="Primary language for labels and descriptions",
                key="manual_language_select"
            )
            
            creator = st.text_input(
                "Creator/Author",
                value=st.session_state.manual_vocab_data['metadata'].get('creator', ''),
                help="Person or organization creating this vocabulary"
            )
        
        # Preview URI
        if base_uri and vocab_name:
            full_uri = f"{base_uri.rstrip('/')}/{vocab_name}/"
            st.info(f"**Full Vocabulary URI:** `{full_uri}`")
        
        # Save metadata
        st.session_state.manual_vocab_data['metadata'] = {
            'title': title,
            'description': description,
            'domain': domain,
            'base_uri': base_uri,
            'vocab_name': vocab_name,
            'language': language,
            'creator': creator
        }
        
        # Validation
        required_fields = ['title', 'description', 'base_uri', 'vocab_name']
        missing_fields = [field for field in required_fields 
                         if not st.session_state.manual_vocab_data['metadata'].get(field)]
        
        if missing_fields:
            st.warning(f"Please fill in required fields: {', '.join(missing_fields)}")
            st.session_state.can_proceed = False
        else:
            st.success("✅ Metadata complete!")
            st.session_state.can_proceed = True
    
    def _render_concept_creation_step(self):
        """Step 2: Concept Creation"""
        st.subheader("🏷️ Step 2: Create Concepts")
        
        # Concept creation form
        with st.expander("➕ Add New Concept", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                concept_id = st.text_input(
                    "Concept ID",
                    help="Unique identifier for this concept (leave empty for auto-generation)"
                )
                
                pref_label = st.text_input(
                    "Preferred Label *",
                    help="Main label for this concept"
                )
                
                alt_labels = st.text_area(
                    "Alternative Labels",
                    help="Alternative labels, one per line"
                )
            
            with col2:
                definition = st.text_area(
                    "Definition",
                    help="Formal definition of this concept"
                )
                
                scope_note = st.text_area(
                    "Scope Note",
                    help="Information about the intended meaning or usage"
                )
                
                example = st.text_area(
                    "Example",
                    help="Example usage or instance of this concept"
                )
            
            # Additional SKOS properties
            st.markdown("#### 📝 Additional Properties")
            col3, col4 = st.columns(2)
            
            with col3:
                note = st.text_area(
                    "General Note",
                    help="General note about this concept"
                )
                
                editorial_note = st.text_area(
                    "Editorial Note",
                    help="Note for editors or internal use"
                )
            
            with col4:
                history_note = st.text_area(
                    "History Note",
                    help="Information about the concept's history"
                )
                
                change_note = st.text_area(
                    "Change Note",
                    help="Information about recent changes"
                )
                
                notation = st.text_input(
                    "Notation",
                    help="Code or notation for this concept"
                )
            
            if st.button("➕ Add Concept"):
                if pref_label:
                    concept = {
                        'id': concept_id or str(uuid.uuid4()),
                        'prefLabel': pref_label,
                        'altLabels': [label.strip() for label in alt_labels.split('\n') if label.strip()],
                        'definition': definition,
                        'scopeNote': scope_note,
                        'example': example,
                        'note': note,
                        'editorialNote': editorial_note,
                        'historyNote': history_note,
                        'changeNote': change_note,
                        'notation': notation,
                        'created': datetime.now().isoformat()
                    }
                    st.session_state.manual_vocab_data['concepts'].append(concept)
                    st.success(f"✅ Added concept: {pref_label}")
                    st.rerun()
                else:
                    st.error("Preferred Label is required!")
        
        # Display existing concepts
        concepts = st.session_state.manual_vocab_data['concepts']
        if concepts:
            st.subheader(f"📋 Current Concepts ({len(concepts)})")
            
            for i, concept in enumerate(concepts):
                with st.expander(f"🏷️ {concept['prefLabel']} (ID: {concept['id'][:8]}...)"):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**Preferred Label:** {concept['prefLabel']}")
                        if concept['definition']:
                            st.write(f"**Definition:** {concept['definition']}")
                        if concept['altLabels']:
                            st.write(f"**Alternative Labels:** {', '.join(concept['altLabels'])}")
                        if concept['notation']:
                            st.write(f"**Notation:** {concept['notation']}")
                    
                    with col2:
                        if st.button(f"✏️ Edit", key=f"edit_{i}"):
                            st.session_state[f'editing_concept_{i}'] = True
                    
                    with col3:
                        if st.button(f"🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.manual_vocab_data['concepts'].pop(i)
                            st.rerun()
            
            st.session_state.can_proceed = True
        else:
            st.info("No concepts created yet. Add at least one concept to proceed.")
            st.session_state.can_proceed = False
    
    def _render_hierarchy_step(self):
        """Step 3: Hierarchical Structure"""
        st.subheader("🌳 Step 3: Define Hierarchical Structure")
        
        concepts = st.session_state.manual_vocab_data['concepts']
        if not concepts:
            st.warning("No concepts available. Please add concepts first.")
            return
        
        # Auto-hierarchy detection
        st.markdown("### 🤖 Automatic Hierarchy Detection")
        
        if st.button("🔍 Detect Hierarchies from Notations"):
            hierarchies = self._detect_hierarchies_from_notations(concepts)
            if hierarchies:
                st.session_state.manual_vocab_data['hierarchies'].extend(hierarchies)
                st.success(f"✅ Detected {len(hierarchies)} hierarchical relationships!")
                st.rerun()
            else:
                st.info("No hierarchical patterns detected in notations.")
        
        # Manual hierarchy creation
        st.markdown("### ✋ Manual Hierarchy Definition")
        
        concept_options = {f"{c['prefLabel']} ({c['id'][:8]}...)": c['id'] for c in concepts}
        
        col1, col2 = st.columns(2)
        
        with col1:
            child_concept = st.selectbox(
                "Child Concept (Narrower)",
                options=list(concept_options.keys()),
                help="Select the more specific concept",
                key="hierarchy_child_select"
            )
        
        with col2:
            parent_concept = st.selectbox(
                "Parent Concept (Broader)",
                options=list(concept_options.keys()),
                help="Select the more general concept",
                key="hierarchy_parent_select"
            )
        
        if st.button("➕ Add Hierarchy Relationship"):
            if child_concept != parent_concept:
                hierarchy = {
                    'child_id': concept_options[child_concept],
                    'parent_id': concept_options[parent_concept],
                    'child_label': child_concept.split(' (')[0],
                    'parent_label': parent_concept.split(' (')[0],
                    'type': 'broader'
                }
                st.session_state.manual_vocab_data['hierarchies'].append(hierarchy)
                st.success("✅ Added hierarchical relationship!")
                st.rerun()
            else:
                st.error("Child and parent concepts must be different!")
        
        # Display existing hierarchies
        hierarchies = st.session_state.manual_vocab_data['hierarchies']
        if hierarchies:
            st.markdown("### 📋 Current Hierarchical Relationships")
            
            for i, hierarchy in enumerate(hierarchies):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"🏷️ **{hierarchy['child_label']}** → 🔼 **{hierarchy['parent_label']}**")
                
                with col2:
                    if st.button(f"🗑️", key=f"del_hier_{i}"):
                        st.session_state.manual_vocab_data['hierarchies'].pop(i)
                        st.rerun()
        
        st.session_state.can_proceed = True
    
    def _render_relationships_step(self):
        """Step 4: Relationships & Properties"""
        st.subheader("🔗 Step 4: Define Relationships & Additional Properties")
        
        concepts = st.session_state.manual_vocab_data['concepts']
        if not concepts:
            st.warning("No concepts available. Please add concepts first.")
            return
        
        # Related concepts
        st.markdown("### 🔗 Related Concepts (Associative Relationships)")
        
        concept_options = {f"{c['prefLabel']} ({c['id'][:8]}...)": c['id'] for c in concepts}
        
        col1, col2 = st.columns(2)
        
        with col1:
            concept1 = st.selectbox(
                "First Concept",
                options=list(concept_options.keys()),
                key="relationship_concept1_select"
            )
        
        with col2:
            concept2 = st.selectbox(
                "Related Concept",
                options=list(concept_options.keys()),
                key="relationship_concept2_select"
            )
        
        if st.button("➕ Add Related Relationship"):
            if concept1 != concept2:
                relationship = {
                    'concept1_id': concept_options[concept1],
                    'concept2_id': concept_options[concept2],
                    'concept1_label': concept1.split(' (')[0],
                    'concept2_label': concept2.split(' (')[0],
                    'type': 'related'
                }
                st.session_state.manual_vocab_data['relationships'].append(relationship)
                st.success("✅ Added related relationship!")
                st.rerun()
            else:
                st.error("Concepts must be different!")
        
        # Display existing relationships
        relationships = st.session_state.manual_vocab_data['relationships']
        if relationships:
            st.markdown("### 📋 Current Relationships")
            
            for i, rel in enumerate(relationships):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"🏷️ **{rel['concept1_label']}** ↔ **{rel['concept2_label']}**")
                
                with col2:
                    if st.button(f"🗑️", key=f"del_rel_{i}"):
                        st.session_state.manual_vocab_data['relationships'].pop(i)
                        st.rerun()
        
        st.session_state.can_proceed = True
    
    def _render_review_step(self):
        """Step 5: Review & Generate"""
        st.subheader("📋 Step 5: Review & Generate TTL")
        
        vocab_data = st.session_state.manual_vocab_data
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Concepts", len(vocab_data['concepts']))
        
        with col2:
            st.metric("Hierarchies", len(vocab_data['hierarchies']))
        
        with col3:
            st.metric("Relationships", len(vocab_data['relationships']))
        
        with col4:
            total_properties = sum(
                len([v for v in concept.values() if v and v != '']) 
                for concept in vocab_data['concepts']
            )
            st.metric("Total Properties", total_properties)
        
        # Vocabulary overview
        st.markdown("### 📚 Vocabulary Overview")
        
        metadata = vocab_data['metadata']
        st.write(f"**Title:** {metadata.get('title', 'N/A')}")
        st.write(f"**Description:** {metadata.get('description', 'N/A')}")
        st.write(f"**Domain:** {metadata.get('domain', 'N/A')}")
        st.write(f"**Language:** {metadata.get('language', 'N/A')}")
        st.write(f"**Base URI:** {metadata.get('base_uri', 'N/A')}")
        
        # Concept preview
        if vocab_data['concepts']:
            st.markdown("### 🏷️ Concept Preview")
            
            preview_df = pd.DataFrame([
                {
                    'Preferred Label': c['prefLabel'],
                    'Definition': c.get('definition', '')[:100] + '...' if c.get('definition') else '',
                    'Alt Labels': len(c.get('altLabels', [])),
                    'Notation': c.get('notation', '')
                }
                for c in vocab_data['concepts'][:10]  # Show first 10
            ])
            
            st.dataframe(preview_df, use_container_width=True)
            
            if len(vocab_data['concepts']) > 10:
                st.info(f"Showing first 10 of {len(vocab_data['concepts'])} concepts")
        
        st.session_state.can_proceed = True
    
    def _render_navigation_buttons(self):
        """Render navigation buttons for stepping through the wizard"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.manual_vocab_step > 0:
                if st.button("⬅️ Previous"):
                    st.session_state.manual_vocab_step -= 1
                    st.rerun()
        
        with col3:
            can_proceed = getattr(st.session_state, 'can_proceed', False)
            
            if st.session_state.manual_vocab_step < len(self.steps) - 1:
                if st.button("Next ➡️", disabled=not can_proceed):
                    st.session_state.manual_vocab_step += 1
                    st.rerun()
            else:
                if st.button("🚀 Generate TTL", disabled=not can_proceed):
                    return True  # Signal to generate TTL
        
        return False
    
    def get_manual_vocab_data(self):
        """Get the current manual vocabulary data"""
        return st.session_state.manual_vocab_data
    
    def reset_manual_vocab(self):
        """Reset the manual vocabulary data"""
        st.session_state.manual_vocab_data = {
            'metadata': {},
            'concepts': [],
            'hierarchies': [],
            'relationships': []
        }
        st.session_state.manual_vocab_step = 0
    
    def _detect_hierarchies_from_notations(self, concepts: List[Dict]) -> List[Dict]:
        """Detect hierarchical relationships from notation patterns"""
        hierarchies = []
        
        # Group concepts by notation patterns
        notation_concepts = {c['notation']: c for c in concepts if c.get('notation')}
        
        for notation, concept in notation_concepts.items():
            # Check for dotted notation (e.g., "1.1" is child of "1")
            if '.' in notation:
                parts = notation.split('.')
                if len(parts) > 1:
                    parent_notation = '.'.join(parts[:-1])
                    if parent_notation in notation_concepts:
                        parent_concept = notation_concepts[parent_notation]
                        hierarchies.append({
                            'child_id': concept['id'],
                            'parent_id': parent_concept['id'],
                            'child_label': concept['prefLabel'],
                            'parent_label': parent_concept['prefLabel'],
                            'type': 'broader'
                        })
        
        return hierarchies
    
    def get_manual_vocab_data(self) -> Dict:
        """Get the current manual vocabulary data"""
        return st.session_state.manual_vocab_data
    
    def reset_manual_vocab(self):
        """Reset the manual vocabulary creator"""
        st.session_state.manual_vocab_step = 0
        st.session_state.manual_vocab_data = {
            'metadata': {},
            'concepts': [],
            'hierarchies': [],
            'relationships': []
        }
