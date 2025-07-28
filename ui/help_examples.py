"""
Help and Examples UI Module for SkoHub TTL Generator

This module provides user guidance, examples, and quick start instructions
integrated directly into the Streamlit application.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

class HelpExamplesUI:
    """UI component for displaying help, examples, and quick start guide"""
    
    def __init__(self):
        self.examples_path = Path("examples")
    
    def render_help_section(self):
        """Render the main help and examples section"""
        st.markdown("## 📚 Kurzanleitung & Beispiele")
        
        # Create tabs for different help sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "🚀 Schnellstart", 
            "📁 Datei-Import", 
            "✋ Manuelle Erstellung", 
            "💡 Beispiele"
        ])
        
        with tab1:
            self._render_quick_start()
        
        with tab2:
            self._render_file_import_guide()
        
        with tab3:
            self._render_manual_creation_guide()
        
        with tab4:
            self._render_examples()
    
    def _render_quick_start(self):
        """Render quick start guide"""
        st.markdown("""
        ### 🎯 Willkommen beim SkoHub TTL Generator!
        
        Dieses Tool hilft Ihnen dabei, **SKOS-konforme Vokabulare** zu erstellen und als TTL-Dateien zu exportieren.
        
        #### 🔄 Drei Hauptworkflows:
        
        **1. 📁 Import from Files**
        - Laden Sie CSV, JSON oder TTL-Dateien hoch
        - Automatische Feldmapping mit KI-Unterstützung
        - Hierarchie-Erkennung aus Datenstruktur
        
        **2. ✋ Manual Creation**
        - Schritt-für-Schritt Assistent (5 Schritte)
        - Vollständige SKOS-Eigenschaftsunterstützung
        - Manuelle Hierarchie- und Beziehungserstellung
        
        **3. 🔧 Batch Processing**
        - Für zukünftige Stapelverarbeitung (in Entwicklung)
        
        #### ⚙️ Konfiguration (Sidebar):
        - **🤖 Local AI**: Aktivieren Sie lokale KI für Feldmapping-Vorschläge
        - **🧠 OpenAI**: Konfigurieren Sie OpenAI API für erweiterte Analyse
        - **🌐 Base URI**: Setzen Sie den Namespace für Ihr Vokabular
        - **📚 Metadata**: Titel, Beschreibung, Sprache des Vokabulars
        """)
        
        # Quick action buttons
        st.markdown("#### 🎬 Sofort loslegen:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📁 Beispiel-CSV laden", key="load_csv_example"):
                st.session_state.show_csv_example = True
        
        with col2:
            if st.button("✋ Manuell erstellen", key="start_manual"):
                st.session_state.mode = "Manual Creation"
                st.rerun()
        
        with col3:
            if st.button("💡 Alle Beispiele", key="show_examples"):
                st.session_state.show_all_examples = True
    
    def _render_file_import_guide(self):
        """Render file import guide"""
        st.markdown("""
        ### 📁 Datei-Import Anleitung
        
        #### Unterstützte Formate:
        
        **📊 CSV-Dateien:**
        - Automatische Encoding-Erkennung (UTF-8, ISO-8859-1, etc.)
        - Hierarchie-Unterstützung durch Level-Spalten
        - Beispiel-Spalten: `Code`, `PrefLabel`, `Definition`, `Level`, `ParentCode`
        
        **📋 JSON-Dateien:**
        - Verschachtelte Eigenschaften werden automatisch abgeflacht
        - `properties.*` Namespace-Expansion
        - Hierarchie-Erkennung aus Struktur
        
        **🔗 TTL-Dateien:**
        - Extraktion bestehender SKOS-Konzepte
        - Zusammenführung mit neuen Daten möglich
        - Vollständige SKOS-Eigenschaftsunterstützung
        
        #### 🔄 Workflow:
        1. **Datei hochladen** - Drag & Drop oder Dateiauswahl
        2. **Feldmapping** - KI-gestützte Zuordnung zu SKOS-Eigenschaften
        3. **Hierarchie konfigurieren** - Level-Felder und Beziehungen definieren
        4. **Vorschau** - Datenstruktur und Mapping überprüfen
        5. **TTL generieren** - SKOS-konforme Ausgabe erstellen
        """)
        
        # Show field mapping example
        st.markdown("#### 🎯 Feldmapping-Beispiel:")
        mapping_example = pd.DataFrame({
            'Ihre Spalte': ['Code', 'Name', 'Beschreibung', 'Ebene', 'Übergeordnet'],
            'SKOS-Eigenschaft': ['skos:notation', 'skos:prefLabel', 'skos:definition', 'level', 'skos:broader'],
            'KI-Konfidenz': ['95%', '98%', '92%', '88%', '85%']
        })
        st.dataframe(mapping_example, use_container_width=True)
    
    def _render_manual_creation_guide(self):
        """Render manual creation guide"""
        st.markdown("""
        ### ✋ Manuelle Erstellung - Schritt-für-Schritt
        
        #### 📋 5-Schritte Assistent:
        
        **Schritt 1: 📚 Vokabular-Metadaten**
        - Titel und Beschreibung des Vokabulars
        - Primäre Sprache auswählen
        - Ersteller/Publisher angeben
        - Domain und Themenbereich definieren
        
        **Schritt 2: 🏷️ Konzepte erstellen**
        - Konzept-ID und bevorzugte Bezeichnung
        - Alternative Bezeichnungen (altLabel)
        - Definition und Notizen
        - Beispiele und Notationen
        - Alle SKOS-Eigenschaften verfügbar
        
        **Schritt 3: 🌳 Hierarchische Struktur**
        - Automatische Hierarchie-Erkennung aus Notationen
        - Manuelle Parent-Child Beziehungen
        - Visuelle Hierarchie-Vorschau
        - Mehrere Hierarchie-Ebenen möglich
        
        **Schritt 4: 🔗 Beziehungen definieren**
        - Verwandte Konzepte (skos:related)
        - Assoziative Beziehungen
        - Mapping-Eigenschaften zu externen Vokabularen
        
        **Schritt 5: ✅ Überprüfung & Export**
        - Vollständige Vokabular-Vorschau
        - TTL-Generierung mit Validierung
        - Download als TTL-Datei
        - Optional: TTL-Bereinigung
        """)
        
        # Progress indicator
        st.markdown("#### 📊 Fortschrittsanzeige:")
        progress_steps = ["Metadaten", "Konzepte", "Hierarchie", "Beziehungen", "Export"]
        cols = st.columns(5)
        for i, (col, step) in enumerate(zip(cols, progress_steps)):
            with col:
                if i < 2:  # Simulate completed steps
                    st.success(f"✅ {step}")
                elif i == 2:  # Current step
                    st.info(f"🔄 {step}")
                else:  # Future steps
                    st.write(f"⏳ {step}")
    
    def _render_examples(self):
        """Render examples section"""
        st.markdown("### 💡 Praktische Beispiele")
        
        # Example selector
        example_type = st.selectbox(
            "Beispiel auswählen:",
            ["KldB-ähnliche Hierarchie (CSV)", "Skill-Ontologie (JSON)", "SKOS-Eigenschaften Übersicht"],
            key="example_selector"
        )
        
        if example_type == "KldB-ähnliche Hierarchie (CSV)":
            self._show_kldb_example()
        elif example_type == "Skill-Ontologie (JSON)":
            self._show_skill_example()
        elif example_type == "SKOS-Eigenschaften Übersicht":
            self._show_skos_properties()
    
    def _show_kldb_example(self):
        """Show KldB-style hierarchy example"""
        st.markdown("""
        #### 🏢 KldB-ähnliche Berufsklassifikation
        
        Dieses Beispiel zeigt eine hierarchische Klassifikation ähnlich der Klassifikation der Berufe (KldB):
        """)
        
        # Sample data
        kldb_data = {
            'Code': ['1', '11', '111', '1111', '2', '21', '211'],
            'PrefLabel': [
                'Bildung und Erziehung',
                'Lehrende Berufe', 
                'Grundschulpädagogik',
                'Kinderbetreuung',
                'Geisteswissenschaften',
                'Sprachwissenschaften',
                'Germanistik'
            ],
            'Definition': [
                'Berufe im Bildungs- und Erziehungswesen',
                'Berufe in der Lehre und Ausbildung',
                'Pädagogische Berufe in der Grundschule',
                'Betreuung und Erziehung von Kindern',
                'Wissenschaftliche Berufe in den Geisteswissenschaften',
                'Berufe in der Sprach- und Literaturwissenschaft',
                'Deutsche Sprach- und Literaturwissenschaft'
            ],
            'Level': [1, 2, 3, 4, 1, 2, 3],
            'ParentCode': ['', '1', '11', '111', '', '2', '21']
        }
        
        df = pd.DataFrame(kldb_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("""
        **🔍 Besonderheiten:**
        - **Hierarchische Codes**: 1 → 11 → 111 → 1111
        - **Level-Spalte**: Definiert die Hierarchie-Ebene
        - **ParentCode**: Verweist auf übergeordnetes Konzept
        - **Automatische Erkennung**: Tool erkennt Hierarchie automatisch
        """)
        
        # Show resulting TTL snippet
        with st.expander("🔗 Generierte TTL-Ausgabe (Auszug)"):
            st.code("""
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix ex: <http://example.org/vocab/> .

ex:1 a skos:Concept ;
    skos:prefLabel "Bildung und Erziehung"@de ;
    skos:definition "Berufe im Bildungs- und Erziehungswesen"@de ;
    skos:notation "1" ;
    skos:topConceptOf ex:vocab .

ex:11 a skos:Concept ;
    skos:prefLabel "Lehrende Berufe"@de ;
    skos:definition "Berufe in der Lehre und Ausbildung"@de ;
    skos:notation "11" ;
    skos:broader ex:1 .

ex:111 a skos:Concept ;
    skos:prefLabel "Grundschulpädagogik"@de ;
    skos:definition "Pädagogische Berufe in der Grundschule"@de ;
    skos:notation "111" ;
    skos:broader ex:11 .
            """, language="turtle")
    
    def _show_skill_example(self):
        """Show skill ontology JSON example"""
        st.markdown("""
        #### 🎯 Skill-Ontologie (JSON)
        
        Beispiel für eine Kompetenz-Ontologie mit verschachtelten Eigenschaften:
        """)
        
        skill_example = {
            "id": "skill-programming",
            "properties": {
                "cclom:title": ["Programming"],
                "cclom:description": ["Ability to write computer programs"],
                "sys:node-uuid": ["skill-001"]
            },
            "broader": ["skill-computer-science"],
            "related": ["skill-software-development", "skill-algorithms"],
            "examples": ["Python programming", "Java development"],
            "level": 2
        }
        
        st.json(skill_example)
        
        st.markdown("""
        **🔄 Automatische Verarbeitung:**
        - `properties.cclom:title` → `skos:prefLabel`
        - `properties.cclom:description` → `skos:definition`
        - `broader` → `skos:broader` Beziehungen
        - `related` → `skos:related` Beziehungen
        - `examples` → `skos:example`
        """)
    
    def _show_skos_properties(self):
        """Show SKOS properties overview"""
        st.markdown("#### 📖 SKOS-Eigenschaften Übersicht")
        
        # Create tabs for different property categories
        prop_tab1, prop_tab2, prop_tab3, prop_tab4 = st.tabs([
            "🏷️ Labels", "📝 Dokumentation", "🔗 Beziehungen", "🗺️ Mapping"
        ])
        
        with prop_tab1:
            st.markdown("""
            **Lexikalische Labels:**
            - `skos:prefLabel` - Bevorzugte Bezeichnung
            - `skos:altLabel` - Alternative Bezeichnung
            - `skos:hiddenLabel` - Versteckte Bezeichnung (für Suche)
            """)
            
            label_example = pd.DataFrame({
                'Eigenschaft': ['prefLabel', 'altLabel', 'hiddenLabel'],
                'Deutsch': ['Programmierung', 'Softwareentwicklung', 'Coding'],
                'Englisch': ['Programming', 'Software Development', 'Coding'],
                'Verwendung': ['Hauptbezeichnung', 'Synonym', 'Suchbegriff']
            })
            st.dataframe(label_example, use_container_width=True)
        
        with prop_tab2:
            st.markdown("""
            **Dokumentations-Eigenschaften:**
            - `skos:definition` - Formale Definition
            - `skos:note` - Allgemeine Notiz
            - `skos:scopeNote` - Anwendungsbereich
            - `skos:example` - Verwendungsbeispiel
            - `skos:editorialNote` - Redaktionelle Notiz
            - `skos:historyNote` - Historische Notiz
            - `skos:changeNote` - Änderungsnotiz
            """)
        
        with prop_tab3:
            st.markdown("""
            **Semantische Beziehungen:**
            - `skos:broader` - Übergeordnetes Konzept
            - `skos:narrower` - Untergeordnetes Konzept
            - `skos:related` - Verwandtes Konzept
            - `skos:broaderTransitive` - Transitiv übergeordnet
            - `skos:narrowerTransitive` - Transitiv untergeordnet
            """)
            
            # Hierarchy visualization
            st.markdown("**🌳 Hierarchie-Beispiel:**")
            st.text("""
            Informatik (broader)
            ├── Programmierung (current)
            │   ├── Python (narrower)
            │   └── Java (narrower)
            └── Datenbanken (related)
            """)
        
        with prop_tab4:
            st.markdown("""
            **Mapping-Eigenschaften:**
            - `skos:exactMatch` - Exakte Entsprechung
            - `skos:closeMatch` - Nahe Entsprechung
            - `skos:broadMatch` - Breitere Entsprechung
            - `skos:narrowMatch` - Engere Entsprechung
            - `skos:relatedMatch` - Verwandte Entsprechung
            """)
    
    def render_sidebar_help(self):
        """Render compact help in sidebar"""
        with st.expander("❓ Schnellhilfe"):
            st.markdown("""
            **🚀 Schnellstart:**
            1. Modus wählen (Import/Manual)
            2. Konfiguration anpassen
            3. Daten eingeben/hochladen
            4. TTL generieren
            
            **💡 Tipp:** Aktivieren Sie Local AI für automatische Feldmapping-Vorschläge!
            """)
    
    def show_welcome_message(self):
        """Show welcome message for new users"""
        if 'first_visit' not in st.session_state:
            st.session_state.first_visit = True
            
            st.info("""
            👋 **Willkommen beim SkoHub TTL Generator!**
            
            Neu hier? Schauen Sie sich die **📚 Kurzanleitung & Beispiele** an, um schnell loszulegen.
            
            💡 **Tipp:** Beginnen Sie mit einem der Beispiele oder dem manuellen Erstellungsmodus.
            """)
