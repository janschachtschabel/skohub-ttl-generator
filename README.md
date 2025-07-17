# 🏷️ SkoHub TTL Generator

**Universal tool for generating SKOS-compliant TTL vocabularies from CSV, JSON, or TTL files**

*Enhanced with local AI intelligence, memory-efficient processing, and comprehensive quality assurance*

## 🚀 Features

### 📁 Multi-Format Data Support
- **CSV Files**: Automatic encoding detection (UTF-8, Latin1, CP1252)
- **JSON Files**: Nested object support with automatic flattening
- **TTL Files**: Concept extraction and transformation
- **Multiple Files**: Memory-efficient combination of large datasets
- **ESCO Support**: Optimized for ESCO Skills and Occupations processing

### 🤖 Local AI Intelligence
- **Sentence-Transformers**: Multilingual semantic similarity (German/English)
- **Offline Processing**: No API keys or internet connection required
- **Smart Field Mapping**: Intelligent SKOS property suggestions
- **Vocabulary Metadata**: AI-generated titles and descriptions
- **Confidence Scoring**: Reliability indicators for suggestions
- **Privacy-First**: All processing happens locally

### 🔗 Flexible URI Management
- **w3id.org Standard**: Default openeduhub namespace
- **Custom Base URIs**: Full flexibility for your vocabulary
- **Auto-Generation**: UUID-based URIs when needed
- **URI Resolution**: Smart relative/absolute URI handling

### 📝 Manual Vocabulary Creation
- **Small Vocabularies**: Create concepts manually through UI
- **Real-time Preview**: See your vocabulary as you build it
- **Concept Management**: Add, edit, remove concepts easily
- **Instant TTL**: Generate TTL from manual entries

### ✅ Enhanced Quality Assurance
- **TTL Cleaner Integration**: Automatic validation and cleaning with audit trail
- **Syntax Validation**: Ensures SKOS compliance and proper TTL formatting
- **Duplicate Detection**: Intelligent removal of duplicate concepts
- **Change Logging**: Comprehensive modification tracking with timestamps
- **Memory Efficiency**: Chunked processing for large datasets (15,000+ concepts)
- **URI Normalization**: Consistent @base URI handling and relative URI resolution
- **SKOS Property Preservation**: Maintains all relationships and metadata

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- 4GB+ RAM (recommended for large datasets)
- Internet connection for initial model download

### Quick Start
```bash
# Clone or navigate to the directory
cd skohub_ttl_generator

# Install dependencies (includes sentence-transformers and torch)
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Dependencies
- `streamlit>=1.28.0` - Web interface
- `pandas>=1.5.0` - Data processing
- `sentence-transformers>=2.2.0` - Local AI model
- `torch>=1.13.0` - Neural network backend
- `numpy>=1.21.0` - Numerical computing

### First Run
The multilingual sentence-transformer model (`all-MiniLM-L6-v2`) will be automatically downloaded on first use (~90MB).

## 📋 Usage Guide

### 🎯 ESCO Skills TTL Creation (Complete Workflow)

To create a comprehensive ESCO Skills TTL containing all 4 skill groups from multiple CSV files:

#### Step 1: Prepare ESCO Data
1. **Download ESCO v1.2.0** German CSV files:
   - `skills_de.csv` - Main skills data (15,000+ skills)
   - `skillGroups_de.csv` - Skill group classifications
   - `skillsCollection_de.csv` - Skill collections
   - `broaderRelationsSkills.csv` - Hierarchical relationships

#### Step 2: Upload Multiple Files
1. **Upload all 4 CSV files** using the file uploader
2. **Preview each dataset** to verify structure and content
3. **Check statistics**: Total concepts, columns, data types

#### Step 3: Configure Dataset Combination
1. **Join Strategy**: Select "Merge" for related data
2. **Join Key**: Use `conceptUri` or `uri` as primary key
3. **Remove Duplicates**: Enable to eliminate duplicate concepts
4. **Memory-Efficient Processing**: Automatic chunked merging for large datasets

#### Step 4: Field Mapping with AI
1. **AI Suggestions**: Click "🤖 Get AI suggestions" for each file
2. **Map ESCO fields**:
   - `conceptUri` → **uri** (unique identifier)
   - `preferredLabel` → **prefLabel** (main skill name)
   - `alternativeLabel` → **altLabel** (synonyms)
   - `description` → **description** (skill definition)
   - `broaderUri` → **broader** (parent skills)
   - `skillType` → **note** (skill classification)
3. **Confidence Scores**: Use suggestions with >0.3 confidence
4. **Manual Adjustments**: Fine-tune mappings as needed

#### Step 5: Base URI Configuration
1. **Recommended**: `http://w3id.org/openeduhub/vocabs/escoSkills/`
2. **Alternative**: `http://w3id.org/openeduhub/vocabs/skills/`
3. **Custom**: Your organization's URI scheme

#### Step 6: Generate Complete TTL
1. **Review Summary**: Verify all 4 datasets are included
2. **Generate TTL**: Creates comprehensive SKOS vocabulary
3. **Automatic Cleaning**: TTL Cleaner removes duplicates and fixes syntax
4. **Download Results**:
   - `esco_skills_complete.ttl` - Main vocabulary file
   - `esco_skills_complete_changes.log` - Audit trail
   - Validation report with statistics

#### Expected Results:
- **Total Concepts**: ~15,000 unique skills
- **Skill Groups**: All 4 ESCO skill categories included
- **Relationships**: Hierarchical broader/narrower preserved
- **Languages**: German labels with @de language tags
- **ESCO References**: `skos:exactMatch` to original ESCO URIs
- **File Size**: ~8-12MB depending on relationships

### 🏢 ESCO Occupations TTL Creation

To create a comprehensive ESCO Occupations TTL from multiple CSV files:

#### Required Files:
1. **Download ESCO v1.2.0** German CSV files:
   - `occupations_de.csv` - Main occupations data (3,000+ occupations)
   - `researchOccupationsCollection_de.csv` - Research occupations (120+ entries)
   - `occupationGroups_de.csv` - ISCO group classifications (optional)

#### Workflow:
1. **Upload Files**: Upload `occupations_de.csv` and `researchOccupationsCollection_de.csv`
2. **Field Mapping**:
   - `conceptUri` → **uri** (unique identifier)
   - `preferredLabel` → **prefLabel** (occupation name)
   - `alternativeLabel` → **altLabel** (alternative names)
   - `description` → **description** (occupation definition)
   - `iscoGroup` → **note** (ISCO classification)
3. **Base URI**: Use `http://w3id.org/openeduhub/vocabs/escoOccupations/`
4. **Expected Results**:
   - **Total Concepts**: ~3,200 unique occupations
   - **Research Collection**: Specialized research occupations included
   - **ISCO References**: Professional classification preserved
   - **File Size**: ~2-3MB

### 1. 📁 General Data Upload
- Upload CSV, JSON, or TTL files
- Multiple files supported for complex vocabularies
- Automatic format detection and encoding handling
- Preview data structure and statistics

### 2. 🔧 Intelligent Field Mapping
- Map your data fields to SKOS properties:
  - **uri**: Unique identifier
  - **prefLabel**: Main term/label
  - **altLabel**: Alternative labels/synonyms
  - **description**: Definition or scope note
  - **broader**: Parent concepts
  - **narrower**: Child concepts
- **Local AI Suggestions**: Semantic similarity-based mapping recommendations
- **Confidence Scores**: Reliability indicators for each suggestion (>0.3 threshold)
- **Multilingual Support**: Works with German and English field names
- **Context-Aware**: Uses field names + sample values for better accuracy
- **Real-time Preview**: See mapped data instantly

### 3. 📝 Manual Entry (Optional)
- Create small vocabularies manually
- Add concepts with all SKOS properties
- Real-time concept management
- Perfect for controlled vocabularies

### 4. ✅ Generate & Validate
- Review configuration summary
- Generate SKOS-compliant TTL
- Automatic validation with TTL Cleaner
- Download clean TTL and validation reports

## 🔧 Configuration

### Base URI Options
- `http://w3id.org/openeduhub/vocabs/` (default)
- `http://w3id.org/openeduhub/vocabs/skills/`
- `http://w3id.org/openeduhub/vocabs/occupations/`
- Custom URI of your choice

### Local AI Model
- **Model**: `all-MiniLM-L6-v2` (multilingual sentence-transformer)
- **Languages**: German and English optimized
- **Size**: ~90MB download on first use
- **Performance**: <100ms inference time
- **Privacy**: All processing happens locally

### Intelligent Field Mapping
The local AI model suggests field mappings using semantic similarity:
- **URI fields**: id, uri, identifier, uuid, code, conceptId
- **Label fields**: label, name, title, term, prefLabel, bezeichnung
- **Alternative labels**: altLabel, synonym, aliases, alternative
- **Descriptions**: description, definition, note, scopeNote, beschreibung
- **Hierarchies**: broader, parent, narrower, children, übergeordnet
- **Confidence threshold**: Only suggestions >0.3 similarity shown

## 📊 Output Structure

Generated TTL files follow SKOS standards:

```turtle
@base <http://w3id.org/openeduhub/vocabs/myVocab/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dct: <http://purl.org/dc/terms/> .

<> a skos:ConceptScheme ;
    dct:title "My Vocabulary"@en ;
    dct:description "Generated with SkoHub TTL Generator"@en ;
    dct:created "2025-01-17"^^xsd:date .

<concept-1> a skos:Concept ;
    skos:prefLabel "Main Term"@de ;
    skos:altLabel "Alternative Term"@de ;
    skos:scopeNote "Definition of the concept"@de ;
    skos:broader <parent-concept> ;
    skos:inScheme <> .
```

## 🔍 Enhanced Quality Assurance

### Integrated TTL Cleaner
- **Location**: `ttl_cleaner.py` in the application directory
- **Duplicate Removal**: Intelligent elimination of duplicate concepts with URI tracking
- **Syntax Validation**: Ensures proper TTL formatting with @base URI handling
- **URI Normalization**: Consistent relative/absolute URI resolution
- **SKOS Property Preservation**: Maintains all relationships (exactMatch, broader, narrower)
- **Change Logging**: Comprehensive audit trail with timestamps
- **Memory Efficiency**: Chunked processing for large datasets (15,000+ concepts)

### Comprehensive Validation Reports
- **Statistics**: Concept counts, duplicates removed, labels processed
- **Change Log**: Detailed modification tracking with before/after states
- **Syntax Fixes**: Comma spacing corrections and TTL structure repairs
- **File Outputs**: Clean TTL + detailed change log + console report
- **ESCO Compatibility**: Preserves ESCO URI references and hierarchies
- **Downloadable Logs**: Complete audit trail for compliance

## 🎯 Use Cases

### 1. ESCO Data Processing
- Convert ESCO CSV exports to TTL
- Maintain ESCO URI references
- Clean and validate large datasets

### 2. Custom Vocabularies
- Create domain-specific vocabularies
- Educational competency frameworks
- Skill taxonomies and classifications

### 3. Data Migration
- Convert legacy vocabularies to SKOS
- Merge multiple data sources
- Standardize vocabulary formats

### 4. Research Projects
- Build research vocabularies
- Prototype concept schemes
- Validate vocabulary structures

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web interface with responsive design
- **Backend**: Python with pandas for efficient data processing
- **AI Engine**: Local sentence-transformers for semantic similarity
- **Neural Network**: PyTorch backend for model inference
- **Validation**: Enhanced TTL Cleaner with audit trail
- **Output**: SKOS-compliant TTL files with comprehensive logging

### Performance & Scalability
- **Large Datasets**: Tested with 17,000+ ESCO concepts
- **Memory Efficiency**: Chunked processing prevents memory overflow
- **Fast AI**: <100ms inference time for field mapping
- **Batch Processing**: Optimized for bulk operations
- **Duplicate Handling**: Efficient hash-based deduplication

### System Requirements
- **Python**: 3.8+ (tested up to 3.11)
- **RAM**: 4GB+ recommended for large datasets
- **Storage**: 200MB+ for model and dependencies
- **OS**: Windows, macOS, Linux compatible

### Dependencies
- **Streamlit**: 1.28+ (web interface)
- **Pandas**: 1.5+ (data processing)
- **Sentence-Transformers**: 2.2+ (local AI)
- **PyTorch**: 1.13+ (neural network backend)
- **NumPy**: 1.21+ (numerical computing)

## 📝 Examples

### CSV to TTL
```csv
id,term,definition,parent
1,Python,Programming language,programming
2,Java,Object-oriented language,programming
```

### Manual Entry
Create concepts through the web interface:
- **URI**: `concept-1`
- **Preferred Label**: `Machine Learning`
- **Alternative Labels**: `ML | Artificial Intelligence`
- **Description**: `Computer systems that learn from data`

### Multiple File Processing
Combine data from:
- `skills.csv` (main vocabulary)
- `hierarchies.json` (parent-child relationships)
- `existing.ttl` (extend existing vocabulary)

## 🆕 Recent Enhancements

### 🤖 Local AI Integration (v2.0)
- **Replaced OpenAI** with local sentence-transformers model
- **Multilingual Support**: Optimized for German and English
- **Privacy-First**: All processing happens offline
- **Cost-Free**: No API keys or usage fees
- **Fast Performance**: <100ms inference time

### 📊 Memory-Efficient Processing
- **Chunked Dataset Merging**: Handles large ESCO datasets without memory errors
- **Smart Duplicate Removal**: Efficient hash-based deduplication
- **Memory Monitoring**: Automatic fallback strategies for large files
- **Optimized Performance**: Tested with 17,000+ concepts

### 📈 Enhanced Quality Assurance
- **Comprehensive Change Logging**: Detailed audit trail with timestamps
- **SKOS Property Preservation**: Maintains all relationships and metadata
- **Advanced URI Handling**: Proper @base URI resolution
- **Validation Reports**: Complete statistics and modification tracking

## 🚀 Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Run**: `streamlit run app.py`
3. **Upload**: Add your data files
4. **Map**: Configure field mappings with AI assistance
5. **Generate**: Create clean TTL vocabulary
6. **Download**: Get your SKOS-compliant file + validation reports

## 🔗 Integration

### With Skills&More
- Generated TTL files are compatible with Skills&More Streamlit app
- Use TTL upload feature for seamless integration
- Maintains w3id.org URI consistency

### With SkoHub
- Output follows SkoHub vocabulary standards
- Ready for SkoHub Vocabs deployment
- SKOS-compliant structure

## 📞 Support

For questions, issues, or feature requests:
- Check the validation logs for detailed error information
- Use verbose mode for debugging
- Ensure proper field mapping configuration
- Verify base URI format

---

**Built with ❤️ for the semantic web community**
