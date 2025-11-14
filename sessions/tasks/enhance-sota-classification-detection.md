---
name: enhance-sota-classification-detection
branch: main
status: in_progress
created: 2025-11-06
updated: 2025-11-10
---

# Enhance Classification Marking Detection with SOTA Techniques

## Problem/Goal

Critical review of the classmark system against 2024-2025 state-of-the-art reveals significant enhancement opportunities. The current system is a well-engineered rule-based expert system with 6-layer detection (Regex → Fuzzy → Context → Location → Page-Association → LLM) and CAPCO compliance, but lacks several SOTA capabilities that would significantly improve performance and accuracy.

### Current System Strengths
- **Multi-layer detection pipeline** with complementary approaches
- **88+ compiled regex patterns** for CAPCO-compliant classification levels and control markings
- **Sophisticated context analysis** for false positive reduction through official indicators
- **100% local processing** (privacy-focused, no external APIs)
- **Comprehensive test coverage** (48 synthetic documents, 27K+ lines of edge case tests)
- **Page-association algorithm** for detecting repeated banner markings
- **Optional LLM verification** with Qwen3:8b for ambiguous cases

### Critical Gaps Identified
1. **OCR DISABLED** - Cannot process scanned government documents (estimated 80%+ of classified docs are scanned)
2. **Text-only analysis** - Misses visual formatting patterns (bold, font size, positioning, centering) that distinguish classification banners from casual usage
3. **Sequential pattern matching** - 88 regex patterns run sequentially O(patterns × text_length), causing 100-500ms overhead per document
4. **No formal evaluation metrics** - Unknown precision/recall against ground truth, no benchmark dataset with quantified accuracy
5. **Uncalibrated confidence scores** - Heuristic-based scores (0.0-1.0) don't correlate with actual accuracy percentage
6. **No performance optimization** - All documents processed with same expensive pipeline (LLM verification adds seconds per doc)

### SOTA Research Sources (2024-2025)
- **DocLayout-YOLO** (Oct 2024) - Precise document layout detection with DocSynth300K dataset
- **LayoutLMv3/Donut** - Multimodal vision+text understanding, 95% accuracy on document classification tasks
- **YOLOv8 + Tesseract pipeline** - Real-time text region detection + OCR, 3-5x faster than full-page OCR
- **Aho-Corasick automaton** - Multi-pattern matching with O(text_length) complexity regardless of pattern count
- **RVL-CDIP dataset** - 400K document benchmark for evaluating document classification systems
- **Scientific Reports 2024** - Document layout analysis using YOLOv4/YOLOv8 with 95%+ accuracy
- **Platt/Temperature scaling** - Confidence calibration for ML systems
- Meta's open-source Automated Sensitive Document Classification
- 32 CFR Part 2001 Subpart C (NARA classification marking standards)
- DoD CUI program guidance on legacy marking handling (FOUO/SBU/LES transition)

## Success Criteria

### PRIORITY 1: Enable OCR with YOLO-Based Text Detection (CRITICAL GAP)
**Goal**: Process scanned government documents (currently 0% support → target 80-95%)

**Implementation Tasks**:
- [ ] Integrate YOLOv8n (nano model) for text region detection (~5-10ms per page on CPU)
- [ ] Implement Tesseract OCR on detected regions only (not full page)
- [ ] Add selective processing: only OCR pages with insufficient text extraction (<50 words threshold)
- [ ] Build fallback pipeline: text extraction → YOLO detection → Tesseract OCR → combine results
- [ ] Add DPI-independent processing for low-quality scans
- [ ] Implement DPI validation per government standards (200 DPI min for 10pt+ fonts, 300 DPI for technical)
- [ ] Create unit tests for OCR functionality on scanned documents
- [ ] Test on real declassified scanned documents from documents/ folder

**Expected Impact**:
- Process 80%+ more documents (enable scanned doc support)
- 3-5x faster than full-page OCR (region-based processing)
- Works on low-quality scans (YOLO robust to DPI variations)

### PRIORITY 2: Add LayoutLM-based Visual Context Analysis
**Goal**: Reduce false positives by 15-25% through visual formatting analysis

**Implementation Tasks**:
- [ ] Integrate LayoutLMv3-base model (125M params, ~500MB)
- [ ] Create VisualPatternDetector class for extracting visual features
- [ ] Extract features: font size, bold/italic, text positioning, color, centering
- [ ] Implement classification region detection based on visual patterns
- [ ] Fine-tune model on synthetic test dataset (48 docs + generated variations)
- [ ] Add visual confidence layer to detection pipeline (Layer 7)
- [ ] Create unit tests for visual pattern detection
- [ ] Benchmark accuracy improvement on test dataset

**Expected Impact**:
- 15-25% reduction in false positives (distinguish "SECRET" in title vs. banner)
- Better context understanding beyond text patterns
- Robust to OCR errors (visual patterns compensate)
- ~100-200ms processing time per page on GPU

### PRIORITY 3: Replace Sequential Regex with Aho-Corasick Multi-Pattern Matching
**Goal**: Achieve 10-50x faster pattern matching (100-500ms → 10-50ms)

**Implementation Tasks**:
- [ ] Install pyahocorasick dependency
- [ ] Create FastPatternMatcher class using Aho-Corasick automaton
- [ ] Build trie structure with all 88+ classification terms at initialization
- [ ] Implement single-pass matching O(text_length + matches)
- [ ] Keep regex for complex patterns requiring lookahead/lookbehind
- [ ] Add metadata tagging (term type: level, control, authority, etc.)
- [ ] Create unit tests comparing Aho-Corasick vs. current regex performance
- [ ] Benchmark on large document batches (1000+ files)

**Expected Impact**:
- 10-50x faster pattern matching regardless of pattern count
- Scalable: add more patterns without performance penalty
- Memory-efficient trie structure

### PRIORITY 4: Implement Formal Evaluation Framework
**Goal**: Establish quantifiable accuracy metrics (precision, recall, F1)

**Implementation Tasks**:
- [ ] Create ClassificationEvaluator class with standard ML metrics
- [ ] Implement ground truth annotation format (JSON schema)
- [ ] Annotate existing 48 synthetic test documents with ground truth
- [ ] Add real declassified documents from documents/ folder (17+ docs)
- [ ] Expand synthetic test dataset to 500+ documents (use test_data_generator.py)
- [ ] Calculate per-level accuracy (TOP SECRET, SECRET, CONFIDENTIAL, CUI)
- [ ] Track processing time metrics (avg, median, p95, p99)
- [ ] Generate evaluation reports with precision/recall/F1/accuracy
- [ ] Create baseline metrics before other enhancements
- [ ] Add continuous evaluation during development

**Expected Impact**:
- Quantifiable accuracy measurements (e.g., "92% precision, 87% recall")
- Track improvements from each enhancement
- Identify specific failure modes for debugging
- Benchmark against SOTA systems

### PRIORITY 5: Add Confidence Calibration & Uncertainty Quantification
**Goal**: Calibrate confidence scores so 90% confidence = 90% actual accuracy

**Implementation Tasks**:
- [ ] Create CalibratedScorer class
- [ ] Implement Platt scaling or temperature scaling
- [ ] Split test dataset into calibration set (30%) and validation set (70%)
- [ ] Collect predicted scores and actual outcomes on calibration set
- [ ] Fit calibration curve (isotonic regression or logistic)
- [ ] Apply calibration to confidence scores in production
- [ ] Validate calibration accuracy on held-out validation set
- [ ] Update sensitivity thresholds based on calibrated scores
- [ ] Add confidence interval estimation for uncertainty quantification
- [ ] Create visualization of calibration curve

**Expected Impact**:
- Trustworthy confidence scores (±5% accuracy)
- Better threshold setting for high/medium/low sensitivity
- Risk quantification for audit/compliance
- Improved user trust in system decisions

### PRIORITY 6: Hybrid Architecture - Fast Path + Slow Path
**Goal**: Achieve 10-25x overall speedup (80%+ docs take fast path)

**Implementation Tasks**:
- [ ] Design two-stage architecture (fast path for obvious cases, slow path for ambiguous)
- [ ] Implement Fast Path: Aho-Corasick → location analysis → early exit
- [ ] Set fast path thresholds: confidence >95% = ACCEPT, <10% = REJECT
- [ ] Implement Slow Path: YOLO → LayoutLM → context → LLM verification
- [ ] Add decision logic for routing documents between paths
- [ ] Create performance monitoring for path selection
- [ ] Optimize fast path for maximum throughput (200-500 docs/min target)
- [ ] Balance accuracy vs. speed trade-offs
- [ ] Add configuration for adjusting path thresholds
- [ ] Benchmark overall system throughput improvement

**Expected Impact**:
- 80%+ of documents take fast path (10-50x speedup on those)
- Complex/ambiguous cases get thorough analysis
- Overall throughput: 200-500 docs/min (fast) + 10-20/min (slow)
- Resource-efficient: LLM/YOLO only for ambiguous cases

## Context Manifest

### Key Code Locations

**classification_scanner.py** (1,583 lines) - Core detection engine

**Pattern Definitions**:
- Lines 152-287: `ClassificationPatterns` class with all regex patterns
  - Level patterns: TOP SECRET, SECRET, CONFIDENTIAL, CUI
  - Control markings: NOFORN, ORCON, IMCON, RELIDO, PROPIN, FISA, SCI
  - Portion markings: (TS), (S), (C), (U)
  - Authority blocks: CLASSIFIED BY, DERIVED FROM, DECLASSIFY ON, REASON
  - Banner patterns (line 208): Repeated classification markings (ReDoS fixed with 500-char limit)
  - Declassification indicators: DECLASSIFIED, RELEASE AS SANITIZED, APPROVED FOR RELEASE
  - Legacy CUI: Lines 165-169 (currently has some legacy patterns, needs expansion)

**Context Analysis**:
- Lines 290-366: `ContextAnalyzer` class
  - Official indicators (boost confidence): authority blocks, derived from statements, SCG references
  - False positive indicators (reduce confidence): casual phrases like "Victoria's Secret", "secret to success"
  - Window analysis: 300-character context around matches
  - Structural features: line-start positioning, slash patterns, proximity to official indicators

**Text Extraction**:
- Lines 368-756: `TextExtractor` class
  - Lines 437-609: PDF processing (pdfplumber + PyPDF2 fallback)
  - Lines 476-504: Region extraction method (splits pages into header/body/footer)
  - Lines 552-608: OCR support (DISABLED in Desktop Edition) - **CRITICAL: needs re-enabling**
  - Lines 610-673: DOCX processing (python-docx + docx2python)
  - Lines 675-699: Text file processing (multi-encoding support)
  - Lines 701-720: JSON processing

**Fuzzy Matching**:
- Lines 758-812: `FuzzyMatcher` class using RapidFuzz
  - Default 85% similarity threshold
  - 22 known classification terms

**LLM Verification**:
- Lines 813-889: `LLMVerifier` class using Ollama
  - Model: Qwen3:8b
  - Temperature: 0.1 (low for consistency)
  - Structured prompt with official vs. casual usage examples

**Page-Association Algorithm**:
- Lines 911-934: `find_repeated_markings()` - Detects repeated markings across pages
- Lines 1150-1172: Page-association confidence boost logic

**Location-Based Confidence**:
- Lines 940-1010: Page region analysis (header top 10%, footer bottom 10%)
- Confidence boosts: header/footer +20%, metadata +10%, body without official -20%

**Main Scanning Logic**:
- Lines 1013-1133: `scan_text()` method - Main detection pipeline
- Lines 1135-1300: `scan_file()` method - File processing entry point
- Lines 974-1013: Confidence calculation algorithm

**config.py** - Configuration constants
- File size limits (100 MB max)
- Classification levels and their hierarchy
- Sensitivity thresholds (high 30%, medium 50%, low 70%)

**file_manager.py** - File quarantine operations

**test_data_generator.py** (540 lines) - Realistic test data generation
- CAPCO-compliant markings
- Authentic classification authorities
- Real declassification patterns
- Intentional marking errors (84% error rate simulation per GAO findings)

**Test Suites**:
- test_sota_features.py (380 lines) - Unit tests for SOTA features
- test_edge_cases_comprehensive.py (27,791 lines) - Comprehensive edge case tests
- test_format_specific_edge_cases.py (19,063 lines) - Format-specific tests
- run_all_tests.py (10,034 lines) - Test runner

**documents/test_data/** - 48 synthetic test documents across all classification levels
**documents/** - 17+ real declassified historical documents (1971-2000, CIA/DoD)

### SOTA Implementation Details

**1. YOLOv8 + Tesseract OCR Pipeline**
```python
from ultralytics import YOLO
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

class YOLOTextDetector:
    def __init__(self):
        # Load YOLOv8n model (nano - fastest, ~6MB)
        self.model = YOLO('yolov8n.pt')

    def detect_text_regions(self, image_path):
        """Detect text regions using YOLO (~5-10ms per page on CPU)"""
        results = self.model(image_path, classes=[0])  # Class 0 = text

        text_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                text_boxes.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(box.conf)
                })

        return text_boxes

    def extract_text_from_region(self, image, bbox):
        """Extract text from detected region using Tesseract"""
        x1, y1, x2, y2 = bbox
        region = image.crop((x1, y1, x2, y2))
        text = pytesseract.image_to_string(region, config='--psm 6')
        return text

    def process_scanned_pdf(self, pdf_path):
        """Full pipeline: PDF → images → YOLO detection → OCR"""
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)

        all_text = []
        for page_num, image in enumerate(images, 1):
            # Detect text regions
            text_boxes = self.detect_text_regions(image)

            # OCR only detected regions (not entire page)
            page_text = []
            for box in text_boxes:
                text = self.extract_text_from_region(image, box['bbox'])
                page_text.append(text)

            all_text.append({
                'page': page_num,
                'text': '\n'.join(page_text),
                'region': 'body'
            })

        return all_text
```

**2. LayoutLMv3 Visual Pattern Detection**
```python
from transformers import AutoModel, AutoProcessor
import torch

class VisualPatternDetector:
    def __init__(self):
        # Load LayoutLMv3-base (125M params, ~500MB)
        self.model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_visual_features(self, pdf_path):
        """Extract visual features: font size, bold, positioning, color"""
        # Convert PDF page to image
        image = convert_from_path(pdf_path, first_page=1, last_page=1)[0]

        # OCR with bounding boxes
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Prepare inputs for LayoutLM
        encoding = self.processor(image, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoding)
            embeddings = outputs.last_hidden_state

        return embeddings

    def detect_classification_regions(self, embeddings):
        """Detect regions likely to contain classification markings"""
        # Fine-tuned classification head (to be trained)
        # Returns: probability of classification marking per region
        pass
```

**3. Aho-Corasick Fast Pattern Matching**
```python
import ahocorasick

class FastPatternMatcher:
    def __init__(self):
        # Build automaton once at initialization
        self.automaton = ahocorasick.Automaton()

        # Add all classification terms with metadata
        terms = {
            "TOP SECRET": ("level", "TOP SECRET", 1.0),
            "SECRET": ("level", "SECRET", 0.8),
            "CONFIDENTIAL": ("level", "CONFIDENTIAL", 0.6),
            "//NOFORN": ("control", "NOFORN", 0.3),
            "//SCI": ("control", "SCI", 0.3),
            "CLASSIFIED BY:": ("authority", "CLASSIFIED_BY", 0.3),
            "DERIVED FROM:": ("authority", "DERIVED_FROM", 0.3),
            "FOUO": ("cui", "FOUO", 0.5),
            "FOR OFFICIAL USE ONLY": ("cui", "FOUO", 0.5),
            "(TS)": ("portion", "TOP SECRET", 0.2),
            "(S)": ("portion", "SECRET", 0.2),
            "(C)": ("portion", "CONFIDENTIAL", 0.2),
            # ... all 88+ patterns
        }

        for term, metadata in terms.items():
            self.automaton.add_word(term, (term, metadata))

        # Build finite state machine (one-time cost)
        self.automaton.make_automaton()

    def find_all_matches(self, text):
        """Single pass through text finds ALL patterns"""
        # O(text_length + matches) complexity
        matches = []
        for end_index, (term, metadata) in self.automaton.iter(text):
            start_index = end_index - len(term) + 1
            matches.append({
                'term': term,
                'start': start_index,
                'end': end_index + 1,
                'type': metadata[0],
                'level': metadata[1],
                'base_confidence': metadata[2]
            })

        return matches
```

**4. Evaluation Framework**
```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json

class ClassificationEvaluator:
    def __init__(self, ground_truth_path):
        # Load ground truth annotations
        with open(ground_truth_path) as f:
            self.ground_truth = json.load(f)

    def evaluate(self, scanner, test_files):
        """Calculate precision, recall, F1, accuracy"""
        y_true = []
        y_pred = []
        processing_times = []

        for file_path in test_files:
            # Get ground truth
            gt = self.ground_truth.get(file_path, {})
            has_classification = gt.get('has_classification', False)
            expected_level = gt.get('classification_level', None)

            # Run scanner
            import time
            start = time.time()
            result = scanner.scan_file(file_path)
            elapsed = time.time() - start

            processing_times.append(elapsed)

            # Compare
            y_true.append(has_classification)
            y_pred.append(result.has_classification)

        # Calculate metrics
        return {
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'avg_time': sum(processing_times) / len(processing_times),
            'p95_time': sorted(processing_times)[int(len(processing_times) * 0.95)]
        }
```

**5. Confidence Calibration**
```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

class CalibratedScorer:
    def __init__(self):
        self.calibrator = None

    def fit(self, validation_results):
        """Fit calibration curve on validation set"""
        # Collect predicted scores and actual outcomes
        scores = [r['predicted_confidence'] for r in validation_results]
        outcomes = [r['actual_has_classification'] for r in validation_results]

        # Fit isotonic regression (monotonic calibration)
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(scores, outcomes)

    def calibrate_confidence(self, raw_confidence):
        """Map raw confidence to calibrated probability"""
        if self.calibrator:
            return self.calibrator.predict([raw_confidence])[0]
        return raw_confidence
```

**6. Hybrid Fast/Slow Path Architecture**
```python
class HybridScanner:
    def __init__(self):
        self.fast_matcher = FastPatternMatcher()  # Aho-Corasick
        self.yolo_detector = YOLOTextDetector()
        self.visual_detector = VisualPatternDetector()

    def scan_document(self, file_path):
        """Two-stage processing with early exit"""
        # Stage 1: Fast Path (Aho-Corasick + location analysis)
        text = self._extract_text_fast(file_path)
        matches = self.fast_matcher.find_all_matches(text)

        fast_confidence = self._calculate_fast_confidence(matches, text)

        # Early exit for obvious cases
        if fast_confidence > 0.95:
            return {'classification': True, 'confidence': fast_confidence, 'path': 'fast'}
        elif fast_confidence < 0.10:
            return {'classification': False, 'confidence': fast_confidence, 'path': 'fast'}

        # Stage 2: Slow Path (YOLO + LayoutLM + LLM)
        visual_features = self.visual_detector.extract_visual_features(file_path)
        yolo_text = self.yolo_detector.process_scanned_pdf(file_path)

        # Full analysis
        slow_confidence = self._calculate_slow_confidence(
            matches, text, visual_features, yolo_text
        )

        return {'classification': slow_confidence > 0.5, 'confidence': slow_confidence, 'path': 'slow'}
```

### Dependencies to Add

**Required Dependencies** (add to requirements.txt):
```
# Priority 1: OCR with YOLO
torch>=2.0.0                    # PyTorch backend for YOLO and LayoutLM
torchvision>=0.15.0             # Vision utilities
ultralytics>=8.0.0              # YOLOv8 implementation
pytesseract>=0.3.10             # Tesseract OCR wrapper
Pillow>=10.0.0                  # Image processing
pdf2image>=1.16.0               # PDF to image conversion

# Priority 2: Visual Analysis
transformers>=4.30.0            # HuggingFace transformers for LayoutLMv3
layoutparser>=0.3.0             # Document layout analysis tools (optional)

# Priority 3: Fast Pattern Matching
pyahocorasick>=2.0.0            # Aho-Corasick automaton

# Priority 4: Evaluation Framework
scikit-learn>=1.3.0             # ML metrics and calibration
pandas>=2.0.0                   # Data analysis for evaluation reports
matplotlib>=3.7.0               # Visualization for calibration curves

# Existing dependencies (already in requirements.txt)
PyPDF2>=3.0.0
python-docx>=1.0.0
pdfplumber>=0.10.0              # Optional, better PDF layout
docx2python>=2.0.0              # Optional, enhanced DOCX extraction
rapidfuzz>=3.0.0                # Optional, fuzzy matching
tqdm>=4.65.0                    # Optional, progress bars
ollama>=0.1.0                   # Optional, LLM verification
```

**System Dependencies**:
- Tesseract OCR engine (system package): `apt-get install tesseract-ocr` (Linux) or `brew install tesseract` (macOS)
- Poppler utils (for pdf2image): `apt-get install poppler-utils` (Linux) or `brew install poppler` (macOS)
- CUDA toolkit (optional, for GPU acceleration): CUDA 11.8+ recommended

**Model Downloads** (automated on first run):
- YOLOv8n model: ~6MB (downloaded by ultralytics)
- LayoutLMv3-base: ~500MB (downloaded by transformers)
- Qwen3:8b (existing): ~4.7GB (downloaded by ollama)

### Expected Performance Improvements

| Metric | Current | After Enhancements | Improvement |
|--------|---------|-------------------|-------------|
| **Scanned document support** | 0% (OCR disabled) | 80-95% | **CRITICAL - enables major use case** |
| **Detection accuracy** | Unknown (no metrics) | 90-95% precision<br>85-90% recall | **+15-25% estimated** |
| **False positive rate** | High on casual usage<br>("secret recipe", etc.) | Low with visual context<br>(distinguishes formatting) | **-50-70% reduction** |
| **Processing speed** | 10-20 docs/min<br>(with LLM enabled) | **Fast path**: 200-500 docs/min<br>**Slow path**: 10-20 docs/min | **10-25x faster overall** |
| **Pattern matching time** | 100-500ms<br>(88 sequential regex) | 10-50ms<br>(Aho-Corasick) | **10-50x faster** |
| **Confidence calibration** | Uncalibrated heuristics<br>(unknown actual accuracy) | Calibrated scores<br>(±5% of stated confidence) | **Trustworthy thresholds** |
| **Evaluation metrics** | None | Precision/Recall/F1/Accuracy<br>Per-level breakdown | **Quantifiable quality** |
| **Throughput (1000 files)** | 50-100 minutes | 2-5 minutes (fast path)<br>50-100 min (slow path) | **10-25x for typical docs** |

**Key Capability Additions**:
- ✅ Scanned document processing (YOLO + Tesseract OCR)
- ✅ Visual formatting analysis (LayoutLMv3)
- ✅ Multi-pattern matching optimization (Aho-Corasick)
- ✅ Formal evaluation framework (ML metrics)
- ✅ Calibrated confidence scores (trustworthy probabilities)
- ✅ Fast/slow path routing (performance optimization)

### Implementation Roadmap

**Phase 1: Foundation (Weeks 1-3)**
*Goal: Establish baseline metrics and critical capabilities*

1. **Week 1: Evaluation Framework (P4)**
   - Create ClassificationEvaluator class
   - Implement ground truth annotation format
   - Annotate existing 48 synthetic test documents
   - Add 17+ real declassified documents
   - Calculate baseline metrics (precision, recall, F1)
   - *Deliverable: Baseline accuracy report*

2. **Week 2-3: OCR with YOLO (P1)**
   - Integrate YOLOv8n for text region detection
   - Implement Tesseract OCR on detected regions
   - Add selective processing (<50 words threshold)
   - Build fallback pipeline
   - DPI validation (200-300 DPI)
   - Unit tests for OCR functionality
   - Test on scanned documents from documents/ folder
   - *Deliverable: Scanned document processing enabled*

**Phase 2: Performance Optimization (Weeks 4-5)**
*Goal: 10-50x speedup on pattern matching*

3. **Week 4: Aho-Corasick Fast Matching (P3)**
   - Install pyahocorasick dependency
   - Create FastPatternMatcher class
   - Build trie with 88+ classification terms
   - Single-pass matching implementation
   - Keep regex for complex patterns
   - Performance benchmarks
   - *Deliverable: 10-50x faster pattern matching*

4. **Week 5: Expand Test Dataset (P4)**
   - Generate 450+ additional synthetic documents
   - Add variations of existing patterns
   - Include edge cases and error scenarios
   - Annotate ground truth for all
   - *Deliverable: 500+ document test dataset*

**Phase 3: Advanced Detection (Weeks 6-8)**
*Goal: 15-25% accuracy improvement*

5. **Week 6-7: LayoutLMv3 Visual Analysis (P2)**
   - Integrate LayoutLMv3-base model
   - Create VisualPatternDetector class
   - Extract visual features (font, bold, position, color)
   - Implement classification region detection
   - Add visual confidence layer (Layer 7)
   - Unit tests for visual patterns
   - *Deliverable: Visual formatting analysis*

6. **Week 8: Fine-tune LayoutLM (P2)**
   - Prepare training data (synthetic + real docs)
   - Fine-tune classification head
   - Validate on held-out test set
   - Measure accuracy improvement
   - *Deliverable: Fine-tuned model, accuracy report*

**Phase 4: Architecture Enhancement (Weeks 9-10)**
*Goal: 10-25x overall speedup*

7. **Week 9: Hybrid Fast/Slow Path (P6)**
   - Design two-stage architecture
   - Implement Fast Path (Aho-Corasick + location)
   - Set early exit thresholds (>95%, <10%)
   - Implement Slow Path (YOLO + LayoutLM + LLM)
   - Routing logic
   - *Deliverable: Hybrid architecture*

8. **Week 10: Performance Tuning (P6)**
   - Optimize fast path throughput
   - Balance accuracy vs. speed trade-offs
   - Configurable path thresholds
   - Benchmark overall system
   - *Deliverable: 200-500 docs/min fast path*

**Phase 5: Calibration & Polish (Weeks 11-12)**
*Goal: Trustworthy confidence scores*

9. **Week 11: Confidence Calibration (P5)**
   - Create CalibratedScorer class
   - Implement Platt scaling / temperature scaling
   - Split dataset (30% calibration, 70% validation)
   - Fit calibration curve
   - Validate on held-out set
   - *Deliverable: Calibrated confidence scores*

10. **Week 12: Final Validation & Documentation (All)**
    - Run comprehensive test suite
    - Generate final accuracy report
    - Compare before/after metrics
    - Update documentation
    - Create installation guide
    - User guide for new features
    - *Deliverable: Production-ready system*

**Milestones**:
- ✅ **Week 3**: Baseline metrics + OCR enabled
- ✅ **Week 5**: 10-50x faster pattern matching + 500 test docs
- ✅ **Week 8**: Visual analysis integrated, accuracy improved
- ✅ **Week 10**: Hybrid architecture, 10-25x speedup
- ✅ **Week 12**: Calibrated confidence, production ready

### Test Cases Needed
1. **OCR Tests**
   - Scanned PDF with TOP SECRET banner
   - Image-based PDF with portion markings
   - Low-quality scan (test graceful degradation)

2. **FOUO Tests**
   - Document marked "FOUO" (should detect as CUI equivalent)
   - "FOR OFFICIAL USE ONLY" full text
   - "SBU" and "LES" markings

3. **Banner Detection Tests**
   - Multi-page document with consistent header banner
   - Document with footer markings on every page
   - Mixed: banner on some pages but not others

4. **Page-Region Tests**
   - Classification banner in top 5% of page
   - Footer marking in bottom 5% of page
   - Body text containing casual "secret" usage (should not trigger)

## User Notes

**Priority Order:**
1. Task 2 (FOUO patterns) - Quick win, no new dependencies, compliance requirement
2. Task 1 (OCR) - Highest impact for real-world scanned documents
3. Task 3 (Banner detection) - Significantly improves accuracy
4. Tasks 5-6 (DPI validation, confidence boost) - Quality improvements
5. Tasks 7-8 (Dependencies, tests) - Infrastructure

**Research References:**
- Meta OSS tool: https://www.helpnetsecurity.com/2025/06/05/meta-open-source-automated-sensitive-document-classification-tool/
- 32 CFR 2001: https://www.ecfr.gov/current/title-32/subtitle-B/chapter-XX/part-2001/subpart-C
- DoD CUI guidance: https://www.dodcui.mil/
- Header/footer research: https://www.researchgate.net/publication/221253782_Header_and_Footer_Extraction_by_Page-Association

## Work Log

### 2025-11-10
- **Comprehensive SOTA review completed**: Analyzed current system against 2024-2025 state-of-the-art
- **Research findings**:
  - DocLayout-YOLO (Oct 2024) for precise document layout detection
  - LayoutLMv3/Donut achieving 95% accuracy on document classification
  - YOLOv8 + Tesseract pipeline for 3-5x faster OCR
  - Aho-Corasick for 10-50x faster pattern matching
- **Gap analysis identified 6 critical priorities**:
  1. OCR DISABLED - Cannot process 80%+ of scanned government documents
  2. Text-only analysis misses visual formatting patterns
  3. Sequential regex (100-500ms overhead)
  4. No formal evaluation metrics
  5. Uncalibrated confidence scores
  6. No performance optimization (all docs same pipeline)
- **Task file updated with**:
  - 6 priorities with detailed implementation tasks
  - Expected performance improvements table
  - 12-week implementation roadmap (3 phases)
  - Comprehensive SOTA implementation code examples
  - Full dependency list with system requirements
- **Confirmed approach with user**:
  - Full implementation (6-9 weeks)
  - LayoutLMv3 integration approved
  - PyTorch dependency approved
  - GPU availability assumed
  - Proper installer (not single-file constraint)
  - Leverage real declassified docs in documents/ folder (17+ docs)
- **Status**: Task activated, moving to Phase 1 implementation

### 2025-11-06
- Task created after initial SOTA research and gap analysis
- Identified 8 critical improvements based on industry standards and government requirements
- Initial focus on OCR, legacy CUI patterns, banner detection, DPI validation
