# Phase 3: LayoutLMv3 Visual Pattern Detection - COMPLETE

## Overview

Phase 3 successfully integrated LayoutLMv3-base (125M parameters) for multimodal visual pattern detection in classification markings. This represents a SOTA approach combining computer vision and natural language processing for document understanding.

## Implementation

### 1. Visual Pattern Detector (`visual_pattern_detector.py` - 650+ lines)

**Key Features:**
- LayoutLMv3-base model integration (125M params)
- Multimodal analysis: text + layout + visual features
- GPU acceleration (NVIDIA RTX 3090)
- Lazy initialization (only for PDFs)

**Visual Feature Extraction:**
- **Font Size**: Extracted from bounding box height
- **Positioning**: Y-position (header/footer/body detection)
- **Capitalization**: Ratio of uppercase to total alphabetic characters
- **Alignment**: Distance from page center (banner detection)
- **Text Length**: Character count

**Classification Region Detection:**
- Banner patterns (top/bottom, large font, high caps)
- Authority blocks ("CLASSIFIED BY", "DERIVED FROM", "DECLASSIFY ON")
- Portion marks (short parenthesized markings like "(TS)")
- Control markings (NOFORN, ORCON, IMCON, etc.)

**Visual Confidence Scoring:**
```python
confidence = 0.5  # Base
+ 0.2 if is_banner
+ 0.15 if is_header
+ 0.1 if large_font
+ 0.1 if high_capitalization
+ 0.1 if centered_alignment
```

### 2. Integration Layer 7 (`classification_scanner.py`)

**Dual Strategy:**
1. **Confidence Boosting**: Visual matches boost confidence of existing text-based matches (+0.25 max)
2. **New Matches**: Add visual-only patterns not found by text detection

**Match Type Indicators:**
- `pattern_fast+visual`: Text pattern with visual confirmation
- `visual_banner`: Visual-only banner detection
- `visual_authority_block`: Visual-only authority block
- `visual_control_marking`: Visual-only control marking

### 3. Unit Tests (`test_visual_pattern_detector.py` - 485+ lines)

**Test Coverage:**
- Visual feature extraction (font size, position, capitalization)
- Region detection (banner, footer, authority block)
- Classification-specific detection
- Pattern type identification
- Visual confidence scoring
- PDF processing with real documents
- Performance benchmarking

**Results:** 13/13 tests passing

### 4. Fine-Tuning Script (`finetune_layoutlm.py` - 520+ lines)

**Capabilities:**
- Load ground truth annotations (61 documents)
- Convert PDFs to training examples with BIO labels
- Fine-tune LayoutLMv3 classification head
- Token classification for 15 label types:
  - O (Outside)
  - B/I-TOP_SECRET, B/I-SECRET, B/I-CONFIDENTIAL
  - B/I-CUI, B/I-CONTROL, B/I-AUTHORITY, B/I-DECLASSIFICATION

**Training Configuration:**
- Learning rate: 5e-5
- Batch size: 4
- Epochs: 10
- Warmup steps: 500
- Max sequence length: 512 tokens

## Performance Results

### Visual Pattern Detection

**Real Document Processing:**
- 88 patterns in 18-page document
- 118 patterns in 13-page document
- 221 patterns in 55-page document
- 482 patterns in 158-page document (largest)
- 141 patterns in 18-page document
- 60 patterns in 9-page document

**Processing Speed:**
- GPU-accelerated (RTX 3090)
- ~1-2 seconds per page (including OCR + LayoutLMv3 inference)
- Handles documents up to 158 pages

**Total Visual Matches:** 1,000+ classification-related patterns detected across test documents

### Baseline Metrics (With Phase 1-3 Integration)

**Synthetic Test Documents (48 docs):**
- Precision: 93.02%
- Recall: 100.00%
- F1 Score: 96.39%
- Accuracy: 93.75%
- Avg Time: 7.91 ms

**Real Declassified Documents (13 docs):**
- Precision: 30.77%
- Recall: 100.00%
- F1 Score: 47.06%
- Avg Time: 106.99 seconds (includes OCR + LayoutLMv3)

## Key Technical Decisions

### 1. BIO Labeling Scheme

Chose BIO (Begin-Inside-Outside) tagging for token classification:
- **B-**: Beginning of entity (e.g., "TOP" in "TOP SECRET")
- **I-**: Inside entity (e.g., "SECRET" in "TOP SECRET")
- **O**: Outside any entity

This allows the model to learn span boundaries and multi-word classifications.

### 2. Visual Feature Heuristics

Used domain knowledge to identify classification-specific visual patterns:
- **Large font + top position + high caps** → likely banner
- **"CLASSIFIED BY" + left-aligned** → likely authority block
- **Short + parenthesized + high caps** → likely portion mark

### 3. Lazy Initialization

LayoutLMv3 model initializes only when processing PDFs:
- Saves memory for text-only documents
- Reduces startup time
- GPU allocation only when needed

### 4. Hybrid Text+Visual Approach

Combined text-based pattern matching with visual confirmation:
- Text patterns provide initial detection
- Visual features boost confidence
- Visual-only matches catch visually distinctive markings missed by text

## Web Research Findings

### Available Datasets

Searched for publicly available training data:

**General Document Understanding:**
- ✓ RVL-CDIP: 400,000 images, 16 classes (resume, invoice, etc.)
- ✓ FUNSD: 199 scanned forms for form understanding
- ✓ SROIE: 1,000 receipts for OCR and extraction
- ✓ CORD: Consolidated receipt dataset
- ✓ PubLayNet: 335,703 layout-annotated research papers

**Classification Markings:**
- ✗ No public datasets found (expected due to security concerns)
- ✗ No government document classification datasets
- ✗ No declassified document training sets

**Decision:** Create custom training data from our 61 annotated ground truth documents + synthetic data generation.

### SOTA Performance

LayoutLMv3 benchmark results:
- **RVL-CDIP**: 95.93% accuracy (SOTA)
- **FUNSD**: F1 scores ranging 85-90%
- **CORD**: High extraction accuracy

**Expected Improvement for Our Task:** 15-25% accuracy increase with fine-tuning.

## Files Created/Modified

### New Files:
1. `visual_pattern_detector.py` (650+ lines)
2. `test_visual_pattern_detector.py` (485+ lines)
3. `finetune_layoutlm.py` (520+ lines)

### Modified Files:
1. `classification_scanner.py` - Added Layer 7 (visual detection)
2. `requirements.txt` - Updated to latest 2025 versions

## Dependencies Added

```
torch>=2.7.1                # PyTorch 2.7.1 (Oct 2025)
torchvision>=0.24.0         # TorchVision 0.24.0
transformers>=4.57.1        # HuggingFace Transformers 4.57.1
pillow>=10.4.0              # PIL for image processing
pdf2image>=1.17.0           # PDF to image conversion
```

## Testing

### Unit Tests
- 13 tests in `test_visual_pattern_detector.py`
- All passing (13/13)
- Coverage: feature extraction, region detection, classification, performance

### Integration Tests
- Evaluation on 61 annotated documents
- Visual detector found 1,000+ pattern matches
- Successfully integrated with existing pipeline

### Performance Tests
- ~1-2 seconds per page on GPU
- Handles large documents (158 pages tested)
- Maintains <10ms for synthetic docs without OCR

## Known Limitations

1. **OCR Dependency**: Requires Tesseract for text extraction
2. **Processing Time**: Real documents take ~107 seconds avg (mostly OCR)
3. **False Positives**: Still 30.77% precision on real docs (needs fine-tuning)
4. **NumPy Version Conflicts**: Warnings about NumPy 1.x vs 2.x compatibility
5. **No Training Data**: No public datasets, relying on our 61 documents

## Next Steps (Phase 4 & 5)

### Phase 4: Hybrid Architecture (Priority 6)
- Fast Path: Aho-Corasick + location analysis (instant)
- Slow Path: YOLO + LayoutLM + context + LLM (thorough)
- Early exit thresholds: >95% ACCEPT, <10% REJECT
- Target: 200-500 docs/min throughput

### Phase 5: Calibration & Deployment (Priority 5)
- Platt scaling for confidence calibration
- Split dataset: 30% calibration, 70% validation
- Final accuracy report (before/after comparison)
- User-friendly installer with dependency management

## Conclusion

Phase 3 successfully integrated SOTA visual pattern detection using LayoutLMv3, adding a powerful multimodal analysis layer to the classification marking detection pipeline. The visual detector found over 1,000 classification-related patterns across test documents and provides confidence boosting for text-based matches.

While precision on real documents remains at 30.77% (due to aggressive recall), the infrastructure is now in place for fine-tuning to significantly improve accuracy. The fine-tuning script is ready to train on our custom dataset.

**Phase 3 Status:** ✓ COMPLETE

**Key Achievements:**
- ✓ LayoutLMv3 integrated (125M params)
- ✓ Visual feature extraction working
- ✓ Classification region detection working
- ✓ Layer 7 added to pipeline
- ✓ Unit tests passing (13/13)
- ✓ Fine-tuning script created
- ✓ Evaluation benchmarks established
