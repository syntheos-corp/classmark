# Classmark Tools

Utilities for dataset management and ground truth annotation.

## Annotation Tools

### annotate_ground_truth.py

Annotate documents for ground truth dataset.

**Usage:**
```bash
cd tools
python annotate_ground_truth.py
```

**Features:**
- Interactive annotation interface
- Mark classification levels and specific markings
- Set confidence ranges
- Save to ground_truth.json

### add_real_docs_to_ground_truth.py

Add real declassified documents to ground truth dataset.

**Usage:**
```bash
cd tools
python add_real_docs_to_ground_truth.py
```

**Features:**
- Scan documents/ directory
- Extract existing annotations
- Add to ground truth database
- Validate annotations

## Ground Truth Format

```json
{
  "file_path": "documents/example.pdf",
  "has_classification": true,
  "classification_level": "SECRET",
  "markings": ["SECRET", "NOFORN"],
  "expected_confidence_range": [0.8, 1.0],
  "notes": "Banner marking on all pages"
}
```

## Workflow

1. **Add Documents:**
   ```bash
   # Add documents to documents/ directory
   cp /path/to/docs/*.pdf documents/
   ```

2. **Annotate:**
   ```bash
   cd tools
   python annotate_ground_truth.py
   ```

3. **Verify:**
   ```bash
   # Check ground_truth.json was updated
   cat ../ground_truth.json | jq '.[] | select(.file_path | contains("new_doc"))'
   ```

4. **Test:**
   ```bash
   cd ..
   python scripts/run_baseline_evaluation.py
   ```

## Best Practices

- **Consistent Annotations:** Use same classification level names
- **Confidence Ranges:** Be realistic (e.g., [0.7, 0.95])
- **Complete Markings:** List ALL markings found
- **Add Notes:** Document unusual cases

## Notes

- Tools operate on files in project root
- Modify PYTHONPATH if needed: `PYTHONPATH=.. python tool.py`
- Backup ground_truth.json before major changes
