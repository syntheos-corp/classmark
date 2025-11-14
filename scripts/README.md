# Classmark Scripts

Build, test, and evaluation scripts for Classmark.

## Test Scripts

### run_all_tests.py

Run all test suites.

**Usage:**
```bash
python scripts/run_all_tests.py
```

**Features:**
- Runs all tests in tests/ directory
- Reports pass/fail statistics
- Shows execution time
- Exit code 0 on success, 1 on failure

### run_baseline_evaluation.py

Run baseline evaluation on ground truth dataset.

**Usage:**
```bash
python scripts/run_baseline_evaluation.py
```

**Features:**
- Evaluates against ground_truth.json
- Calculates precision, recall, F1, accuracy
- Per-level metrics
- Generates detailed report
- Exports results to evaluation_results/

**Output:**
```
Precision: 93.02%
Recall: 100.00%
F1 Score: 96.36%
Accuracy: 93.02%
```

## Build Scripts (in project root)

### Windows Build

```bash
# Build executable
build_windows.bat

# Create installer
build_installer.bat
```

**Output:**
- `dist/Classmark/Classmark.exe`
- `Output/ClassmarkSetup.exe`

### macOS Build

```bash
# Build app bundle
./build_macos.sh

# Create DMG
./build_dmg.sh
```

**Output:**
- `dist/Classmark.app`
- `Output/Classmark-1.0.0.dmg`

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run tests
  run: python scripts/run_all_tests.py

- name: Run evaluation
  run: python scripts/run_baseline_evaluation.py
```

### Local Development

```bash
# Before committing
python scripts/run_all_tests.py && echo "Tests passed"

# Check performance
python scripts/run_baseline_evaluation.py
```

## Performance Monitoring

Run evaluation regularly to track performance:

```bash
# Weekly performance check
python scripts/run_baseline_evaluation.py > weekly_eval.txt
git add weekly_eval.txt
git commit -m "Weekly evaluation results"
```

## Notes

- Scripts should be run from project root
- Ensure all dependencies installed
- Models must be downloaded first
- Results are saved with timestamps
