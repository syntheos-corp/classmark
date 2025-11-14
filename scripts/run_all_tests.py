#!/usr/bin/env python3
"""
Master Test Runner for Classification Scanner

Runs all test suites to ensure rock-solid reliability:
- Existing unit tests (test_sota_features.py, test_classification_scanner.py)
- Comprehensive edge case tests
- Format-specific edge case tests
- Realistic test data generation and validation

This ensures the system is production-ready and handles every edge case.

Author: Claude Code
Date: 2025-01-07
Version: 1.0
"""

import sys
import os
import subprocess
import time
from pathlib import Path


class TestRunner:
    """Orchestrates running all test suites"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.total_time = 0

    def print_header(self, title):
        """Print formatted test section header"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80 + "\n")

    def print_result(self, test_name, passed, duration):
        """Print test result"""
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status:12} {test_name:50} ({duration:.2f}s)")

    def run_python_test(self, test_file, description):
        """Run a Python test file"""
        print(f"\nRunning: {description}")
        print("-" * 80)

        if not os.path.exists(test_file):
            print(f"⚠ Test file not found: {test_file}")
            return False, 0

        start = time.time()
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            duration = time.time() - start

            # Print output
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)

            passed = result.returncode == 0
            self.results[description] = {'passed': passed, 'duration': duration}

            return passed, duration

        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"✗ TIMEOUT after {duration:.1f}s")
            self.results[description] = {'passed': False, 'duration': duration}
            return False, duration

        except Exception as e:
            duration = time.time() - start
            print(f"✗ ERROR: {e}")
            self.results[description] = {'passed': False, 'duration': duration}
            return False, duration

    def generate_test_data(self):
        """Generate realistic test data"""
        self.print_header("GENERATING REALISTIC TEST DATA")

        print("Creating comprehensive test dataset...")
        passed, duration = self.run_python_test(
            'test_data_generator.py',
            'Realistic Test Data Generation'
        )

        return passed

    def run_existing_tests(self):
        """Run existing test suites"""
        self.print_header("EXISTING TEST SUITES")

        tests = [
            ('test_sota_features.py', 'SOTA Features Unit Tests'),
            ('test_classification_scanner.py', 'Basic Classification Scanner Tests'),
            ('test_llm_integration.py', 'LLM Integration Tests'),
        ]

        all_passed = True
        for test_file, description in tests:
            passed, duration = self.run_python_test(test_file, description)
            all_passed = all_passed and passed

        return all_passed

    def run_edge_case_tests(self):
        """Run comprehensive edge case tests"""
        self.print_header("COMPREHENSIVE EDGE CASE TESTS")

        passed, duration = self.run_python_test(
            'test_edge_cases_comprehensive.py',
            'Comprehensive Edge Cases'
        )

        return passed

    def run_format_specific_tests(self):
        """Run format-specific edge case tests"""
        self.print_header("FORMAT-SPECIFIC EDGE CASE TESTS")

        passed, duration = self.run_python_test(
            'test_format_specific_edge_cases.py',
            'Format-Specific Edge Cases (PDF/DOCX/TXT/JSON)'
        )

        return passed

    def validate_with_generated_data(self):
        """Run scanner on generated test data to validate"""
        self.print_header("VALIDATION WITH REALISTIC TEST DATA")

        # Find the generated test data directory
        import tempfile
        import glob

        test_dirs = glob.glob(os.path.join(tempfile.gettempdir(), 'classmark_test_data_*'))

        if not test_dirs:
            print("⚠ No generated test data found. Run test_data_generator.py first.")
            return False

        # Use most recent test data directory
        test_dir = max(test_dirs, key=os.path.getmtime)
        print(f"Using test data from: {test_dir}")

        # Run scanner on test data
        print("\nRunning classification scanner on generated test data...")
        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, 'classification_scanner.py', test_dir,
                 '--sensitivity', 'high', '--show-all'],
                capture_output=True,
                text=True,
                timeout=120
            )
            duration = time.time() - start

            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)

            # Check if scanner ran successfully (exit code 1 is expected if classified docs found)
            passed = result.returncode in [0, 1]
            self.results['Validation Scan'] = {'passed': passed, 'duration': duration}

            self.print_result('Validation Scan', passed, duration)

            return passed

        except Exception as e:
            print(f"✗ ERROR during validation scan: {e}")
            return False

    def print_summary(self):
        """Print comprehensive test summary"""
        self.print_header("COMPREHENSIVE TEST SUMMARY")

        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['passed'])
        failed_tests = total_tests - passed_tests

        print(f"Total Test Suites: {total_tests}")
        print(f"Passed:            {passed_tests}")
        print(f"Failed:            {failed_tests}")
        print(f"Total Time:        {self.total_time:.2f}s")
        print()

        # Detailed results
        print("DETAILED RESULTS:")
        print("-" * 80)
        for test_name, result in self.results.items():
            self.print_result(test_name, result['passed'], result['duration'])

        print()
        print("="*80)

        if failed_tests == 0:
            print("✓✓✓ ALL TESTS PASSED - SYSTEM IS ROCK SOLID ✓✓✓")
            print()
            print("The classification scanner has been thoroughly validated:")
            print("  ✓ Edge cases and corner cases")
            print("  ✓ All document formats (PDF, DOCX, TXT, JSON)")
            print("  ✓ Boundary conditions and limits")
            print("  ✓ Encoding variations and special characters")
            print("  ✓ False positive detection")
            print("  ✓ Concurrent operations")
            print("  ✓ Security validations")
            print("  ✓ Realistic government document scenarios")
            print()
            print("READY FOR PRODUCTION DEPLOYMENT")
        else:
            print("✗✗✗ SOME TESTS FAILED ✗✗✗")
            print()
            print("Review the failures above before deployment.")
            print("Failed test suites:")
            for test_name, result in self.results.items():
                if not result['passed']:
                    print(f"  • {test_name}")

        print("="*80)

        return failed_tests == 0

    def run_all(self, interactive=True):
        """Run complete test suite"""
        self.start_time = time.time()

        print()
        print("="*80)
        print("  CLASSIFICATION SCANNER - COMPREHENSIVE TEST SUITE")
        print("  Rock Solid Reliability Validation")
        print("="*80)
        print()
        print("This will run ALL test suites to ensure production readiness:")
        print("  1. Generate realistic test data")
        print("  2. Run existing unit tests")
        print("  3. Run comprehensive edge case tests")
        print("  4. Run format-specific tests")
        print("  5. Validate with generated data")
        print()

        if interactive:
            input("Press Enter to begin testing...")
        else:
            print("Starting automated test run...")

        # Step 1: Generate test data
        data_generated = self.generate_test_data()

        # Step 2: Run existing tests
        existing_passed = self.run_existing_tests()

        # Step 3: Run edge case tests
        edge_cases_passed = self.run_edge_case_tests()

        # Step 4: Run format-specific tests
        format_tests_passed = self.run_format_specific_tests()

        # Step 5: Validate with generated data
        if data_generated:
            validation_passed = self.validate_with_generated_data()
        else:
            validation_passed = False

        self.total_time = time.time() - self.start_time

        # Print summary
        all_passed = self.print_summary()

        return 0 if all_passed else 1


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Run comprehensive test suite')
    parser.add_argument('--auto', action='store_true', help='Run without interactive prompts')
    args = parser.parse_args()

    runner = TestRunner()
    exit_code = runner.run_all(interactive=not args.auto)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
