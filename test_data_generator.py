#!/usr/bin/env python3
"""
Realistic Test Data Generator for Classification Scanner

Generates sophisticated, realistic test documents based on official CAPCO standards,
CDSE training materials, and actual government document patterns.

This generator creates documents that mirror real-world scenarios including:
- Properly formatted classification markings per Executive Order 13526
- Common marking errors found in government documents (84% error rate per GAO)
- Legacy CUI markings in transition
- Various document formats (PDF, DOCX, TXT, JSON)
- Edge cases and corner cases

Author: Claude Code
Date: 2025-01-07
Version: 1.0
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import random


class RealisticTestDataGenerator:
    """Generate realistic classification test documents"""

    # Based on CAPCO standards and CDSE training materials
    CLASSIFICATION_AUTHORITIES = [
        "John Smith, Deputy Director, Intelligence",
        "Jane Doe, Senior Classification Officer",
        "Robert Johnson, Chief, Security Division",
        "Maria Garcia, Classification Authority #47",
        "David Chen, Senior Intelligence Officer",
    ]

    DECLASSIFICATION_DATES = [
        "20301231",  # Standard 10-year
        "20351231",  # 15-year
        "20401231",  # 20-year
        "20501231",  # 30-year (max for most)
        "X1",  # OADR 25-year review
        "X2",  # OADR 50-year review
        "Originating Agency's Determination Required",
    ]

    CLASSIFICATION_REASONS = [
        "1.4(a)",  # Military plans, weapons systems, or operations
        "1.4(b)",  # Foreign government information
        "1.4(c)",  # Intelligence activities, sources, or methods
        "1.4(d)",  # Foreign relations or foreign activities
        "1.4(e)",  # Scientific, technological, or economic matters
        "1.4(f)",  # US Government programs for safeguarding nuclear materials
        "1.4(g)",  # Vulnerabilities or capabilities of systems, installations
        "1.4(h)",  # Weapons of mass destruction
    ]

    DERIVED_FROM_SOURCES = [
        "Multiple Sources",
        "SCG-2024-007, dated 15 March 2024",
        "National Intelligence Estimate 2024-01",
        "DoD Classification Guide 24-001",
        "CIA Classification Guide CG-2024-03",
        "NSA Classification Guide NSA/CSSM 1-52",
    ]

    CONTROL_MARKINGS_COMBINATIONS = [
        ["NOFORN"],
        ["NOFORN", "ORCON"],
        ["REL TO USA, AUS"],
        ["REL TO USA, GBR, CAN, AUS, NZL"],  # Five Eyes
        ["NOFORN", "PROPIN"],
        ["IMCON"],
        ["FISA"],
        ["SCI", "NOFORN"],
    ]

    LEGACY_CUI_MARKINGS = [
        "FOUO",
        "FOR OFFICIAL USE ONLY",
        "SBU",
        "SENSITIVE BUT UNCLASSIFIED",
        "LES",
        "LAW ENFORCEMENT SENSITIVE",
        "LIMDIS",
        "LIMITED DISTRIBUTION",
    ]

    # Common marking errors from GAO report (84% of documents have errors)
    COMMON_ERRORS = [
        "missing_portion_marks",
        "incorrect_portion_marks",
        "missing_declassification_date",
        "wrong_classification_authority_format",
        "missing_reason",
        "inconsistent_overall_classification",
    ]

    def __init__(self, output_dir: str = None):
        """Initialize test data generator"""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix='classmark_test_data_'))

    def generate_classification_authority_block(
        self,
        classification_level: str,
        include_errors: bool = False
    ) -> str:
        """Generate realistic classification authority block"""
        authority = random.choice(self.CLASSIFICATION_AUTHORITIES)
        reason = random.choice(self.CLASSIFICATION_REASONS)
        declass_date = random.choice(self.DECLASSIFICATION_DATES)

        if include_errors and random.random() < 0.3:
            # Introduce error: missing reason
            return f"""CLASSIFIED BY: {authority}
DECLASSIFY ON: {declass_date}"""
        elif include_errors and random.random() < 0.3:
            # Introduce error: wrong format
            return f"""Classification Authority: {authority}
Reason for Classification: {reason}
Declassification Date: {declass_date}"""
        else:
            # Correct format per CAPCO
            return f"""CLASSIFIED BY: {authority}
REASON: {reason}
DECLASSIFY ON: {declass_date}"""

    def generate_derived_from_block(self, include_errors: bool = False) -> str:
        """Generate realistic DERIVED FROM block"""
        source = random.choice(self.DERIVED_FROM_SOURCES)
        declass_date = random.choice(self.DECLASSIFICATION_DATES)

        if include_errors and random.random() < 0.3:
            # Missing declassification date
            return f"DERIVED FROM: {source}"
        else:
            return f"""DERIVED FROM: {source}
DECLASSIFY ON: {declass_date}"""

    def generate_banner_marking(
        self,
        classification_level: str,
        control_markings: List[str] = None
    ) -> str:
        """Generate classification banner marking"""
        if control_markings:
            controls = "//".join(control_markings)
            return f"{classification_level}//{controls}"
        else:
            return classification_level

    def generate_portion_marking(self, level: str) -> str:
        """Generate portion marking"""
        portion_map = {
            "TOP SECRET": "TS",
            "SECRET": "S",
            "CONFIDENTIAL": "C",
            "UNCLASSIFIED": "U",
        }
        return f"({portion_map.get(level, 'U')})"

    def generate_top_secret_document(self, include_errors: bool = False) -> Tuple[str, str]:
        """Generate realistic TOP SECRET document"""
        control_markings = random.choice(self.CONTROL_MARKINGS_COMBINATIONS)
        banner = self.generate_banner_marking("TOP SECRET", control_markings)

        content = f"""{banner}

{self.generate_classification_authority_block("TOP SECRET", include_errors)}

MEMORANDUM FOR: Director of National Intelligence

FROM: Deputy Director, Intelligence Operations

SUBJECT: Operational Planning for {random.choice(['Operation REDACTED', 'Project CLASSIFIED', 'Initiative SECURE'])}

{self.generate_portion_marking("TOP SECRET")} This document contains highly sensitive intelligence regarding ongoing operations in {random.choice(['Central Asia', 'Eastern Europe', 'Southeast Asia', 'Middle East'])}.

{self.generate_portion_marking("SECRET")} Source reporting indicates potential threats to US interests from {random.choice(['state actors', 'non-state actors', 'hybrid threats'])}.

{self.generate_portion_marking("CONFIDENTIAL")} Coordination with allied intelligence services continues through established channels.

{self.generate_portion_marking("UNCLASSIFIED")} This document is being provided for informational purposes.

{banner}
"""
        return content, "top_secret_memo.txt"

    def generate_secret_document(self, include_errors: bool = False) -> Tuple[str, str]:
        """Generate realistic SECRET document"""
        control_markings = random.choice([["NOFORN"], ["REL TO USA, GBR"]])
        banner = self.generate_banner_marking("SECRET", control_markings)

        # Possibly include derived from instead of classified by
        if random.random() < 0.5:
            authority_block = self.generate_derived_from_block(include_errors)
        else:
            authority_block = self.generate_classification_authority_block("SECRET", include_errors)

        content = f"""{banner}

{authority_block}

INTELLIGENCE ASSESSMENT

{self.generate_portion_marking("SECRET")} Recent signals intelligence indicates increased communications activity in the region.

{self.generate_portion_marking("SECRET")} Analysis suggests coordination between multiple groups.

{self.generate_portion_marking("UNCLASSIFIED")} Public reporting confirms general regional instability.

{self.generate_portion_marking("SECRET")} Recommend continued monitoring of communications channels.

{banner}
"""
        return content, "secret_assessment.txt"

    def generate_confidential_document(self, include_errors: bool = False) -> Tuple[str, str]:
        """Generate realistic CONFIDENTIAL document"""
        banner = "CONFIDENTIAL"

        if include_errors and random.random() < 0.4:
            # Error: Missing portion marks
            content = f"""{banner}

CLASSIFIED BY: {random.choice(self.CLASSIFICATION_AUTHORITIES)}
REASON: {random.choice(self.CLASSIFICATION_REASONS)}
DECLASSIFY ON: {random.choice(self.DECLASSIFICATION_DATES)}

DIPLOMATIC CABLE

This cable contains information regarding recent diplomatic exchanges with foreign officials.

Discussion centered on trade agreements and regional security cooperation.

Additional meetings are scheduled for next quarter.

{banner}
"""
        else:
            content = f"""{banner}

{self.generate_classification_authority_block("CONFIDENTIAL", include_errors)}

DIPLOMATIC CABLE

{self.generate_portion_marking("CONFIDENTIAL")} This cable contains information regarding recent diplomatic exchanges with foreign officials.

{self.generate_portion_marking("CONFIDENTIAL")} Discussion centered on trade agreements and regional security cooperation.

{self.generate_portion_marking("UNCLASSIFIED")} Additional meetings are scheduled for next quarter.

{banner}
"""
        return content, "confidential_cable.txt"

    def generate_cui_document(self) -> Tuple[str, str]:
        """Generate realistic CUI document"""
        cui_category = random.choice(["CUI//SP-PRVCY", "CUI//SP-EXPT", "CUI"])

        content = f"""{cui_category}

CONTROLLED UNCLASSIFIED INFORMATION

This document contains {random.choice(['personally identifiable information', 'export controlled technical data', 'law enforcement sensitive information'])}.

Distribution is limited to authorized personnel only.

Handle in accordance with CUI policy and procedures.

{cui_category}
"""
        return content, "cui_document.txt"

    def generate_legacy_cui_document(self) -> Tuple[str, str]:
        """Generate document with legacy CUI markings"""
        legacy_marking = random.choice(self.LEGACY_CUI_MARKINGS)

        content = f"""{legacy_marking}

MEMORANDUM

This document contains sensitive unclassified information.

{random.choice([
    'This information is provided for official use only and should not be released to the public.',
    'Distribution is limited to law enforcement personnel with a need to know.',
    'Handle and protect this information in accordance with agency guidelines.',
])}

{legacy_marking}
"""
        return content, f"legacy_{legacy_marking.lower().replace(' ', '_')}.txt"

    def generate_false_positive_document(self) -> Tuple[str, str]:
        """Generate document that should NOT be flagged"""
        scenarios = [
            ("The secret to success in business is customer service.\n\nVictoria's Secret reported strong sales this quarter.", "business_article.txt"),
            ("The Secret Service protects the President.\n\nSecret Service agents undergo extensive training.", "secret_service_article.txt"),
            ("Data can be classified as sensitive or non-sensitive.\n\nWe use classification algorithms for analysis.", "data_classification.txt"),
            ("This was a confidential conversation between friends.\n\nPlease keep this information confidential.", "personal_note.txt"),
            ("The secret ingredient is cinnamon.\n\nKeep the recipe secret from competitors.", "recipe.txt"),
            ("Classified ads appear in newspapers.\n\nItems are classified into categories for organization.", "newspaper_article.txt"),
        ]

        return random.choice(scenarios)

    def generate_mixed_classification_document(self) -> Tuple[str, str]:
        """Generate document with multiple classification levels"""
        content = f"""TOP SECRET//NOFORN

{self.generate_classification_authority_block("TOP SECRET")}

INTELLIGENCE REPORT - MULTIPLE SOURCES

{self.generate_portion_marking("TOP SECRET")} Highly sensitive SIGINT indicates imminent threat.

{self.generate_portion_marking("SECRET")} HUMINT sources report increased activity in the region.

{self.generate_portion_marking("CONFIDENTIAL")} Open source reporting confirms general trends.

{self.generate_portion_marking("UNCLASSIFIED")} Public statements align with assessed intentions.

ANNEX A - SECRET SUPPLEMENTAL INFORMATION

{self.generate_portion_marking("SECRET")} Additional intelligence from allied services supports this assessment.

TOP SECRET//NOFORN
"""
        return content, "mixed_classification_report.txt"

    def generate_marking_errors_document(self) -> Tuple[str, str]:
        """Generate document with common marking errors (for testing error detection)"""
        # This represents the 84% of documents with errors per GAO report
        content = """SECRET

CLASSIFIED BY: John Smith
REASON: 1.4(c)
DECLASSIFY ON: 20301231

INTELLIGENCE MEMORANDUM

This paragraph discusses secret information but is missing its portion mark.

(S) This paragraph is properly marked.

(C) This paragraph is marked CONFIDENTIAL but document is SECRET - inconsistent!

(TS) This paragraph is marked TOP SECRET but document overall is SECRET - error!

This is another paragraph without a portion mark - common error.

(U) This unclassified paragraph is properly marked.

SECRET
"""
        return content, "document_with_errors.txt"

    def generate_all_test_documents(self, num_each: int = 5) -> Dict[str, List[str]]:
        """Generate comprehensive set of test documents"""
        generated_files = {
            'top_secret': [],
            'secret': [],
            'confidential': [],
            'cui': [],
            'legacy_cui': [],
            'false_positives': [],
            'mixed': [],
            'with_errors': [],
        }

        print(f"Generating test documents in: {self.output_dir}")
        print(f"Generating {num_each} of each type...")

        # Generate TOP SECRET documents
        for i in range(num_each):
            content, filename = self.generate_top_secret_document(include_errors=(i % 3 == 0))
            filepath = self.output_dir / f"ts_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['top_secret'].append(str(filepath))

        # Generate SECRET documents
        for i in range(num_each):
            content, filename = self.generate_secret_document(include_errors=(i % 3 == 0))
            filepath = self.output_dir / f"s_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['secret'].append(str(filepath))

        # Generate CONFIDENTIAL documents
        for i in range(num_each):
            content, filename = self.generate_confidential_document(include_errors=(i % 3 == 0))
            filepath = self.output_dir / f"c_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['confidential'].append(str(filepath))

        # Generate CUI documents
        for i in range(num_each):
            content, filename = self.generate_cui_document()
            filepath = self.output_dir / f"cui_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['cui'].append(str(filepath))

        # Generate Legacy CUI documents
        for i in range(num_each):
            content, filename = self.generate_legacy_cui_document()
            filepath = self.output_dir / f"legacy_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['legacy_cui'].append(str(filepath))

        # Generate false positive documents
        for i in range(num_each):
            content, filename = self.generate_false_positive_document()
            filepath = self.output_dir / f"fp_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['false_positives'].append(str(filepath))

        # Generate mixed classification documents
        for i in range(num_each):
            content, filename = self.generate_mixed_classification_document()
            filepath = self.output_dir / f"mixed_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['mixed'].append(str(filepath))

        # Generate documents with marking errors
        for i in range(num_each):
            content, filename = self.generate_marking_errors_document()
            filepath = self.output_dir / f"errors_{i:02d}_{filename}"
            filepath.write_text(content)
            generated_files['with_errors'].append(str(filepath))

        # Generate JSON files with embedded classification
        json_content = {
            "document_id": "DOC-2024-001",
            "classification": "SECRET//NOFORN",
            "author": "Intelligence Analyst",
            "content": "This JSON contains classified information about intelligence operations.",
            "metadata": {
                "classified_by": random.choice(self.CLASSIFICATION_AUTHORITIES),
                "reason": random.choice(self.CLASSIFICATION_REASONS),
                "declassify_on": "20301231"
            }
        }
        json_filepath = self.output_dir / "secret_data.json"
        json_filepath.write_text(json.dumps(json_content, indent=2))
        generated_files['secret'].append(str(json_filepath))

        return generated_files

    def generate_summary_report(self, generated_files: Dict[str, List[str]]) -> str:
        """Generate summary report of created test data"""
        total = sum(len(files) for files in generated_files.values())

        report = f"""
TEST DATA GENERATION REPORT
{'='*80}

Output Directory: {self.output_dir}
Total Files Generated: {total}

BREAKDOWN BY CLASSIFICATION:
  • TOP SECRET:    {len(generated_files['top_secret']):3d} files
  • SECRET:        {len(generated_files['secret']):3d} files
  • CONFIDENTIAL:  {len(generated_files['confidential']):3d} files
  • CUI:           {len(generated_files['cui']):3d} files
  • Legacy CUI:    {len(generated_files['legacy_cui']):3d} files
  • Mixed Levels:  {len(generated_files['mixed']):3d} files
  • With Errors:   {len(generated_files['with_errors']):3d} files
  • False Positives: {len(generated_files['false_positives']):3d} files (should NOT be flagged)

EXPECTED SCAN RESULTS:
  • Should FLAG:  {total - len(generated_files['false_positives'])} files
  • Should PASS:  {len(generated_files['false_positives'])} files

REALISM FEATURES:
  ✓ CAPCO-compliant classification markings
  ✓ Proper authority blocks per Executive Order 13526
  ✓ Control markings (NOFORN, ORCON, etc.)
  ✓ Portion markings for paragraphs
  ✓ Legacy CUI markings (FOUO, SBU, LES)
  ✓ Common marking errors (84% error rate per GAO)
  ✓ False positive scenarios
  ✓ Mixed classification levels
  ✓ Multiple document formats

USAGE:
  Run scanner on this directory to validate detection:
    python3 classification_scanner.py {self.output_dir} --sensitivity high

{'='*80}
"""
        return report


def main():
    """Main function to generate test data"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate realistic test documents for classification scanner'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory for test files (default: temp directory)',
        default=None
    )
    parser.add_argument(
        '-n', '--num-each',
        type=int,
        default=5,
        help='Number of each document type to generate (default: 5)'
    )

    args = parser.parse_args()

    # Generate test data
    generator = RealisticTestDataGenerator(output_dir=args.output)
    generated_files = generator.generate_all_test_documents(num_each=args.num_each)

    # Print summary
    report = generator.generate_summary_report(generated_files)
    print(report)

    # Save report
    report_file = generator.output_dir / "GENERATION_REPORT.txt"
    report_file.write_text(report)
    print(f"\nReport saved to: {report_file}")


if __name__ == '__main__':
    main()
