#!/usr/bin/env python3
"""
File Manager for Classification Scanner

This module handles file operations including moving flagged files
to output directories with proper error handling and collision management.

Author: Claude Code
Date: 2025-01-05
Version: 2.2
"""

import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from config import TIMESTAMP_FORMAT


class FileManager:
    """Manages file operations for the classification scanner"""

    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize FileManager

        Args:
            source_dir: Source directory path
            output_dir: Output directory path for flagged files
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

    def validate_directories(self) -> tuple[bool, Optional[str]]:
        """
        Validate source and output directories

        Returns:
            tuple: (is_valid, error_message)
        """
        if not self.source_dir.exists():
            return False, "Source directory does not exist"

        if not self.output_dir.exists():
            return False, "Output directory does not exist"

        if self.source_dir.resolve() == self.output_dir.resolve():
            return False, "Output directory cannot be the same as source directory"

        return True, None

    def move_files(self, file_results: List) -> Dict:
        """
        Move flagged files to output directory

        Args:
            file_results: List of ScanResult objects with has_markings=True

        Returns:
            dict: Move report with statistics and errors
        """
        move_report = {
            'total': len(file_results),
            'moved': 0,
            'failed': 0,
            'errors': [],
            'moved_files': []
        }

        for result in file_results:
            try:
                source_path = Path(result.file_path)
                success, dest_path, error = self._move_single_file(source_path)

                if success:
                    move_report['moved'] += 1
                    move_report['moved_files'].append({
                        'source': str(source_path),
                        'destination': str(dest_path)
                    })
                else:
                    move_report['failed'] += 1
                    move_report['errors'].append({
                        'file': str(source_path),
                        'error': error
                    })

            except Exception as e:
                move_report['failed'] += 1
                move_report['errors'].append({
                    'file': str(result.file_path),
                    'error': str(e)
                })

        return move_report

    def _move_single_file(self, source_path: Path) -> tuple[bool, Optional[Path], Optional[str]]:
        """
        Move a single file to output directory

        Args:
            source_path: Path to source file

        Returns:
            tuple: (success, destination_path, error_message)
        """
        try:
            # Calculate relative path to preserve directory structure
            try:
                rel_path = source_path.relative_to(self.source_dir)
            except ValueError:
                # File is not under source_dir, just use filename
                rel_path = source_path.name

            # Create destination path
            dest_path = self.output_dir / rel_path

            # Security: Validate destination path to prevent directory traversal
            try:
                resolved_dest = dest_path.resolve()
                resolved_output = self.output_dir.resolve()

                # Ensure the resolved destination is within the output directory
                if not str(resolved_dest).startswith(str(resolved_output)):
                    return False, None, "Path traversal detected - destination outside output directory"
            except Exception as e:
                return False, None, f"Path validation error: {str(e)}"

            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic file move with collision handling (fixes TOCTOU race condition)
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    # Re-validate path before each attempt (after collision handling)
                    resolved_dest = dest_path.resolve()
                    if not str(resolved_dest).startswith(str(resolved_output)):
                        return False, None, "Path traversal detected - destination outside output directory"

                    # Try to move the file atomically
                    # Note: shutil.move uses os.rename when on same filesystem, which is atomic
                    if dest_path.exists():
                        # File exists, handle collision before retrying
                        dest_path = self._handle_collision(dest_path)
                        continue

                    shutil.move(str(source_path), str(dest_path))
                    break  # Success

                except FileExistsError:
                    # File was created between check and move - handle collision and retry
                    dest_path = self._handle_collision(dest_path)
                    if attempt == max_retries - 1:
                        return False, None, "Failed to move file after multiple collision retries"
            else:
                return False, None, "Failed to find unique filename after maximum retries"

            return True, dest_path, None

        except PermissionError as e:
            return False, None, f"Permission denied: {str(e)}"
        except OSError as e:
            return False, None, f"OS error: {str(e)}"
        except Exception as e:
            return False, None, str(e)

    def _handle_collision(self, dest_path: Path) -> Path:
        """
        Handle file name collision by adding timestamp

        Args:
            dest_path: Original destination path

        Returns:
            Path: New destination path with timestamp
        """
        timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
        stem = dest_path.stem
        suffix = dest_path.suffix
        return dest_path.parent / f"{stem}_{timestamp}{suffix}"

    @staticmethod
    def format_move_report(move_report: Dict) -> str:
        """
        Format move report as human-readable text

        Args:
            move_report: Move report dictionary

        Returns:
            str: Formatted report text
        """
        lines = []
        lines.append("File Move Report")
        lines.append("=" * 60)
        lines.append(f"Total files: {move_report['total']}")
        lines.append(f"Successfully moved: {move_report['moved']}")
        lines.append(f"Failed: {move_report['failed']}")

        if move_report['errors']:
            lines.append("\nErrors:")
            for error in move_report['errors'][:5]:
                file_name = Path(error['file']).name
                lines.append(f"  • {file_name}: {error['error']}")
            if len(move_report['errors']) > 5:
                lines.append(f"  ... and {len(move_report['errors']) - 5} more errors")

        return "\n".join(lines)

    @staticmethod
    def create_move_summary(move_report: Dict, max_errors: int = 5) -> str:
        """
        Create a summary message for display in message box

        Args:
            move_report: Move report dictionary
            max_errors: Maximum number of errors to display

        Returns:
            str: Summary message
        """
        summary = f"File Move Complete\n\n"
        summary += f"Total files: {move_report['total']}\n"
        summary += f"Successfully moved: {move_report['moved']}\n"
        summary += f"Failed: {move_report['failed']}\n"

        if move_report['errors']:
            summary += "\nErrors:\n"
            for error in move_report['errors'][:max_errors]:
                summary += f"• {Path(error['file']).name}: {error['error']}\n"
            if len(move_report['errors']) > max_errors:
                summary += f"... and {len(move_report['errors']) - max_errors} more errors\n"

        return summary


def validate_output_directory(output_dir: str, source_dir: str) -> tuple[bool, Optional[str]]:
    """
    Validate output directory without creating FileManager instance

    Args:
        output_dir: Output directory path
        source_dir: Source directory path

    Returns:
        tuple: (is_valid, error_message)
    """
    output_path = Path(output_dir)
    source_path = Path(source_dir)

    if not output_path.exists():
        return False, "Output directory does not exist"

    if source_path.resolve() == output_path.resolve():
        return False, "Output directory cannot be the same as source directory"

    return True, None
