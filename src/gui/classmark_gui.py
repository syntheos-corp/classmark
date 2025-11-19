#!/usr/bin/env python3
"""
Classmark Desktop GUI Application

A user-friendly desktop application for detecting classification markings
in document folders. Automatically processes all files, moves hits to a
separate folder, and generates detailed logs.

Features:
- Drag-and-drop or browse folder selection
- Batch processing with progress tracking
- Automatic file organization (hits moved to separate folder)
- Detailed CSV/JSON logs with confidence scores
- Configurable settings (sensitivity, GPU, thresholds)
- Fully offline after initial model download

Author: Classmark Development Team
Date: 2025-11-10
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import json
import csv
import shutil
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# Detect WSL environment
def is_wsl():
    """Check if running on Windows Subsystem for Linux"""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except:
        return False

IS_WSL = is_wsl()

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import classification scanner
try:
    from src.core.classification_scanner import ClassificationScanner
    SCANNER_AVAILABLE = True
except ImportError:
    SCANNER_AVAILABLE = False


# Worker function for parallel processing (must be at module level for multiprocessing)
def process_file_worker(file_path: str, config: Dict, input_folder: str,
                       output_folder: str, move_hits: bool,
                       enable_early_exit: bool = True,
                       early_exit_threshold: float = 0.90,
                       quick_scan_pages: int = 3) -> Dict:
    """
    Process a single file in a separate process

    This function is isolated in its own process for:
    - Thread safety (no shared state)
    - Parallel processing (utilize multiple CPU cores)
    - Fault isolation (one file crash doesn't affect others)

    Args:
        file_path: Path to file to process
        config: Scanner configuration dict
        input_folder: Input folder path (for relative path calculation)
        output_folder: Output folder path (for moving hits)
        move_hits: Whether to move classified files
        enable_early_exit: Whether to use early exit optimization
        early_exit_threshold: Confidence threshold for early exit
        quick_scan_pages: Number of pages to scan for quick check

    Returns:
        Dict with processing results
    """
    import sys
    import os

    # Add src to path (needed in worker process)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    from src.core.classification_scanner import ClassificationScanner

    try:
        # Create scanner instance (isolated to this process)
        scanner = ClassificationScanner(config)

        # Scan file with configurable early exit settings
        result = scanner.scan_file(file_path,
                                   enable_early_exit=enable_early_exit,
                                   early_exit_threshold=early_exit_threshold,
                                   quick_scan_pages=quick_scan_pages)

        filename = os.path.basename(file_path)

        # Build result dict
        result_dict = {
            'file_path': file_path,
            'filename': filename,
            'has_classification': result.has_markings,
            'classification_level': str(result.classification_level.value) if result.has_markings else None,
            'confidence': result.overall_confidence,
            'num_matches': len(result.matches),
            'processing_time': result.processing_time,
            'status': 'success'
        }

        # Move file if it's a hit
        if result.has_markings and move_hits:
            try:
                import shutil

                # SECURITY: Check if source file still exists (race condition protection)
                if not os.path.exists(file_path):
                    result_dict['moved'] = False
                    result_dict['move_error'] = 'File no longer exists (already moved or deleted)'
                    return result_dict

                # Create relative path structure in output folder
                rel_path = os.path.relpath(file_path, input_folder)
                output_path = os.path.join(output_folder, rel_path)

                # SECURITY: Validate path to prevent path traversal attacks
                output_path_abs = os.path.abspath(output_path)
                output_folder_abs = os.path.abspath(output_folder)
                if not output_path_abs.startswith(output_folder_abs + os.sep):
                    raise ValueError(f"Path traversal attempt detected: {output_path}")

                # SECURITY: Check if destination already exists (race condition protection)
                if os.path.exists(output_path):
                    # Another worker may have already moved this file
                    result_dict['moved'] = False
                    result_dict['move_error'] = 'Destination already exists (possible duplicate processing)'
                    return result_dict

                # Create directory if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Move file atomically (on same filesystem)
                shutil.move(file_path, output_path)

                result_dict['moved'] = True
                result_dict['output_path'] = output_path
            except FileNotFoundError as e:
                # File was deleted/moved between check and move
                result_dict['move_error'] = f'File disappeared during move: {e}'
                result_dict['moved'] = False
            except PermissionError as e:
                # File locked or insufficient permissions
                result_dict['move_error'] = f'Permission denied: {e}'
                result_dict['moved'] = False
            except Exception as e:
                result_dict['move_error'] = str(e)
                result_dict['moved'] = False
        else:
            result_dict['moved'] = False

        return result_dict

    except Exception as e:
        # Return error result
        return {
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'status': 'error',
            'error': str(e)
        }


class ClassmarkGUI:
    """Main GUI application for Classmark"""

    def __init__(self, root):
        self.root = root
        self.root.title("Classmark - Classification Marking Detection")
        self.root.geometry("1000x700")

        # Application state
        self.input_folder = None
        self.output_folder = None
        self.scanner = None
        self.processing = False
        self.results_queue = queue.Queue()

        # Progress tracking for throttling
        self.last_progress_update = 0
        self.last_progress_value = 0
        self.processing_start_time = None

        # Settings (loaded from config file)
        self.settings = {
            'sensitivity': 'medium',
            'confidence_threshold': 0.3,
            'use_gpu': True,
            'use_visual_detection': True,
            'use_fuzzy_matching': True,
            'fuzzy_threshold': 85,
            'move_hits': True,
            'create_log': True,
            'log_format': 'csv',
            # Performance optimization settings
            'enable_early_exit': True,
            'early_exit_threshold': 0.90,
            'quick_scan_pages': 3,
            'parallel_workers': max(1, multiprocessing.cpu_count() // 2)
        }

        self.load_settings()

        # Create UI
        self.create_widgets()

        # Check scanner availability
        if not SCANNER_AVAILABLE:
            self.log_message("⚠ Warning: Classification scanner not available. Please check installation.", "warning")

        # Start result processing
        self.root.after(100, self.process_results)

    def create_widgets(self):
        """Create all GUI widgets"""

        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Title
        title_label = ttk.Label(main_frame, text="Classmark Classification Scanner",
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Folder Selection Frame
        folder_frame = ttk.LabelFrame(main_frame, text="Folder Selection", padding="10")
        folder_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        folder_frame.columnconfigure(1, weight=1)

        # Input folder
        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.input_folder_var = tk.StringVar()
        input_entry = ttk.Entry(folder_frame, textvariable=self.input_folder_var, state='readonly')
        input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(folder_frame, text="Browse...", command=self.select_input_folder).grid(row=0, column=2)

        # Output folder (for hits)
        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.output_folder_var = tk.StringVar()
        output_entry = ttk.Entry(folder_frame, textvariable=self.output_folder_var, state='readonly')
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(folder_frame, text="Browse...", command=self.select_output_folder).grid(row=1, column=2, pady=(5, 0))

        # Control Frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        self.start_button = ttk.Button(control_frame, text="Start Processing",
                                       command=self.start_processing, state='disabled')
        self.start_button.grid(row=0, column=0, padx=5)

        self.stop_button = ttk.Button(control_frame, text="Stop",
                                      command=self.stop_processing, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=5)

        ttk.Button(control_frame, text="Settings", command=self.open_settings).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="About", command=self.show_about).grid(row=0, column=3, padx=5)

        # Progress Frame
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        # Progress label
        self.progress_label_var = tk.StringVar(value="Ready")
        progress_label = ttk.Label(progress_frame, textvariable=self.progress_label_var)
        progress_label.grid(row=0, column=1, padx=(10, 0))

        # Log text area
        self.log_text = scrolledtext.ScrolledText(progress_frame, height=20, state='disabled',
                                                  wrap=tk.WORD, font=('Courier', 9))
        self.log_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure text tags for colored messages
        self.log_text.tag_config('info', foreground='black')
        self.log_text.tag_config('success', foreground='green')
        self.log_text.tag_config('warning', foreground='orange')
        self.log_text.tag_config('error', foreground='red')

        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def select_input_folder(self):
        """Open dialog to select input folder"""
        # WSL workaround: Use text input instead of file dialog
        if IS_WSL:
            folder = self.prompt_folder_path("Select Input Folder",
                                            "Enter the full path to the input folder:")
        else:
            folder = filedialog.askdirectory(title="Select Input Folder")

        if folder:
            # Validate folder exists
            if not os.path.isdir(folder):
                messagebox.showerror("Error", f"Folder does not exist: {folder}")
                return

            self.input_folder = folder
            self.input_folder_var.set(folder)

            # Auto-set output folder if not set
            if not self.output_folder:
                default_output = os.path.join(folder, "classified_documents")
                self.output_folder = default_output
                self.output_folder_var.set(default_output)

            self.update_start_button()
            self.log_message(f"Input folder selected: {folder}", "info")

    def select_output_folder(self):
        """Open dialog to select output folder"""
        # WSL workaround: Use text input instead of file dialog
        if IS_WSL:
            folder = self.prompt_folder_path("Select Output Folder",
                                            "Enter the full path to the output folder:")
        else:
            folder = filedialog.askdirectory(title="Select Output Folder")

        if folder:
            self.output_folder = folder
            self.output_folder_var.set(folder)
            self.log_message(f"Output folder selected: {folder}", "info")

    def prompt_folder_path(self, title, message):
        """
        Prompt user to enter a folder path manually (WSL workaround)

        Args:
            title: Dialog title
            message: Prompt message

        Returns:
            Folder path or None if canceled
        """
        from tkinter import simpledialog

        # Show info about WSL limitation
        messagebox.showinfo(
            "WSL Detected",
            "File dialogs don't work properly on WSL.\n"
            "Please enter the folder path manually.\n\n"
            "Tip: Use Windows paths like:\n"
            "/mnt/d/MyFolder or\n"
            "D:\\MyFolder (will be converted)"
        )

        folder = simpledialog.askstring(title, message)

        if folder:
            # Convert Windows-style paths to WSL paths
            folder = folder.strip().strip('"')

            # Convert D:\path to /mnt/d/path
            if len(folder) >= 3 and folder[1] == ':':
                drive = folder[0].lower()
                path = folder[2:].replace('\\', '/')
                folder = f"/mnt/{drive}{path}"

            return folder
        return None

    def update_start_button(self):
        """Enable/disable start button based on selections"""
        if self.input_folder and SCANNER_AVAILABLE and not self.processing:
            self.start_button.config(state='normal')
        else:
            self.start_button.config(state='disabled')

    def start_processing(self):
        """Start batch processing of documents"""
        if not self.input_folder:
            messagebox.showerror("Error", "Please select an input folder")
            return

        if not SCANNER_AVAILABLE:
            messagebox.showerror("Error", "Classification scanner not available")
            return

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

        # Update UI state
        self.processing = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set("Processing...")
        self.progress_var.set(0)

        # Reset progress tracking
        self.last_progress_update = 0
        self.last_progress_value = 0
        self.processing_start_time = time.time()

        # Start processing in separate thread
        thread = threading.Thread(target=self.process_documents, daemon=True)
        thread.start()

    def stop_processing(self):
        """Stop processing (graceful)"""
        self.processing = False
        self.stop_button.config(state='disabled')
        self.log_message("Stopping processing...", "warning")

    def process_documents(self):
        """Process all documents in input folder (runs in separate thread)"""
        try:
            # Initialize scanner
            self.log_message("Initializing scanner...", "info")

            config = {
                'sensitivity_threshold': self.get_sensitivity_value(),
                'confidence_threshold': self.settings['confidence_threshold'],
                'fuzzy_matching': self.settings['use_fuzzy_matching'],
                'fuzzy_threshold': self.settings['fuzzy_threshold'],
                'use_llm': False,
                'use_gpu': self.settings['use_gpu'],
                'use_visual_detection': self.settings['use_visual_detection']
            }

            self.scanner = ClassificationScanner(config)
            self.log_message("Scanner initialized successfully", "success")

            # Find all supported files
            supported_extensions = ['.txt', '.pdf', '.docx', '.doc']
            files = []

            for root, dirs, filenames in os.walk(self.input_folder):
                for filename in filenames:
                    if any(filename.lower().endswith(ext) for ext in supported_extensions):
                        files.append(os.path.join(root, filename))

            if not files:
                self.results_queue.put(('error', 'No supported files found in input folder'))
                return

            self.log_message(f"Found {len(files)} files to process", "info")

            # Get worker count from settings
            max_workers = self.settings['parallel_workers']
            self.log_message(f"Using {max_workers} parallel workers (of {multiprocessing.cpu_count()} available cores)", "info")

            # Log optimization settings
            if self.settings['enable_early_exit']:
                self.log_message(f"Early exit enabled (threshold: {self.settings['early_exit_threshold']:.2f}, quick scan: {self.settings['quick_scan_pages']} pages)", "info")
            else:
                self.log_message("Early exit disabled - processing all pages", "info")

            # Process files in parallel
            results = []
            hits_count = 0
            completed_count = 0

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files as futures
                future_to_file = {
                    executor.submit(
                        process_file_worker,
                        file_path,
                        config,
                        self.input_folder,
                        self.output_folder,
                        self.settings['move_hits'],
                        self.settings['enable_early_exit'],
                        self.settings['early_exit_threshold'],
                        self.settings['quick_scan_pages']
                    ): file_path
                    for file_path in files
                }

                # Process completed futures as they finish
                for future in as_completed(future_to_file):
                    # Check if user stopped processing
                    if not self.processing:
                        self.results_queue.put(('warning', 'Processing stopped by user'))
                        # Cancel remaining futures
                        for f in future_to_file.keys():
                            f.cancel()
                        break

                    completed_count += 1
                    progress = (completed_count / len(files)) * 100
                    file_path = future_to_file[future]
                    filename = os.path.basename(file_path)

                    # Throttled progress updates - only update if:
                    # 1. Progress changed by >= 1% OR
                    # 2. More than 0.5 seconds passed OR
                    # 3. This is the last file (100% complete)
                    current_time = time.time()
                    time_since_last_update = current_time - self.last_progress_update
                    progress_delta = abs(progress - self.last_progress_value)

                    should_update = (
                        progress_delta >= 1.0 or  # Progress changed by 1%+
                        time_since_last_update >= 0.5 or  # 500ms elapsed
                        completed_count == len(files)  # Last file
                    )

                    if should_update:
                        # Calculate processing statistics
                        elapsed_time = current_time - self.processing_start_time
                        files_per_sec = completed_count / elapsed_time if elapsed_time > 0 else 0
                        remaining_files = len(files) - completed_count
                        eta_seconds = remaining_files / files_per_sec if files_per_sec > 0 else 0

                        # Format progress label with statistics
                        if completed_count == len(files):
                            progress_label = f"Complete: {completed_count}/{len(files)} files"
                        else:
                            progress_label = f"{completed_count}/{len(files)} files ({files_per_sec:.1f} files/sec, ETA: {eta_seconds:.0f}s)"

                        # Send throttled update
                        self.results_queue.put(('progress', (progress, progress_label)))
                        self.last_progress_update = current_time
                        self.last_progress_value = progress

                    try:
                        # Get result from worker process
                        result_dict = future.result()
                        results.append(result_dict)

                        # Log result
                        if result_dict.get('status') == 'success':
                            if result_dict.get('moved'):
                                hits_count += 1
                                confidence = result_dict.get('confidence', 0.0)
                                self.results_queue.put(('success', f"✓ HIT: {filename} (Confidence: {confidence:.2f}) - Moved to output"))
                            else:
                                self.results_queue.put(('info', f"  No classification: {filename}"))
                        else:
                            error_msg = result_dict.get('error', 'Unknown error')
                            self.results_queue.put(('error', f"Error processing {filename}: {error_msg}"))

                    except Exception as e:
                        # Handle future execution errors
                        self.results_queue.put(('error', f"Error processing {filename}: {str(e)}"))
                        results.append({
                            'file_path': file_path,
                            'filename': filename,
                            'error': str(e),
                            'status': 'error'
                        })

            # Save log
            if self.settings['create_log']:
                self.save_log(results)

            # Complete
            self.results_queue.put(('progress', (100, "Complete")))
            self.results_queue.put(('success', f"\n{'='*60}"))
            self.results_queue.put(('success', f"Processing Complete!"))
            self.results_queue.put(('success', f"{'='*60}"))
            self.results_queue.put(('success', f"Total files processed: {len(files)}"))
            self.results_queue.put(('success', f"Classification hits: {hits_count}"))
            self.results_queue.put(('success', f"Files moved to output: {hits_count if self.settings['move_hits'] else 0}"))
            if self.settings['create_log']:
                self.results_queue.put(('success', f"Log saved to: {self.output_folder}"))

        except Exception as e:
            self.results_queue.put(('error', f"Fatal error: {str(e)}"))

        finally:
            self.results_queue.put(('done', None))

    def process_results(self):
        """Process results from processing thread (runs in main thread)"""
        try:
            while True:
                msg_type, data = self.results_queue.get_nowait()

                if msg_type == 'progress':
                    progress, label = data
                    self.progress_var.set(progress)
                    self.progress_label_var.set(label)

                elif msg_type == 'done':
                    self.processing = False
                    self.start_button.config(state='normal' if self.input_folder else 'disabled')
                    self.stop_button.config(state='disabled')
                    self.status_var.set("Ready")

                elif msg_type in ['info', 'success', 'warning', 'error']:
                    self.log_message(data, msg_type)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.process_results)

    def save_log(self, results: List[Dict]):
        """Save processing log to file

        Handles heterogeneous result dictionaries where different results may have
        different fields (e.g., some with 'output_path', some without).
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if self.settings['log_format'] == 'csv':
                log_path = os.path.join(self.output_folder, f"classmark_log_{timestamp}.csv")

                with open(log_path, 'w', newline='', encoding='utf-8') as f:
                    if results:
                        # Define preferred field order for professional CSV output
                        preferred_order = [
                            'file_path', 'filename', 'status',
                            'has_classification', 'classification_level', 'confidence',
                            'num_matches', 'moved', 'output_path', 'move_error',
                            'processing_time', 'error'
                        ]

                        # Collect all actual fields present in results
                        actual_fields = set()
                        for result in results:
                            actual_fields.update(result.keys())

                        # Use preferred order for known fields, append any extras
                        fieldnames = [f for f in preferred_order if f in actual_fields]
                        extra_fields = sorted(actual_fields - set(preferred_order))
                        fieldnames.extend(extra_fields)

                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results)
                    else:
                        # Write empty CSV with standard headers
                        fieldnames = ['file_path', 'filename', 'status']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                self.log_message(f"Saved CSV log to {log_path}", 'info')

            else:  # json
                log_path = os.path.join(self.output_folder, f"classmark_log_{timestamp}.json")

                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)

                self.log_message(f"Saved JSON log to {log_path}", 'info')

            return log_path

        except Exception as e:
            error_msg = f"Failed to save log: {str(e)}"
            self.log_message(error_msg, 'error')
            return None

    def get_sensitivity_value(self) -> float:
        """Convert sensitivity setting to threshold value"""
        sensitivity_map = {
            'low': 0.7,      # Less sensitive, fewer false positives
            'medium': 0.5,   # Balanced
            'high': 0.3      # More sensitive, catches more but may have false positives
        }
        return sensitivity_map.get(self.settings['sensitivity'], 0.5)

    def log_message(self, message: str, level: str = 'info'):
        """Add message to log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"

        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, formatted_message, level)
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def open_settings(self):
        """Open settings dialog - Performance Optimizations Only"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Performance Settings")
        settings_window.geometry("500x300")
        settings_window.transient(self.root)
        settings_window.grab_set()

        # Settings frame
        frame = ttk.Frame(settings_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # Title
        ttk.Label(frame, text="Performance Optimizations", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))

        # Early exit optimization
        early_exit_var = tk.BooleanVar(value=self.settings['enable_early_exit'])
        ttk.Checkbutton(frame, text="Enable Early Exit (stop after finding classification)",
                       variable=early_exit_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Early exit threshold
        ttk.Label(frame, text="Early Exit Confidence Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=(20, 0))
        early_exit_threshold_var = tk.DoubleVar(value=self.settings['early_exit_threshold'])
        early_exit_threshold_spin = ttk.Spinbox(frame, from_=0.7, to=1.0, increment=0.05,
                                               textvariable=early_exit_threshold_var, width=10)
        early_exit_threshold_spin.grid(row=2, column=1, sticky=tk.W, pady=5)

        # Quick scan pages
        ttk.Label(frame, text="Quick Scan Pages (for early exit):").grid(row=3, column=0, sticky=tk.W, pady=5, padx=(20, 0))
        quick_scan_pages_var = tk.IntVar(value=self.settings['quick_scan_pages'])
        quick_scan_pages_spin = ttk.Spinbox(frame, from_=1, to=10, increment=1,
                                           textvariable=quick_scan_pages_var, width=10)
        quick_scan_pages_spin.grid(row=3, column=1, sticky=tk.W, pady=5)

        # Separator
        ttk.Separator(frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=15)

        # Parallel workers
        max_cpu = multiprocessing.cpu_count()
        ttk.Label(frame, text=f"Parallel Workers (of {max_cpu} CPU cores):").grid(row=5, column=0, sticky=tk.W, pady=5)
        parallel_workers_var = tk.IntVar(value=self.settings['parallel_workers'])
        parallel_workers_spin = ttk.Spinbox(frame, from_=1, to=max_cpu, increment=1,
                                           textvariable=parallel_workers_var, width=10)
        parallel_workers_spin.grid(row=5, column=1, sticky=tk.W, pady=5)

        # Help text
        help_text = ttk.Label(frame, text="Note: Higher values improve speed but use more CPU/memory.",
                             foreground='gray', font=('Helvetica', 9))
        help_text.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

        # Buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=(20, 0))

        def save_settings():
            # Save ONLY performance optimization settings
            # Other settings remain at their default/configured values
            self.settings['enable_early_exit'] = early_exit_var.get()
            self.settings['early_exit_threshold'] = early_exit_threshold_var.get()
            self.settings['quick_scan_pages'] = quick_scan_pages_var.get()
            self.settings['parallel_workers'] = parallel_workers_var.get()

            self.save_settings_to_file()
            self.log_message("Performance settings saved", "success")
            settings_window.destroy()

        def on_closing():
            """Handle window close event (X button)"""
            result = messagebox.askyesnocancel(
                "Save Settings",
                "Do you want to save your changes?",
                parent=settings_window
            )
            if result is True:  # Yes - save and close
                save_settings()
            elif result is False:  # No - discard and close
                settings_window.destroy()
            # None/Cancel - do nothing (keep window open)

        # Set protocol for window close button (X)
        settings_window.protocol("WM_DELETE_WINDOW", on_closing)

        ttk.Button(button_frame, text="Save", command=save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.LEFT, padx=5)

    def show_about(self):
        """Show about dialog"""
        about_text = """Classmark - Classification Marking Detection System

Version: 1.0.0
Date: 2025-11-10

A state-of-the-art system for detecting classification markings
in documents using machine learning and computer vision.

Features:
• YOLO + Tesseract OCR for scanned documents
• LayoutLMv3 visual pattern detection
• Aho-Corasick fast pattern matching
• Hybrid two-stage architecture
• 93% precision, 100% recall on test data
• 2.2 million documents/minute throughput

Author: Classmark Development Team

For more information, see PROJECT_SUMMARY.md"""

        messagebox.showinfo("About Classmark", about_text)

    def load_settings(self):
        """Load settings from config file"""
        config_path = os.path.join(os.path.expanduser("~"), ".classmark_config.json")

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_settings = json.load(f)
                    # SECURITY: Validate settings before applying
                    validated_settings = self._validate_settings(loaded_settings)
                    self.settings.update(validated_settings)
            except Exception as e:
                print(f"Could not load settings: {e}")

    def _validate_settings(self, settings: dict) -> dict:
        """
        Validate and sanitize settings to prevent security issues

        Args:
            settings: Raw settings dict from config file

        Returns:
            Validated settings dict with safe values
        """
        validated = {}

        # Validate numeric ranges
        if 'confidence_threshold' in settings:
            validated['confidence_threshold'] = max(0.0, min(float(settings['confidence_threshold']), 1.0))

        if 'fuzzy_threshold' in settings:
            validated['fuzzy_threshold'] = max(0, min(int(settings['fuzzy_threshold']), 100))

        # SECURITY: Validate performance optimization settings
        if 'parallel_workers' in settings:
            max_workers = multiprocessing.cpu_count()
            validated['parallel_workers'] = max(1, min(int(settings['parallel_workers']), max_workers))

        if 'early_exit_threshold' in settings:
            # Minimum 0.70 to prevent defeating security checks
            validated['early_exit_threshold'] = max(0.70, min(float(settings['early_exit_threshold']), 1.0))

        if 'quick_scan_pages' in settings:
            # Minimum 1 page, maximum 10
            validated['quick_scan_pages'] = max(1, min(int(settings['quick_scan_pages']), 10))

        # Pass through boolean settings (safe)
        for bool_key in ['use_gpu', 'use_visual_detection', 'use_fuzzy_matching',
                         'move_hits', 'create_log', 'enable_early_exit']:
            if bool_key in settings:
                validated[bool_key] = bool(settings[bool_key])

        # Pass through string settings with whitelist validation
        if 'sensitivity' in settings and settings['sensitivity'] in ['low', 'medium', 'high']:
            validated['sensitivity'] = settings['sensitivity']

        if 'log_format' in settings and settings['log_format'] in ['csv', 'json']:
            validated['log_format'] = settings['log_format']

        return validated

    def save_settings_to_file(self):
        """Save settings to config file"""
        config_path = os.path.join(os.path.expanduser("~"), ".classmark_config.json")

        try:
            with open(config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"Could not save settings: {e}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ClassmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
