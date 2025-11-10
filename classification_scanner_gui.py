#!/usr/bin/env python3
"""
Classification Marking Scanner - Desktop GUI
User-friendly desktop application for scanning documents for classification markings.

Author: Claude Code
Date: 2025-01-05
Version: 2.2 Desktop Edition
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, simpledialog
import threading
import queue
import os
import sys
from pathlib import Path
import json
from datetime import datetime
import platform
import subprocess

# Import configuration
from config import *

# Import GUI components
from gui_components import (
    create_tooltip,
    create_scrolled_text_dialog,
    StatusBar
)

# Import file manager
from file_manager import FileManager, validate_output_directory

# Import LLM installer
from llm_installer import LLMInstaller

# Import the scanner backend
from classification_scanner import (
    ClassificationScanner,
    ReportGenerator,
    ClassificationLevel
)


class ScannerGUI:
    """Main GUI application for classification scanner"""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)

        # Set minimum size
        self.root.minsize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        # Variables
        self.selected_dir = tk.StringVar()
        self.recursive = tk.BooleanVar(value=DEFAULT_RECURSIVE)
        self.fuzzy_matching = tk.BooleanVar(value=DEFAULT_FUZZY_MATCHING)
        self.use_llm = tk.BooleanVar(value=DEFAULT_USE_LLM)
        self.sensitivity = tk.StringVar(value=DEFAULT_SENSITIVITY)
        self.workers = tk.IntVar(value=WORKERS_DEFAULT)
        self.move_flagged = tk.BooleanVar(value=DEFAULT_MOVE_FLAGGED)
        self.output_dir = tk.StringVar()

        # Scanning state
        self.is_scanning = False
        self.scan_lock = threading.Lock()  # Protects is_scanning flag
        self.scan_results = []
        self.progress_queue = queue.Queue()

        # LLM state
        self.ollama_installed = tk.BooleanVar(value=False)
        self.model_downloaded = tk.BooleanVar(value=False)
        self.llm_status = tk.StringVar(value="Checking...")
        self.is_downloading = False

        # Initialize LLM installer
        self.llm_installer = LLMInstaller(
            progress_callback=lambda msg: self.progress_queue.put(("llm_status", msg))
        )

        # File manager (initialized when needed)
        self.file_manager = None

        # Create menu bar
        self.create_menu()

        # Create UI
        self.create_widgets()

        # Check LLM status on startup
        self.check_llm_status()

        # Start progress checker
        self.check_progress()

    def create_menu(self):
        """Create menu bar with Help menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Quick Start Guide", command=self.show_quick_start)
        help_menu.add_separator()
        help_menu.add_command(label="View README", command=self.open_readme)
        help_menu.add_command(label="View Build Notes", command=self.open_build_notes)
        help_menu.add_separator()
        help_menu.add_command(label="Report Issue", command=self.report_issue)

    def show_about(self):
        """Show About dialog"""
        messagebox.showinfo("About Classification Scanner", TEXT_ABOUT)

    def show_quick_start(self):
        """Show Quick Start Guide dialog"""
        create_scrolled_text_dialog(
            self.root,
            "Quick Start Guide",
            TEXT_QUICK_START
        )

    def open_readme(self):
        """Open README file"""
        readme_path = Path(__file__).parent / "README.md"
        if readme_path.exists():
            try:
                if platform.system() == "Windows":
                    os.startfile(readme_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(['open', str(readme_path)])
                else:  # Linux
                    subprocess.Popen(['xdg-open', str(readme_path)])
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not open README file:\n{str(e)}\n\nFile location:\n{readme_path}"
                )
        else:
            messagebox.showwarning(
                "File Not Found",
                f"README.md not found at:\n{readme_path}"
            )

    def open_build_notes(self):
        """Open BUILD_NOTES file"""
        build_notes_path = Path(__file__).parent / "BUILD_NOTES.md"
        if build_notes_path.exists():
            try:
                if platform.system() == "Windows":
                    os.startfile(build_notes_path)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.Popen(['open', str(build_notes_path)])
                else:  # Linux
                    subprocess.Popen(['xdg-open', str(build_notes_path)])
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Could not open BUILD_NOTES file:\n{str(e)}\n\nFile location:\n{build_notes_path}"
                )
        else:
            messagebox.showwarning(
                "File Not Found",
                f"BUILD_NOTES.md not found at:\n{build_notes_path}"
            )

    def report_issue(self):
        """Open issue reporting information"""
        messagebox.showinfo("Report Issue", TEXT_REPORT_ISSUE)

    def create_widgets(self):
        """Create all UI widgets"""

        # Header
        header = ttk.Frame(self.root, padding="10")
        header.pack(fill=tk.X)

        title_label = ttk.Label(
            header,
            text="US Government Classification Marking Scanner",
            font=("Arial", 16, "bold")
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            header,
            text="Scans PDF, DOCX, TXT, and JSON files for security classification markings",
            font=("Arial", 9)
        )
        subtitle_label.pack()

        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Directory Selection ===
        dir_frame = ttk.LabelFrame(main_frame, text="Directory Selection", padding="10")
        dir_frame.pack(fill=tk.X, pady=(0, 10))

        dir_entry_frame = ttk.Frame(dir_frame)
        dir_entry_frame.pack(fill=tk.X)

        dir_entry = ttk.Entry(
            dir_entry_frame,
            textvariable=self.selected_dir,
            state="readonly",
            width=60
        )
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        create_tooltip(dir_entry, TOOLTIPS['dir_entry'])

        browse_btn = ttk.Button(
            dir_entry_frame,
            text="Browse...",
            command=self.browse_directory
        )
        browse_btn.pack(side=tk.LEFT)
        create_tooltip(browse_btn, TOOLTIPS['browse_btn'])

        # === Output Directory Selection ===
        output_frame = ttk.LabelFrame(main_frame, text="Output Directory (Optional)", padding="10")
        output_frame.pack(fill=tk.X, pady=(0, 10))

        # Move flagged files checkbox
        move_cb_frame = ttk.Frame(output_frame)
        move_cb_frame.pack(fill=tk.X, pady=(0, 5))

        self.move_flagged_cb = ttk.Checkbutton(
            move_cb_frame,
            text="Move flagged files to output folder",
            variable=self.move_flagged,
            command=self.toggle_output_dir
        )
        self.move_flagged_cb.pack(side=tk.LEFT)
        create_tooltip(self.move_flagged_cb, TOOLTIPS['move_flagged_cb'])

        # Output directory entry
        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill=tk.X)

        self.output_entry = ttk.Entry(
            output_entry_frame,
            textvariable=self.output_dir,
            state="disabled",
            width=60
        )
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        create_tooltip(self.output_entry, TOOLTIPS['output_entry'])

        self.output_browse_btn = ttk.Button(
            output_entry_frame,
            text="Browse...",
            command=self.browse_output_directory,
            state=tk.DISABLED
        )
        self.output_browse_btn.pack(side=tk.LEFT)
        create_tooltip(self.output_browse_btn, TOOLTIPS['output_browse_btn'])

        # === Scan Options ===
        options_frame = ttk.LabelFrame(main_frame, text="Scan Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        # First row of options
        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=(0, 5))

        recursive_cb = ttk.Checkbutton(
            options_row1,
            text="Scan subdirectories recursively",
            variable=self.recursive
        )
        recursive_cb.pack(side=tk.LEFT, padx=(0, 20))
        create_tooltip(recursive_cb, TOOLTIPS['recursive_cb'])

        fuzzy_cb = ttk.Checkbutton(
            options_row1,
            text="Enable fuzzy matching",
            variable=self.fuzzy_matching
        )
        fuzzy_cb.pack(side=tk.LEFT, padx=(0, 20))
        create_tooltip(fuzzy_cb, TOOLTIPS['fuzzy_cb'])

        llm_cb = ttk.Checkbutton(
            options_row1,
            text="Use LLM verification (requires Ollama)",
            variable=self.use_llm
        )
        llm_cb.pack(side=tk.LEFT)
        create_tooltip(llm_cb, TOOLTIPS['llm_cb'])

        # Second row of options
        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill=tk.X)

        ttk.Label(options_row2, text="Sensitivity:").pack(side=tk.LEFT, padx=(0, 5))
        sensitivity_combo = ttk.Combobox(
            options_row2,
            textvariable=self.sensitivity,
            values=["low", "medium", "high"],
            state="readonly",
            width=10
        )
        sensitivity_combo.pack(side=tk.LEFT, padx=(0, 20))
        create_tooltip(sensitivity_combo, TOOLTIPS['sensitivity_combo'])

        ttk.Label(options_row2, text="Parallel Workers:").pack(side=tk.LEFT, padx=(0, 5))
        workers_spin = ttk.Spinbox(
            options_row2,
            from_=WORKERS_MIN,
            to=WORKERS_MAX,
            textvariable=self.workers,
            width=5
        )
        workers_spin.pack(side=tk.LEFT)
        create_tooltip(workers_spin, TOOLTIPS['workers_spin'])

        # === LLM Setup ===
        llm_frame = ttk.LabelFrame(main_frame, text="LLM Setup (Optional)", padding="10")
        llm_frame.pack(fill=tk.X, pady=(0, 10))

        # Status row
        llm_status_row = ttk.Frame(llm_frame)
        llm_status_row.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(llm_status_row, text="Status:").pack(side=tk.LEFT, padx=(0, 5))
        self.llm_status_label = ttk.Label(
            llm_status_row,
            textvariable=self.llm_status,
            font=("Arial", 9, "italic")
        )
        self.llm_status_label.pack(side=tk.LEFT)

        # Model info row (static)
        model_info_row = ttk.Frame(llm_frame)
        model_info_row.pack(fill=tk.X, pady=(5, 5))

        ttk.Label(
            model_info_row,
            text=f"Model: {LLM_MODEL_NAME}",
            font=("Arial", 9)
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(
            model_info_row,
            text="(~5 GB download)",
            font=("Arial", 8),
            foreground="gray"
        ).pack(side=tk.LEFT)

        # Buttons row
        llm_buttons_row = ttk.Frame(llm_frame)
        llm_buttons_row.pack(fill=tk.X)

        self.install_ollama_btn = ttk.Button(
            llm_buttons_row,
            text="Install Ollama",
            command=self.install_ollama,
            state=tk.DISABLED
        )
        self.install_ollama_btn.pack(side=tk.LEFT, padx=(0, 10))
        create_tooltip(self.install_ollama_btn, TOOLTIPS['install_ollama_btn'])

        self.download_model_btn = ttk.Button(
            llm_buttons_row,
            text="Download Model (~5 GB)",
            command=self.download_model,
            state=tk.DISABLED
        )
        self.download_model_btn.pack(side=tk.LEFT, padx=(0, 10))
        create_tooltip(self.download_model_btn, TOOLTIPS['download_model_btn'])

        # Progress bar for LLM downloads (initially hidden)
        self.llm_progress_frame = ttk.Frame(llm_frame)
        self.llm_progress_frame.pack(fill=tk.X, pady=(5, 0))

        self.llm_progress_bar = ttk.Progressbar(
            self.llm_progress_frame,
            mode='determinate',
            length=400
        )
        self.llm_progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.llm_progress_label = ttk.Label(
            self.llm_progress_frame,
            text="",
            font=("Arial", 8)
        )
        self.llm_progress_label.pack(side=tk.LEFT)

        # Hide progress bar initially
        self.llm_progress_frame.pack_forget()

        # Info label
        ttk.Label(
            llm_frame,
            text="LLM verification improves accuracy by reducing false positives (requires ~8GB download)",
            font=("Arial", 8),
            foreground="gray"
        ).pack(anchor=tk.W, pady=(5, 0))

        # === Scan Controls ===
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        self.scan_button = ttk.Button(
            control_frame,
            text="Start Scan",
            command=self.start_scan,
            style="Accent.TButton"
        )
        self.scan_button.pack(side=tk.LEFT, padx=(0, 10))
        create_tooltip(self.scan_button, TOOLTIPS['scan_button'])

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Scan",
            command=self.stop_scan,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        create_tooltip(self.stop_button, TOOLTIPS['stop_button'])

        # Help button
        help_button = ttk.Button(
            control_frame,
            text="? Help",
            command=self.show_quick_start
        )
        help_button.pack(side=tk.RIGHT)
        create_tooltip(help_button, TOOLTIPS['help_button'])

        # === Progress ===
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="indeterminate"
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        self.status_label = ttk.Label(
            progress_frame,
            text="Ready to scan",
            font=("Arial", 9)
        )
        self.status_label.pack(anchor=tk.W)

        # === Results ===
        results_frame = ttk.LabelFrame(main_frame, text="Scan Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Results text area with scrollbar
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=("Courier New", 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        create_tooltip(self.results_text, TOOLTIPS['results_text'])

        # Configure text tags for colored output
        self.results_text.tag_configure("header", font=FONT_RESULTS_BOLD)
        self.results_text.tag_configure("flagged", foreground=COLOR_FLAGGED, font=FONT_RESULTS_BOLD)
        self.results_text.tag_configure("clean", foreground=COLOR_CLEAN)
        self.results_text.tag_configure("warning", foreground=COLOR_WARNING)

        # === Export Controls ===
        export_frame = ttk.Frame(main_frame)
        export_frame.pack(fill=tk.X)

        self.export_json_btn = ttk.Button(
            export_frame,
            text="Export JSON Report",
            command=self.export_json,
            state=tk.DISABLED
        )
        self.export_json_btn.pack(side=tk.LEFT, padx=(0, 10))
        create_tooltip(self.export_json_btn, TOOLTIPS['export_json_btn'])

        self.export_csv_btn = ttk.Button(
            export_frame,
            text="Export CSV Report",
            command=self.export_csv,
            state=tk.DISABLED
        )
        self.export_csv_btn.pack(side=tk.LEFT, padx=(0, 10))
        create_tooltip(self.export_csv_btn, TOOLTIPS['export_csv_btn'])

        clear_btn = ttk.Button(
            export_frame,
            text="Clear Results",
            command=self.clear_results
        )
        clear_btn.pack(side=tk.LEFT)
        create_tooltip(clear_btn, TOOLTIPS['clear_btn'])

        # === Status Bar ===
        status_bar = ttk.Frame(self.root, relief=tk.SUNKEN, padding="2")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.help_message = tk.StringVar(value=STATUS_MESSAGES['ready'])
        help_label = ttk.Label(
            status_bar,
            textvariable=self.help_message,
            font=FONT_ITALIC,
            foreground=COLOR_STATUS_FG
        )
        help_label.pack(side=tk.LEFT, padx=5)

        # Bind focus events to update help messages
        browse_btn.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['browse']))
        recursive_cb.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['recursive']))
        fuzzy_cb.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['fuzzy']))
        llm_cb.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['llm']))
        sensitivity_combo.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['sensitivity']))
        workers_spin.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['workers']))
        self.scan_button.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['scan']))
        self.export_json_btn.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['export_json']))
        self.export_csv_btn.bind("<FocusIn>", lambda e: self.help_message.set(STATUS_MESSAGES['export_csv']))

    def browse_directory(self):
        """Open directory selection dialog"""
        directory = filedialog.askdirectory(
            title="Select Directory to Scan"
        )
        if directory:
            self.selected_dir.set(directory)

    def browse_output_directory(self):
        """Open output directory selection dialog"""
        directory = filedialog.askdirectory(
            title="Select Output Directory for Flagged Files"
        )
        if directory:
            self.output_dir.set(directory)

    def toggle_output_dir(self):
        """Enable/disable output directory controls based on checkbox"""
        if self.move_flagged.get():
            self.output_entry.config(state="readonly")
            self.output_browse_btn.config(state=tk.NORMAL)
        else:
            self.output_entry.config(state="disabled")
            self.output_browse_btn.config(state=tk.DISABLED)

    def start_scan(self):
        """Start the scanning process"""
        if not self.selected_dir.get():
            messagebox.showerror(
                "Error",
                "Please select a directory to scan"
            )
            return

        if not os.path.isdir(self.selected_dir.get()):
            messagebox.showerror(
                "Error",
                "Selected directory does not exist"
            )
            return

        # Validate output directory if move is enabled
        if self.move_flagged.get():
            if not self.output_dir.get():
                messagebox.showerror(
                    "Error",
                    "Please select an output directory for flagged files"
                )
                return
            if not os.path.isdir(self.output_dir.get()):
                messagebox.showerror(
                    "Error",
                    "Selected output directory does not exist"
                )
                return
            # Check if output dir is same as source dir
            if os.path.abspath(self.output_dir.get()) == os.path.abspath(self.selected_dir.get()):
                messagebox.showerror(
                    "Error",
                    "Output directory cannot be the same as the source directory"
                )
                return

        # Update UI state
        with self.scan_lock:
            self.is_scanning = True
        self.scan_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar.start()
        self.status_label.config(text="Scanning in progress...")
        self.clear_results()

        # Disable export buttons
        self.export_json_btn.config(state=tk.DISABLED)
        self.export_csv_btn.config(state=tk.DISABLED)

        # Start scan in separate thread
        scan_thread = threading.Thread(target=self.run_scan, daemon=True)
        scan_thread.start()

    def stop_scan(self):
        """Stop the scanning process"""
        with self.scan_lock:
            self.is_scanning = False
        self.status_label.config(text="Scan stopped by user")

    def run_scan(self):
        """Run the actual scan (in separate thread)"""
        try:
            # Build configuration
            config = {
                'fuzzy_matching': self.fuzzy_matching.get(),
                'fuzzy_threshold': 85,
                'use_llm': self.use_llm.get(),
                'llm_model': LLM_MODEL_NAME,
                'confidence_threshold': SENSITIVITY_THRESHOLDS[self.sensitivity.get()],
                'workers': self.workers.get(),
            }

            # Create scanner
            scanner = ClassificationScanner(config)

            # Update status
            self.progress_queue.put(("status", "Scanning files..."))

            # Run scan
            results = scanner.scan_directory(
                self.selected_dir.get(),
                recursive=self.recursive.get()
            )

            # Store results
            self.scan_results = results

            # Generate report
            with self.scan_lock:
                should_generate_report = self.is_scanning

            if should_generate_report:
                self.progress_queue.put(("status", "Generating report..."))
                self.progress_queue.put(("results", results))
                self.progress_queue.put(("complete", None))

        except Exception as e:
            self.progress_queue.put(("error", str(e)))

    def check_progress(self):
        """Check for progress updates from scan thread"""
        try:
            while True:
                msg_type, msg_data = self.progress_queue.get_nowait()

                if msg_type == "status":
                    self.status_label.config(text=msg_data)

                elif msg_type == "results":
                    self.display_results(msg_data)

                elif msg_type == "complete":
                    self.scan_complete()

                elif msg_type == "error":
                    self.scan_error(msg_data)

                elif msg_type == "llm_status":
                    self.llm_status.set(msg_data)
                    # Parse progress percentage from message and update progress bar
                    import re
                    percent_match = re.search(r'\((\d+)%\)', msg_data)
                    if percent_match:
                        percent = int(percent_match.group(1))
                        self.llm_progress_bar['value'] = percent
                        self.llm_progress_label.config(text=f"{percent}%")
                        # Show progress bar if hidden
                        if not self.llm_progress_frame.winfo_ismapped():
                            self.llm_progress_frame.pack(fill=tk.X, pady=(5, 0), before=self.llm_status_label.master.master.winfo_children()[-1])

                elif msg_type == "llm_complete":
                    self.llm_status.set("Ready")
                    self.is_downloading = False
                    self.install_ollama_btn.config(state=tk.DISABLED)
                    # Hide progress bar
                    self.llm_progress_frame.pack_forget()
                    self.llm_progress_bar['value'] = 0
                    messagebox.showinfo("Success", msg_data)
                    # Re-check LLM status
                    self.check_llm_status()

                elif msg_type == "llm_error":
                    self.llm_status.set("Error occurred")
                    self.is_downloading = False
                    self.install_ollama_btn.config(state=tk.NORMAL)
                    self.download_model_btn.config(state=tk.DISABLED)
                    # Hide progress bar
                    self.llm_progress_frame.pack_forget()
                    self.llm_progress_bar['value'] = 0
                    messagebox.showerror("Error", msg_data)

                elif msg_type == "move_complete":
                    self.progress_bar.stop()
                    move_report = msg_data

                    # Build summary message
                    summary = f"File Move Complete\n\n"
                    summary += f"Total files: {move_report['total']}\n"
                    summary += f"Successfully moved: {move_report['moved']}\n"
                    summary += f"Failed: {move_report['failed']}\n"

                    if move_report['errors']:
                        summary += "\nErrors:\n"
                        for error in move_report['errors'][:5]:
                            summary += f"• {Path(error['file']).name}: {error['error']}\n"
                        if len(move_report['errors']) > 5:
                            summary += f"... and {len(move_report['errors']) - 5} more errors\n"

                    messagebox.showinfo("Move Complete", summary)
                    self.status_label.config(
                        text=f"Files moved: {move_report['moved']} successful, {move_report['failed']} failed"
                    )

                    # Append move report to results
                    self.append_move_report_to_results(move_report)

                elif msg_type == "move_error":
                    self.progress_bar.stop()
                    self.status_label.config(text="Error moving files")
                    messagebox.showerror("Move Error", f"Failed to move files:\n\n{msg_data}")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.check_progress)

    def display_results(self, results):
        """Display scan results in the text area"""
        self.results_text.delete(1.0, tk.END)

        flagged = [r for r in results if r.has_markings]
        clean = [r for r in results if not r.has_markings]

        # Header
        self.results_text.insert(tk.END, "="*80 + "\n", "header")
        self.results_text.insert(tk.END, "CLASSIFICATION MARKING SCAN RESULTS\n", "header")
        self.results_text.insert(tk.END, "="*80 + "\n\n", "header")

        # Summary
        self.results_text.insert(tk.END, f"Total files scanned: {len(results)}\n")
        self.results_text.insert(tk.END, f"Files flagged: {len(flagged)}\n", "flagged" if flagged else "clean")
        self.results_text.insert(tk.END, f"Clean files: {len(clean)}\n\n", "clean")

        if flagged:
            self.results_text.insert(tk.END, "⚠️  FLAGGED FILES\n", "flagged")
            self.results_text.insert(tk.END, "-"*80 + "\n\n", "header")

            for result in sorted(flagged, key=lambda x: x.classification_level):
                self.results_text.insert(tk.END, f"📄 File: {result.file_path}\n")
                self.results_text.insert(
                    tk.END,
                    f"   Classification: {result.classification_level.value}\n",
                    "flagged"
                )
                self.results_text.insert(
                    tk.END,
                    f"   Confidence: {result.overall_confidence:.1%}\n"
                )
                self.results_text.insert(
                    tk.END,
                    f"   Matches: {len(result.matches)}\n"
                )

                # Show top 3 matches
                for i, match in enumerate(result.matches[:3], 1):
                    self.results_text.insert(tk.END, f"\n   Match {i}:\n")
                    self.results_text.insert(
                        tk.END,
                        f"     Text: {match.text[:60]}\n"
                    )
                    self.results_text.insert(
                        tk.END,
                        f"     Line: {match.line_number}, "
                        f"Location: {match.location}, "
                        f"Confidence: {match.confidence:.1%}\n"
                    )

                if len(result.matches) > 3:
                    self.results_text.insert(
                        tk.END,
                        f"\n   ... and {len(result.matches) - 3} more matches\n"
                    )

                self.results_text.insert(tk.END, "\n")

        else:
            self.results_text.insert(
                tk.END,
                "✅ No classified documents found\n\n",
                "clean"
            )

        # Clean files summary
        if clean:
            self.results_text.insert(tk.END, f"✅ {len(clean)} clean files\n", "clean")

        self.results_text.insert(tk.END, "\n" + "="*80 + "\n", "header")

        # Scroll to top
        self.results_text.see(1.0)

    def append_move_report_to_results(self, move_report):
        """Append file move report to results display"""
        if not move_report:
            return

        # Add separator
        self.results_text.insert(tk.END, "\n" + "="*80 + "\n", "header")
        self.results_text.insert(tk.END, "FILE MOVE REPORT\n", "header")
        self.results_text.insert(tk.END, "="*80 + "\n\n", "header")

        # Summary
        self.results_text.insert(tk.END, f"Total files to move: {move_report['total']}\n")
        self.results_text.insert(
            tk.END,
            f"Successfully moved: {move_report['moved']}\n",
            "clean" if move_report['moved'] > 0 else None
        )
        if move_report['failed'] > 0:
            self.results_text.insert(
                tk.END,
                f"Failed to move: {move_report['failed']}\n",
                "flagged"
            )
        self.results_text.insert(tk.END, "\n")

        # List moved files
        if move_report['moved_files']:
            self.results_text.insert(tk.END, "MOVED FILES:\n", "header")
            for item in move_report['moved_files']:
                source_name = Path(item['source']).name
                dest_path = item['destination']
                self.results_text.insert(tk.END, f"✓ {source_name}\n", "clean")
                self.results_text.insert(tk.END, f"  → {dest_path}\n")
            self.results_text.insert(tk.END, "\n")

        # List errors
        if move_report['errors']:
            self.results_text.insert(tk.END, "ERRORS:\n", "flagged")
            for error in move_report['errors']:
                file_name = Path(error['file']).name
                self.results_text.insert(tk.END, f"✗ {file_name}\n", "flagged")
                self.results_text.insert(tk.END, f"  Error: {error['error']}\n")
            self.results_text.insert(tk.END, "\n")

        self.results_text.insert(tk.END, "="*80 + "\n", "header")

        # Scroll to bottom to show the report
        self.results_text.see(tk.END)

    def scan_complete(self):
        """Handle scan completion"""
        with self.scan_lock:
            self.is_scanning = False
        self.scan_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()

        flagged_count = sum(1 for r in self.scan_results if r.has_markings)

        if flagged_count > 0:
            self.status_label.config(
                text=f"Scan complete - {flagged_count} classified document(s) found"
            )

            # Ask to move files if option is enabled
            if self.move_flagged.get():
                flagged_results = [r for r in self.scan_results if r.has_markings]

                # Show confirmation dialog
                file_list = "\n".join([f"• {Path(r.file_path).name}" for r in flagged_results[:10]])
                if len(flagged_results) > 10:
                    file_list += f"\n... and {len(flagged_results) - 10} more files"

                confirm = messagebox.askyesno(
                    "Move Flagged Files",
                    f"Found {flagged_count} file(s) with classification markings:\n\n{file_list}\n\n"
                    f"Move these files to:\n{self.output_dir.get()}\n\n"
                    "Do you want to proceed?",
                    icon='warning'
                )

                if confirm:
                    self.status_label.config(text="Moving flagged files...")
                    self.progress_bar.start()

                    # Move files in a thread to keep UI responsive
                    threading.Thread(
                        target=self._move_files_thread,
                        args=(flagged_results,),
                        daemon=True
                    ).start()
                else:
                    self.status_label.config(
                        text=f"Scan complete - {flagged_count} classified document(s) found (not moved)"
                    )
        else:
            self.status_label.config(
                text="Scan complete - No classified documents found"
            )

        # Enable export buttons
        if self.scan_results:
            self.export_json_btn.config(state=tk.NORMAL)
            self.export_csv_btn.config(state=tk.NORMAL)

    def scan_error(self, error_msg):
        """Handle scan error"""
        with self.scan_lock:
            self.is_scanning = False
        self.scan_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.status_label.config(text=f"Error: {error_msg}")

        messagebox.showerror(
            "Scan Error",
            f"An error occurred during scanning:\n\n{error_msg}"
        )

    def move_flagged_files(self, flagged_results):
        """Move flagged files to output directory using FileManager"""
        if not flagged_results or not self.move_flagged.get():
            return None

        # Initialize FileManager
        self.file_manager = FileManager(
            self.selected_dir.get(),
            self.output_dir.get()
        )

        # Validate directories
        is_valid, error = self.file_manager.validate_directories()
        if not is_valid:
            return {
                'total': len(flagged_results),
                'moved': 0,
                'failed': len(flagged_results),
                'errors': [{'file': 'Validation', 'error': error}],
                'moved_files': []
            }

        # Move files
        return self.file_manager.move_files(flagged_results)

    def _move_files_thread(self, flagged_results):
        """Thread function to move files"""
        try:
            move_report = self.move_flagged_files(flagged_results)

            if move_report:
                self.progress_queue.put(("move_complete", move_report))
            else:
                self.progress_queue.put(("move_error", "Failed to move files"))

        except Exception as e:
            self.progress_queue.put(("move_error", str(e)))

    def export_json(self):
        """Export results to JSON file"""
        if not self.scan_results:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"classification_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if filename:
            try:
                report_data = ReportGenerator.generate_json_report(self.scan_results)
                with open(filename, 'w') as f:
                    json.dump(report_data, f, indent=2)

                messagebox.showinfo(
                    "Export Successful",
                    f"JSON report saved to:\n{filename}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Export Error",
                    f"Failed to export JSON report:\n{str(e)}"
                )

    def export_csv(self):
        """Export results to CSV file"""
        if not self.scan_results:
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"classification_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        if filename:
            try:
                csv_data = ReportGenerator.generate_csv_report(self.scan_results)
                with open(filename, 'w') as f:
                    f.write(csv_data)

                messagebox.showinfo(
                    "Export Successful",
                    f"CSV report saved to:\n{filename}"
                )
            except Exception as e:
                messagebox.showerror(
                    "Export Error",
                    f"Failed to export CSV report:\n{str(e)}"
                )

    def clear_results(self):
        """Clear the results display"""
        self.results_text.delete(1.0, tk.END)
        self.scan_results = []
        self.export_json_btn.config(state=tk.DISABLED)
        self.export_csv_btn.config(state=tk.DISABLED)

    # === LLM Management Methods ===

    def check_llm_status(self):
        """Check if Ollama and model are installed"""
        # Run in thread to avoid blocking UI
        thread = threading.Thread(target=self._check_llm_status_thread, daemon=True)
        thread.start()

    def _check_llm_status_thread(self):
        """Thread function to check LLM status"""
        try:
            # Use LLMInstaller to check status
            ollama_installed, model_downloaded, status_msg = self.llm_installer.get_status()

            self.ollama_installed.set(ollama_installed)
            self.model_downloaded.set(model_downloaded)
            self.llm_status.set(status_msg)

            # Update button states
            if ollama_installed and model_downloaded:
                self.install_ollama_btn.config(state=tk.DISABLED)
                self.download_model_btn.config(state=tk.DISABLED)
            elif ollama_installed:
                self.install_ollama_btn.config(state=tk.DISABLED)
                self.download_model_btn.config(state=tk.NORMAL)
            else:
                self.install_ollama_btn.config(state=tk.NORMAL)
                self.download_model_btn.config(state=tk.DISABLED)

        except Exception as e:
            self.llm_status.set(f"Error checking status: {str(e)}")

    def install_ollama(self):
        """Download and install Ollama"""
        if self.is_downloading:
            messagebox.showwarning(
                "Download in Progress",
                "A download is already in progress. Please wait."
            )
            return

        # Confirm with user
        response = messagebox.askyesno(
            "Install Ollama",
            "This will download and install Ollama (~100-200 MB).\n\n"
            "Installation steps:\n"
            "1. Download Ollama installer\n"
            "2. Run the installer\n"
            "3. Follow installation prompts\n\n"
            "Continue?"
        )

        if not response:
            return

        # Disable button and update status
        self.install_ollama_btn.config(state=tk.DISABLED)
        self.llm_status.set("Downloading Ollama installer...")
        self.is_downloading = True

        # Run in thread
        thread = threading.Thread(target=self._install_ollama_thread, daemon=True)
        thread.start()

    def _install_ollama_thread(self):
        """Thread function to install Ollama using LLMInstaller"""
        try:
            success, message = self.llm_installer.install_ollama()

            if success:
                self.progress_queue.put(("llm_complete", message or "Ollama installation started!"))
            else:
                self.progress_queue.put(("llm_error", message or "Installation failed"))

        except Exception as e:
            self.progress_queue.put(("llm_error", f"Installation failed: {str(e)}"))

    def download_model(self):
        """Download the LLM model"""
        if self.is_downloading:
            messagebox.showwarning(
                "Download in Progress",
                "A download is already in progress. Please wait."
            )
            return

        if not self.ollama_installed.get():
            messagebox.showerror(
                "Ollama Not Installed",
                "Please install Ollama first before downloading the model."
            )
            return

        # Confirm with user
        response = messagebox.askyesno(
            "Download Model",
            f"This will download the {LLM_MODEL_NAME} model (~5 GB).\n\n"
            "This may take 10-30 minutes depending on your connection.\n\n"
            "Continue?"
        )

        if not response:
            return

        # Disable button and update status
        self.download_model_btn.config(state=tk.DISABLED)
        self.llm_status.set("Downloading model (this may take a while)...")
        self.is_downloading = True

        # Run in thread
        thread = threading.Thread(target=self._download_model_thread, daemon=True)
        thread.start()

    def _download_model_thread(self):
        """Thread function to download model using LLMInstaller"""
        try:
            # Callback to update progress through queue
            def progress_callback(message):
                self.progress_queue.put(("llm_status", message))

            success, message = self.llm_installer.download_model(
                LLM_MODEL_NAME,
                progress_callback
            )

            if success:
                self.progress_queue.put((
                    "llm_complete",
                    message + "\n\nYou can now enable 'Use LLM verification' in scan options."
                ))
            else:
                self.progress_queue.put(("llm_error", message))

        except Exception as e:
            self.progress_queue.put(("llm_error", f"Download failed: {str(e)}"))


def main():
    """Main entry point"""
    root = tk.Tk()

    # Set icon (if available)
    # root.iconbitmap('icon.ico')

    app = ScannerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
