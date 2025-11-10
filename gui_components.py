#!/usr/bin/env python3
"""
GUI Components for Classification Scanner

This module contains reusable GUI components and widgets used
throughout the application.

Author: Claude Code
Date: 2025-01-05
Version: 2.2
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from config import (
    COLOR_TOOLTIP_BG,
    COLOR_TOOLTIP_FG,
    FONT_TOOLTIP,
    TOOLTIP_DELAY_MS
)


class ToolTip:
    """
    Create a tooltip for a given widget with enhanced styling

    The tooltip appears after hovering over the widget for a specified delay
    and automatically hides when the mouse leaves or when clicking.
    """

    def __init__(self, widget, text):
        """
        Initialize tooltip for a widget

        Args:
            widget: The tkinter widget to attach the tooltip to
            text: The tooltip text to display
        """
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0

        # Bind events
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)

    def on_enter(self, event=None):
        """Show tooltip on mouse enter"""
        self.schedule()

    def on_leave(self, event=None):
        """Hide tooltip on mouse leave"""
        self.unschedule()
        self.hide()

    def schedule(self):
        """Schedule tooltip to appear after delay"""
        self.unschedule()
        self.id = self.widget.after(TOOLTIP_DELAY_MS, self.show)

    def unschedule(self):
        """Cancel scheduled tooltip"""
        id_val = self.id
        self.id = None
        if id_val:
            self.widget.after_cancel(id_val)

    def show(self):
        """Display the tooltip"""
        if self.tipwindow or not self.text:
            return

        # Get widget position
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        # Create tooltip window
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Remove window decorations
        tw.wm_geometry(f"+{x}+{y}")

        # Add styling
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background=COLOR_TOOLTIP_BG,
            foreground=COLOR_TOOLTIP_FG,
            relief=tk.SOLID,
            borderwidth=1,
            font=FONT_TOOLTIP,
            padx=8,
            pady=6
        )
        label.pack()

    def hide(self):
        """Hide the tooltip"""
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


def create_tooltip(widget, text):
    """
    Helper function to create tooltips easily

    Args:
        widget: The tkinter widget to attach the tooltip to
        text: The tooltip text to display

    Returns:
        ToolTip: The created ToolTip instance
    """
    return ToolTip(widget, text)


def create_scrolled_text_dialog(parent, title, text, width=500, height=550):
    """
    Create a dialog window with scrollable text

    Args:
        parent: Parent window
        title: Dialog title
        text: Text content to display
        width: Dialog width in pixels
        height: Dialog height in pixels

    Returns:
        tk.Toplevel: The created dialog window
    """
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.geometry(f"{width}x{height}")
    dialog.resizable(False, False)

    # Center the dialog
    dialog.transient(parent)
    dialog.grab_set()

    # Add text
    text_widget = scrolledtext.ScrolledText(
        dialog,
        wrap=tk.WORD,
        font=("Arial", 10),
        padx=10,
        pady=10
    )
    text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    text_widget.insert(1.0, text)
    text_widget.config(state=tk.DISABLED)

    # Add close button
    close_btn = ttk.Button(
        dialog,
        text="Close",
        command=dialog.destroy
    )
    close_btn.pack(pady=(0, 10))

    return dialog


def create_labeled_entry(parent, label_text, variable, width=60, state="normal"):
    """
    Create a labeled entry field with consistent styling

    Args:
        parent: Parent widget
        label_text: Label text
        variable: tk.StringVar for the entry
        width: Entry width
        state: Entry state ("normal", "readonly", "disabled")

    Returns:
        tuple: (frame, label, entry) widgets
    """
    frame = ttk.Frame(parent)

    label = ttk.Label(frame, text=label_text)
    label.pack(side=tk.LEFT, padx=(0, 5))

    entry = ttk.Entry(frame, textvariable=variable, state=state, width=width)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    return frame, label, entry


def create_button_row(parent, buttons_config):
    """
    Create a row of buttons with consistent spacing

    Args:
        parent: Parent widget
        buttons_config: List of dicts with button configuration
            Each dict should have: 'text', 'command', and optional 'state'

    Returns:
        tuple: (frame, list of button widgets)
    """
    frame = ttk.Frame(parent)
    buttons = []

    for i, config in enumerate(buttons_config):
        btn = ttk.Button(
            frame,
            text=config['text'],
            command=config['command'],
            state=config.get('state', tk.NORMAL)
        )
        padx = (0, 10) if i < len(buttons_config) - 1 else 0
        btn.pack(side=tk.LEFT, padx=padx)
        buttons.append(btn)

    return frame, buttons


def create_checkbox_row(parent, checkboxes_config):
    """
    Create a row of checkboxes with consistent spacing

    Args:
        parent: Parent widget
        checkboxes_config: List of dicts with checkbox configuration
            Each dict should have: 'text', 'variable', and optional 'command'

    Returns:
        tuple: (frame, list of checkbox widgets)
    """
    frame = ttk.Frame(parent)
    checkboxes = []

    for i, config in enumerate(checkboxes_config):
        cb = ttk.Checkbutton(
            frame,
            text=config['text'],
            variable=config['variable'],
            command=config.get('command')
        )
        padx = (0, 20) if i < len(checkboxes_config) - 1 else 0
        cb.pack(side=tk.LEFT, padx=padx)
        checkboxes.append(cb)

    return frame, checkboxes


def bind_focus_to_status(widget, status_var, message):
    """
    Bind widget focus events to update a status message

    Args:
        widget: Widget to bind to
        status_var: tk.StringVar to update
        message: Message to display on focus
    """
    widget.bind("<FocusIn>", lambda e: status_var.set(message))


class StatusBar(ttk.Frame):
    """Status bar widget with message display"""

    def __init__(self, parent, initial_message="Ready"):
        """
        Initialize status bar

        Args:
            parent: Parent widget
            initial_message: Initial status message
        """
        super().__init__(parent, relief=tk.SUNKEN, padding="2")

        self.message = tk.StringVar(value=initial_message)
        self.label = ttk.Label(
            self,
            textvariable=self.message,
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        self.label.pack(side=tk.LEFT, padx=5)

    def set_message(self, message):
        """Update status message"""
        self.message.set(message)

    def get_message_var(self):
        """Get the message StringVar for binding"""
        return self.message
