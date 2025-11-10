#!/usr/bin/env python3
"""
LLM Installer for Classification Scanner

This module handles LLM (Large Language Model) installation and management,
including Ollama installation and model downloading.

Author: Claude Code
Date: 2025-01-05
Version: 2.2
"""

import subprocess
import platform
import urllib.request
from pathlib import Path
from typing import Tuple, Optional, Callable
from config import (
    OLLAMA_URLS,
    LLM_MODEL_NAME,
    LLM_CHECK_TIMEOUT_SEC,
    LLM_LIST_TIMEOUT_SEC
)


class LLMInstaller:
    """Manages LLM installation and model downloading"""

    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize LLM Installer

        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.progress_callback = progress_callback
        self.system = platform.system()

    def check_ollama_installed(self) -> bool:
        """
        Check if Ollama is installed

        Returns:
            bool: True if Ollama is installed, False otherwise
        """
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                timeout=LLM_CHECK_TIMEOUT_SEC,
                stdin=subprocess.DEVNULL
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def check_model_downloaded(self, model_name: str = LLM_MODEL_NAME) -> bool:
        """
        Check if a specific model is downloaded

        Args:
            model_name: Name of the model to check

        Returns:
            bool: True if model is downloaded, False otherwise
        """
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=LLM_LIST_TIMEOUT_SEC,
                stdin=subprocess.DEVNULL
            )
            if result.returncode == 0:
                return model_name in result.stdout
            return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def get_status(self) -> Tuple[bool, bool, str]:
        """
        Get overall LLM status

        Returns:
            tuple: (ollama_installed, model_downloaded, status_message)
        """
        ollama_installed = self.check_ollama_installed()

        if not ollama_installed:
            return False, False, "Ollama not installed"

        model_downloaded = self.check_model_downloaded()

        if model_downloaded:
            return True, True, "✓ Ready (Ollama + Model)"
        else:
            return True, False, "Ollama installed, model not downloaded"

    def install_ollama(self) -> Tuple[bool, Optional[str]]:
        """
        Download and install Ollama

        Returns:
            tuple: (success, error_message)
        """
        try:
            if self.system == "Linux":
                return self._install_ollama_linux()
            elif self.system in ["Windows", "Darwin"]:
                return self._install_ollama_gui_platform()
            else:
                return False, f"Unsupported platform: {self.system}"

        except Exception as e:
            return False, f"Installation failed: {str(e)}"

    def _install_ollama_linux(self) -> Tuple[bool, Optional[str]]:
        """
        Install Ollama on Linux using curl script

        Returns:
            tuple: (success, error_message)
        """
        self._update_progress("Installing Ollama on Linux...")

        try:
            # Security: Validate URL is from official Ollama domain
            install_url = OLLAMA_URLS['Linux']
            if not install_url.startswith('https://ollama.com/'):
                return False, "Security: Installation URL must be from ollama.com"

            # Fetch install script
            result = subprocess.run(
                ['curl', '-fsSL', install_url],
                capture_output=True,
                text=True,
                timeout=30,
                stdin=subprocess.DEVNULL
            )

            if result.returncode != 0:
                return False, "Failed to download installation script"

            # Security: Basic validation of script content
            script_content = result.stdout
            if len(script_content) < 100:  # Sanity check - install script should be substantial
                return False, "Downloaded script appears invalid (too short)"

            # Run install script
            install_result = subprocess.run(
                ['sh', '-c', script_content],
                capture_output=True,
                timeout=300,  # 5 minutes timeout
                stdin=subprocess.DEVNULL
            )

            if install_result.returncode == 0:
                return True, None
            else:
                return False, "Installation script failed"

        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, str(e)

    def _install_ollama_gui_platform(self) -> Tuple[bool, Optional[str]]:
        """
        Install Ollama on Windows/macOS (downloads installer)

        Returns:
            tuple: (success, error_message)
        """
        url = OLLAMA_URLS.get(self.system)
        if not url:
            return False, f"No installer URL for {self.system}"

        # Security: Validate URL is from official Ollama domain
        if not url.startswith('https://ollama.com/'):
            return False, "Security: Installation URL must be from ollama.com"

        self._update_progress(f"Downloading from {url}...")

        try:
            # Determine installer filename
            if self.system == "Windows":
                filename = "OllamaSetup.exe"
            else:  # Darwin (macOS)
                filename = "Ollama-darwin.zip"

            installer_path = Path.home() / "Downloads" / filename

            # Download installer with progress tracking
            def reporthook(block_num, block_size, total_size):
                """Progress callback for download"""
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, (downloaded * 100) // total_size)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    progress_msg = f"Downloading: {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent}%)"
                    self._update_progress(progress_msg)

            urllib.request.urlretrieve(url, installer_path, reporthook=reporthook)

            # Security: Validate downloaded file size (should be between 50MB and 2GB)
            file_size = installer_path.stat().st_size
            min_size = 50 * 1024 * 1024  # 50 MB
            max_size = 2 * 1024 * 1024 * 1024  # 2 GB
            if file_size < min_size or file_size > max_size:
                installer_path.unlink()  # Delete suspicious file
                return False, f"Downloaded file size ({file_size / 1024 / 1024:.1f} MB) is outside expected range"

            self._update_progress("Download complete. Starting installer...")

            # Launch installer
            if self.system == "Windows":
                subprocess.Popen([str(installer_path)], stdin=subprocess.DEVNULL)
                message = (
                    "Ollama installer downloaded and started!\n\n"
                    "Please complete the installation wizard.\n"
                    "After installation, click 'Download Model' to get the LLM."
                )
            else:  # macOS
                subprocess.Popen(['open', str(installer_path)], stdin=subprocess.DEVNULL)
                message = (
                    "Ollama installer downloaded!\n\n"
                    "Please complete the installation.\n"
                    "After installation, click 'Download Model' to get the LLM."
                )

            return True, message

        except Exception as e:
            return False, str(e)

    def download_model(
        self,
        model_name: str = LLM_MODEL_NAME,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Download LLM model

        Args:
            model_name: Name of the model to download
            progress_callback: Optional callback for progress updates

        Returns:
            tuple: (success, error_message)
        """
        if not self.check_ollama_installed():
            return False, "Ollama is not installed"

        try:
            self._update_progress(f"Downloading {model_name} (~8.0 GB)...")

            # Run ollama pull command
            process = subprocess.Popen(
                ['ollama', 'pull', model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',  # Replace invalid characters instead of crashing
                bufsize=1,
                stdin=subprocess.DEVNULL
            )

            # Monitor progress - parse ollama output for percentage
            import re
            for line in process.stdout:
                if line.strip():
                    # Parse ollama progress output (e.g., "pulling 8934d96d3f08... 50% ▕████████░░░░░░░░▏ 1.9 GB")
                    percent_match = re.search(r'(\d+)%', line)
                    size_match = re.search(r'([\d.]+\s*[KMG]B)', line)

                    if percent_match and size_match:
                        percent = percent_match.group(1)
                        size = size_match.group(1)
                        progress_text = f"Downloading model: {size} ({percent}%)"
                    elif "pulling manifest" in line.lower():
                        progress_text = "Preparing download... (0%)"
                    elif "verifying" in line.lower():
                        progress_text = "Verifying download (100%)"
                    elif "success" in line.lower():
                        progress_text = "Download complete (100%)"
                    else:
                        # Fallback: show truncated raw output
                        progress_text = f"Downloading: {line.strip()[:60]}"

                    self._update_progress(progress_text)
                    if progress_callback:
                        progress_callback(progress_text)

            process.wait()

            if process.returncode == 0:
                return True, "Model downloaded successfully!"
            else:
                return False, "Model download failed"

        except FileNotFoundError:
            return False, "Ollama command not found"
        except Exception as e:
            return False, str(e)

    def _update_progress(self, message: str):
        """
        Update progress through callback

        Args:
            message: Progress message
        """
        if self.progress_callback:
            self.progress_callback(message)


def check_llm_available() -> Tuple[bool, str]:
    """
    Quick check if LLM is available for use

    Returns:
        tuple: (is_available, status_message)
    """
    installer = LLMInstaller()
    ollama_installed, model_downloaded, status = installer.get_status()

    if ollama_installed and model_downloaded:
        return True, "LLM ready"
    elif ollama_installed:
        return False, "Model not downloaded"
    else:
        return False, "Ollama not installed"


def get_llm_status_message() -> str:
    """
    Get human-readable LLM status message

    Returns:
        str: Status message
    """
    installer = LLMInstaller()
    _, _, status = installer.get_status()
    return status
