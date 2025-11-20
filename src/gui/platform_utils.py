#!/usr/bin/env python3
"""
Platform Detection and Utilities

Provides enhanced platform detection for Mac, Windows, Linux, and WSL
to enable platform-specific UI/UX optimizations.

Author: Classmark Development Team
Date: 2025-11-19
"""

import sys
import platform
import os
from typing import Tuple, Optional
from enum import Enum


class Platform(Enum):
    """Supported platforms"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    WSL = "wsl"
    UNKNOWN = "unknown"


class PlatformInfo:
    """Comprehensive platform information"""

    def __init__(self):
        self._platform = self._detect_platform()
        self._version = self._detect_version()
        self._is_dark_mode = self._detect_dark_mode()

    @property
    def platform(self) -> Platform:
        """Get the current platform"""
        return self._platform

    @property
    def version(self) -> str:
        """Get the platform version"""
        return self._version

    @property
    def is_windows(self) -> bool:
        """Check if running on Windows"""
        return self._platform == Platform.WINDOWS

    @property
    def is_macos(self) -> bool:
        """Check if running on macOS"""
        return self._platform == Platform.MACOS

    @property
    def is_linux(self) -> bool:
        """Check if running on Linux (not WSL)"""
        return self._platform == Platform.LINUX

    @property
    def is_wsl(self) -> bool:
        """Check if running on Windows Subsystem for Linux"""
        return self._platform == Platform.WSL

    @property
    def is_dark_mode(self) -> bool:
        """Check if system is in dark mode"""
        return self._is_dark_mode

    @property
    def system_font(self) -> str:
        """Get the recommended system font"""
        if self.is_macos:
            return "-apple-system"  # San Francisco
        elif self.is_windows or self.is_wsl:
            return "Segoe UI"
        else:
            return "sans-serif"

    @property
    def monospace_font(self) -> str:
        """Get the recommended monospace font"""
        if self.is_macos:
            return "SF Mono"
        elif self.is_windows or self.is_wsl:
            return "Consolas"
        else:
            return "monospace"

    @property
    def supports_native_dialogs(self) -> bool:
        """Check if platform supports native file dialogs"""
        # WSL has issues with tkinter file dialogs
        return not self.is_wsl

    @property
    def supports_notifications(self) -> bool:
        """Check if platform supports system notifications"""
        # All platforms support notifications
        return True

    @property
    def accent_color(self) -> Optional[str]:
        """Get system accent color if available"""
        if self.is_windows:
            return self._get_windows_accent_color()
        elif self.is_macos:
            return self._get_macos_accent_color()
        return None

    def _detect_platform(self) -> Platform:
        """Detect the current platform"""
        system = platform.system().lower()

        if system == "windows":
            return Platform.WINDOWS
        elif system == "darwin":
            return Platform.MACOS
        elif system == "linux":
            # Check if WSL
            if self._is_wsl():
                return Platform.WSL
            return Platform.LINUX
        else:
            return Platform.UNKNOWN

    def _detect_version(self) -> str:
        """Detect the platform version"""
        try:
            if self.is_macos:
                return platform.mac_ver()[0]
            elif self.is_windows:
                return platform.version()
            elif self.is_linux or self.is_wsl:
                return platform.release()
        except:
            pass
        return "unknown"

    def _is_wsl(self) -> bool:
        """Check if running on Windows Subsystem for Linux"""
        try:
            with open('/proc/version', 'r') as f:
                version = f.read().lower()
                return 'microsoft' in version or 'wsl' in version
        except:
            return False

    def _detect_dark_mode(self) -> bool:
        """Detect if system is in dark mode"""
        if self.is_macos:
            return self._detect_macos_dark_mode()
        elif self.is_windows or self.is_wsl:
            return self._detect_windows_dark_mode()
        else:
            return self._detect_linux_dark_mode()

    def _detect_macos_dark_mode(self) -> bool:
        """Detect macOS dark mode"""
        try:
            import subprocess
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleInterfaceStyle'],
                capture_output=True,
                text=True,
                timeout=1
            )
            return result.returncode == 0 and 'dark' in result.stdout.lower()
        except:
            return False

    def _detect_windows_dark_mode(self) -> bool:
        """Detect Windows dark mode"""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r'Software\Microsoft\Windows\CurrentVersion\Themes\Personalize'
            )
            value, _ = winreg.QueryValueEx(key, 'AppsUseLightTheme')
            winreg.CloseKey(key)
            return value == 0  # 0 = dark mode, 1 = light mode
        except:
            return False

    def _detect_linux_dark_mode(self) -> bool:
        """Detect Linux dark mode (GTK)"""
        try:
            # Try to read GTK theme setting
            gtk_settings = os.path.expanduser('~/.config/gtk-3.0/settings.ini')
            if os.path.exists(gtk_settings):
                with open(gtk_settings, 'r') as f:
                    content = f.read().lower()
                    return 'dark' in content
        except:
            pass
        return False

    def _get_windows_accent_color(self) -> Optional[str]:
        """Get Windows accent color"""
        try:
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r'Software\Microsoft\Windows\DWM'
            )
            value, _ = winreg.QueryValueEx(key, 'AccentColor')
            winreg.CloseKey(key)

            # Convert DWORD to RGB hex
            # Windows stores as AABBGGRR
            bgr = value & 0xFFFFFF
            r = bgr & 0xFF
            g = (bgr >> 8) & 0xFF
            b = (bgr >> 16) & 0xFF
            return f"#{r:02x}{g:02x}{b:02x}"
        except:
            return None

    def _get_macos_accent_color(self) -> Optional[str]:
        """Get macOS accent color"""
        try:
            import subprocess
            result = subprocess.run(
                ['defaults', 'read', '-g', 'AppleAccentColor'],
                capture_output=True,
                text=True,
                timeout=1
            )
            if result.returncode == 0:
                # macOS accent colors: -1=graphite, 0=red, 1=orange, 2=yellow,
                # 3=green, 4=blue (default), 5=purple, 6=pink
                accent_map = {
                    '-1': '#8e8e93',  # Graphite
                    '0': '#ff3b30',   # Red
                    '1': '#ff9500',   # Orange
                    '2': '#ffcc00',   # Yellow
                    '3': '#34c759',   # Green
                    '4': '#007aff',   # Blue (default)
                    '5': '#af52de',   # Purple
                    '6': '#ff2d55',   # Pink
                }
                color_id = result.stdout.strip()
                return accent_map.get(color_id, '#007aff')
        except:
            pass
        return '#007aff'  # Default to blue

    def __str__(self) -> str:
        """String representation"""
        return f"Platform: {self._platform.value}, Version: {self._version}, Dark Mode: {self._is_dark_mode}"

    def __repr__(self) -> str:
        """Debug representation"""
        return f"PlatformInfo(platform={self._platform}, version='{self._version}', dark_mode={self._is_dark_mode})"


# Global instance
_platform_info = None


def get_platform_info() -> PlatformInfo:
    """Get the platform information singleton"""
    global _platform_info
    if _platform_info is None:
        _platform_info = PlatformInfo()
    return _platform_info


def get_notification_command(title: str, message: str) -> Optional[list]:
    """
    Get the system notification command for the current platform

    Args:
        title: Notification title
        message: Notification message

    Returns:
        Command list to execute, or None if not supported
    """
    info = get_platform_info()

    if info.is_macos:
        # Use osascript for macOS notifications
        script = f'display notification "{message}" with title "{title}"'
        return ['osascript', '-e', script]

    elif info.is_linux:
        # Use notify-send for Linux
        return ['notify-send', title, message]

    elif info.is_windows:
        # Use PowerShell for Windows 10/11 toast notifications
        script = f'''
        [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
        $Template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
        $RawXml = [xml] $Template.GetXml()
        ($RawXml.toast.visual.binding.text|where {{$_.id -eq "1"}}).AppendChild($RawXml.CreateTextNode("{title}")) > $null
        ($RawXml.toast.visual.binding.text|where {{$_.id -eq "2"}}).AppendChild($RawXml.CreateTextNode("{message}")) > $null
        $SerializedXml = New-Object Windows.Data.Xml.Dom.XmlDocument
        $SerializedXml.LoadXml($RawXml.OuterXml)
        $Toast = [Windows.UI.Notifications.ToastNotification]::new($SerializedXml)
        $Toast.Tag = "Classmark"
        $Toast.Group = "Classmark"
        $Notifier = [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Classmark")
        $Notifier.Show($Toast);
        '''
        return ['powershell', '-Command', script]

    return None


if __name__ == "__main__":
    # Test platform detection
    info = get_platform_info()
    print(info)
    print(f"System Font: {info.system_font}")
    print(f"Monospace Font: {info.monospace_font}")
    print(f"Native Dialogs: {info.supports_native_dialogs}")
    print(f"Accent Color: {info.accent_color}")
