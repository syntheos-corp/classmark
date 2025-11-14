#!/usr/bin/env python3
"""
Offline Configuration Manager

Manages configuration for offline operation of Classmark, including:
- Model cache directory configuration
- Offline mode detection and enforcement
- Model verification
- Path resolution for local models

Author: Classmark Development Team
Date: 2025-11-10
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class OfflineConfig:
    """Configuration for offline operation"""
    offline_mode: bool = False
    models_dir: Optional[str] = None
    layoutlmv3_path: Optional[str] = None
    yolo_path: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if offline configuration is valid"""
        if not self.offline_mode:
            return True

        # In offline mode, all paths must exist
        if not self.models_dir or not os.path.exists(self.models_dir):
            return False

        if not self.layoutlmv3_path or not os.path.exists(self.layoutlmv3_path):
            return False

        if not self.yolo_path or not os.path.exists(self.yolo_path):
            return False

        return True

    def get_missing_models(self) -> list:
        """Get list of missing models"""
        missing = []

        if self.offline_mode:
            if not self.models_dir or not os.path.exists(self.models_dir):
                missing.append("models_dir")

            if not self.layoutlmv3_path or not os.path.exists(self.layoutlmv3_path):
                missing.append("layoutlmv3")

            if not self.yolo_path or not os.path.exists(self.yolo_path):
                missing.append("yolo")

        return missing


class OfflineConfigManager:
    """Manages offline configuration for Classmark"""

    # Default configuration locations (in order of priority)
    CONFIG_LOCATIONS = [
        './models/models_config.json',  # Local models directory
        './models_config.json',  # Current directory
        '~/.classmark/config.json',  # User home directory
        '/etc/classmark/config.json',  # System-wide (Linux/Mac)
    ]

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file (optional)
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> OfflineConfig:
        """Load configuration from file"""

        # Try specified path first
        if self.config_path:
            config_data = self._read_config_file(self.config_path)
            if config_data:
                return self._parse_config(config_data)

        # Try default locations
        for location in self.CONFIG_LOCATIONS:
            expanded_path = os.path.expanduser(location)
            config_data = self._read_config_file(expanded_path)
            if config_data:
                self.config_path = expanded_path
                return self._parse_config(config_data)

        # No config found - return default (online mode)
        return OfflineConfig(offline_mode=False)

    def _read_config_file(self, path: str) -> Optional[Dict[str, Any]]:
        """Read configuration file"""
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read config from {path}: {e}")

        return None

    def _parse_config(self, data: Dict[str, Any]) -> OfflineConfig:
        """Parse configuration data"""
        return OfflineConfig(
            offline_mode=data.get('offline_mode', False),
            models_dir=data.get('models_dir'),
            layoutlmv3_path=data.get('layoutlmv3_path'),
            yolo_path=data.get('yolo_path')
        )

    def save_config(self, config: OfflineConfig, path: Optional[str] = None):
        """
        Save configuration to file

        Args:
            config: Configuration to save
            path: Path to save to (optional, uses current config_path if not specified)
        """
        save_path = path or self.config_path or self.CONFIG_LOCATIONS[0]
        save_path = os.path.expanduser(save_path)

        # Create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Convert to dict
        config_data = {
            'offline_mode': config.offline_mode,
            'models_dir': config.models_dir,
            'layoutlmv3_path': config.layoutlmv3_path,
            'yolo_path': config.yolo_path
        }

        # Save
        with open(save_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ Configuration saved to {save_path}")

    def verify_offline_models(self) -> bool:
        """
        Verify all required models are present for offline operation

        Returns:
            True if all models are present and valid
        """
        if not self.config.offline_mode:
            print("Not in offline mode - skipping verification")
            return True

        print("Verifying offline models...")

        if not self.config.is_valid():
            missing = self.config.get_missing_models()
            print(f"✗ Missing models: {', '.join(missing)}")
            return False

        # Verify LayoutLMv3
        try:
            from transformers import LayoutLMv3Processor
            processor = LayoutLMv3Processor.from_pretrained(
                self.config.layoutlmv3_path,
                local_files_only=True
            )
            print("✓ LayoutLMv3 model verified")
        except Exception as e:
            print(f"✗ LayoutLMv3 verification failed: {e}")
            return False

        # Verify YOLO
        try:
            from ultralytics import YOLO
            model = YOLO(self.config.yolo_path)
            print("✓ YOLOv8n model verified")
        except Exception as e:
            print(f"✗ YOLOv8n verification failed: {e}")
            return False

        print("✓ All offline models verified successfully")
        return True

    def get_layoutlmv3_config(self) -> Dict[str, Any]:
        """
        Get LayoutLMv3 configuration for model loading

        Returns:
            Dict with model_name and kwargs for from_pretrained()
        """
        if self.config.offline_mode:
            return {
                'model_name': self.config.layoutlmv3_path,
                'kwargs': {'local_files_only': True}
            }
        else:
            return {
                'model_name': 'microsoft/layoutlmv3-base',
                'kwargs': {}
            }

    def get_yolo_config(self) -> str:
        """
        Get YOLO model path

        Returns:
            Path to YOLO model file
        """
        if self.config.offline_mode:
            return self.config.yolo_path
        else:
            return 'yolov8n.pt'


def create_offline_config(models_dir: str) -> OfflineConfig:
    """
    Create offline configuration from models directory

    Args:
        models_dir: Path to models directory

    Returns:
        OfflineConfig instance
    """
    models_dir = os.path.abspath(models_dir)

    return OfflineConfig(
        offline_mode=True,
        models_dir=models_dir,
        layoutlmv3_path=os.path.join(models_dir, "layoutlmv3-base"),
        yolo_path=os.path.join(models_dir, "yolo", "yolov8n.pt")
    )


# Global configuration instance
_global_config_manager = None


def get_config_manager() -> OfflineConfigManager:
    """Get global configuration manager instance"""
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = OfflineConfigManager()

    return _global_config_manager


def set_config_manager(config_path: str):
    """Set global configuration manager with specific config file"""
    global _global_config_manager
    _global_config_manager = OfflineConfigManager(config_path)


if __name__ == "__main__":
    # Test configuration
    import sys

    if len(sys.argv) > 1:
        # Verify specified config
        manager = OfflineConfigManager(sys.argv[1])
    else:
        # Use default config
        manager = OfflineConfigManager()

    print(f"Configuration loaded from: {manager.config_path or 'default'}")
    print(f"Offline mode: {manager.config.offline_mode}")

    if manager.config.offline_mode:
        print(f"Models directory: {manager.config.models_dir}")
        print(f"LayoutLMv3 path: {manager.config.layoutlmv3_path}")
        print(f"YOLO path: {manager.config.yolo_path}")

        # Verify
        if manager.verify_offline_models():
            print("\n✓ Ready for offline operation")
            sys.exit(0)
        else:
            print("\n✗ Offline verification failed")
            print("Run download_models.py to download models")
            sys.exit(1)
    else:
        print("\nOnline mode - will download models as needed")
        sys.exit(0)
