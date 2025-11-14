#!/usr/bin/env python3
"""
Classmark Model Downloader

Downloads all required models and creates an offline model cache.
This script should be run once during installation to enable fully
offline operation.

Models Downloaded:
- LayoutLMv3-base (~500MB) - Visual pattern detection
- YOLOv8n (~6MB) - Text region detection

After running this script, Classmark can operate completely offline.

Usage:
    python download_models.py [--models-dir PATH]

Author: Classmark Development Team
Date: 2025-11-10
"""

import os
import sys
import argparse
from pathlib import Path
import requests
from tqdm import tqdm


def download_file(url: str, output_path: str):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as file, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress_bar.update(size)


def download_layoutlmv3(models_dir: str):
    """Download LayoutLMv3-base model"""
    print("\n" + "="*70)
    print("Downloading LayoutLMv3-base Model")
    print("="*70)

    model_dir = os.path.join(models_dir, "layoutlmv3-base")
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Use HuggingFace transformers to download
        from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

        print("Downloading processor...")
        processor = LayoutLMv3Processor.from_pretrained(
            'microsoft/layoutlmv3-base',
            cache_dir=model_dir
        )

        print("Downloading model (125M parameters, ~500MB)...")
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            'microsoft/layoutlmv3-base',
            cache_dir=model_dir
        )

        print("✓ LayoutLMv3 downloaded successfully")
        return True

    except Exception as e:
        print(f"✗ Error downloading LayoutLMv3: {e}")
        return False


def download_yolo(models_dir: str):
    """Download YOLOv8n model"""
    print("\n" + "="*70)
    print("Downloading YOLOv8n Model")
    print("="*70)

    model_dir = os.path.join(models_dir, "yolo")
    os.makedirs(model_dir, exist_ok=True)

    try:
        # Use ultralytics to download
        from ultralytics import YOLO

        print("Downloading YOLOv8n (~6MB)...")
        model = YOLO('yolov8n.pt')

        # Move to models directory
        model_path = os.path.join(model_dir, 'yolov8n.pt')
        if os.path.exists('yolov8n.pt'):
            os.rename('yolov8n.pt', model_path)

        print("✓ YOLOv8n downloaded successfully")
        return True

    except Exception as e:
        print(f"✗ Error downloading YOLOv8n: {e}")
        return False


def create_config(models_dir: str):
    """Create configuration file for offline operation"""
    try:
        from offline_config import OfflineConfig, OfflineConfigManager, create_offline_config

        # Create offline configuration
        config = create_offline_config(models_dir)

        # Save to models directory
        config_path = os.path.join(models_dir, "models_config.json")
        manager = OfflineConfigManager()
        manager.save_config(config, config_path)

        print(f"\n✓ Offline configuration created and saved to {config_path}")
        print("✓ Copy this file to your project directory or ~/.classmark/ to enable offline mode")

    except ImportError:
        # Fallback if offline_config module not available
        config = {
            "offline_mode": True,
            "models_dir": models_dir,
            "layoutlmv3_path": os.path.join(models_dir, "layoutlmv3-base"),
            "yolo_path": os.path.join(models_dir, "yolo", "yolov8n.pt")
        }

        config_path = os.path.join(models_dir, "models_config.json")

        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Configuration saved to {config_path}")


def verify_models(models_dir: str):
    """Verify all models are present and loadable"""
    print("\n" + "="*70)
    print("Verifying Models")
    print("="*70)

    # Try to use offline_config module for verification
    try:
        from offline_config import OfflineConfigManager, create_offline_config

        # Create config for this models directory
        config = create_offline_config(models_dir)

        # Create temporary config manager
        config_path = os.path.join(models_dir, "models_config.json")
        if os.path.exists(config_path):
            manager = OfflineConfigManager(config_path)
        else:
            # Create temporary manager with this config
            manager = OfflineConfigManager()
            manager.config = config

        # Use the manager's verification
        return manager.verify_offline_models()

    except ImportError:
        # Fallback to manual verification
        all_good = True

        # Check LayoutLMv3
        layoutlmv3_dir = os.path.join(models_dir, "layoutlmv3-base")
        if os.path.exists(layoutlmv3_dir):
            print("✓ LayoutLMv3 directory exists")

            try:
                from transformers import LayoutLMv3Processor
                processor = LayoutLMv3Processor.from_pretrained(
                    layoutlmv3_dir,
                    local_files_only=True
                )
                print("✓ LayoutLMv3 can be loaded")
            except Exception as e:
                print(f"✗ LayoutLMv3 load error: {e}")
                all_good = False
        else:
            print("✗ LayoutLMv3 not found")
            all_good = False

        # Check YOLO
        yolo_path = os.path.join(models_dir, "yolo", "yolov8n.pt")
        if os.path.exists(yolo_path):
            print("✓ YOLOv8n file exists")

            try:
                from ultralytics import YOLO
                model = YOLO(yolo_path)
                print("✓ YOLOv8n can be loaded")
            except Exception as e:
                print(f"✗ YOLOv8n load error: {e}")
                all_good = False
        else:
            print("✗ YOLOv8n not found")
            all_good = False

        return all_good


def get_model_size(models_dir: str):
    """Calculate total size of downloaded models"""
    total_size = 0

    for root, dirs, files in os.walk(models_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)

    # Convert to MB
    size_mb = total_size / (1024 * 1024)
    return size_mb


def main():
    parser = argparse.ArgumentParser(description='Download Classmark models for offline use')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='Directory to store models (default: ./models)')
    parser.add_argument('--skip-layoutlmv3', action='store_true',
                       help='Skip LayoutLMv3 download (visual detection disabled)')
    parser.add_argument('--skip-yolo', action='store_true',
                       help='Skip YOLO download (OCR on scanned docs disabled)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing models, do not download')

    args = parser.parse_args()

    # Create models directory
    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)

    print("="*70)
    print("Classmark Model Downloader")
    print("="*70)
    print(f"\nModels will be saved to: {models_dir}")
    print(f"Estimated download size: ~500-600 MB")
    print(f"Internet connection required for this step only")
    print()

    if not args.verify_only:
        input("Press Enter to continue or Ctrl+C to cancel...")

    # Verify only mode
    if args.verify_only:
        if verify_models(models_dir):
            print("\n✓ All models verified successfully")
            print("Classmark is ready for offline operation")
            return 0
        else:
            print("\n✗ Model verification failed")
            print("Run without --verify-only to download models")
            return 1

    # Download models
    success = True

    if not args.skip_layoutlmv3:
        if not download_layoutlmv3(models_dir):
            success = False
    else:
        print("\n⚠ Skipping LayoutLMv3 (visual detection will be unavailable)")

    if not args.skip_yolo:
        if not download_yolo(models_dir):
            success = False
    else:
        print("\n⚠ Skipping YOLO (OCR on scanned documents will be unavailable)")

    # Create config
    if success:
        create_config(models_dir)

    # Verify
    if verify_models(models_dir):
        print("\n" + "="*70)
        print("Download Complete!")
        print("="*70)

        size_mb = get_model_size(models_dir)
        print(f"\nTotal models size: {size_mb:.1f} MB")
        print(f"Models location: {models_dir}")
        print("\n✓ Classmark is now ready for OFFLINE operation")
        print("✓ No internet connection required for future use")
        print("\nNext steps:")
        print("  1. Run the installer to package the application")
        print("  2. Include the models/ directory in the installer")
        print("  3. Distribute the complete package to users")

        return 0
    else:
        print("\n✗ Download or verification failed")
        print("Please check your internet connection and try again")
        return 1


if __name__ == "__main__":
    sys.exit(main())
