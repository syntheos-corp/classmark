"""
LayoutLMv3 Fine-Tuning for Classification Marking Detection

This script fine-tunes the LayoutLMv3-base model's classification head
on our custom dataset of annotated government documents.

Since no public datasets exist for classification marking detection,
we use our own ground truth annotations (61 documents) and optionally
generate synthetic training data.

Author: Classmark Development Team
Date: 2025-11-10
"""

import os
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    get_linear_schedule_with_warmup
)
from PIL import Image
from pdf2image import convert_from_path
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Classification labels for token classification
LABELS = [
    'O',  # Outside (no classification marking)
    'B-TOP_SECRET',  # Beginning of TOP SECRET
    'I-TOP_SECRET',  # Inside TOP SECRET
    'B-SECRET',  # Beginning of SECRET
    'I-SECRET',  # Inside SECRET
    'B-CONFIDENTIAL',  # Beginning of CONFIDENTIAL
    'I-CONFIDENTIAL',  # Inside CONFIDENTIAL
    'B-CUI',  # Beginning of CUI
    'I-CUI',  # Inside CUI
    'B-CONTROL',  # Beginning of control marking (NOFORN, etc.)
    'I-CONTROL',  # Inside control marking
    'B-AUTHORITY',  # Beginning of authority block
    'I-AUTHORITY',  # Inside authority block
    'B-DECLASSIFICATION',  # Beginning of declassification marking
    'I-DECLASSIFICATION',  # Inside declassification marking
]

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


@dataclass
class TrainingExample:
    """Single training example"""
    image_path: str
    words: List[str]
    bboxes: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    labels: List[int]  # Label IDs
    page_num: int


class ClassificationMarkingDataset(Dataset):
    """Dataset for classification marking detection"""

    def __init__(
        self,
        examples: List[TrainingExample],
        processor: LayoutLMv3Processor,
        max_length: int = 512
    ):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Load image
        image = Image.open(example.image_path).convert("RGB")

        # Prepare encoding
        encoding = self.processor(
            image,
            example.words,
            boxes=example.bboxes,
            word_labels=example.labels,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        return encoding


class LayoutLMv3Finetuner:
    """Fine-tune LayoutLMv3 for classification marking detection"""

    def __init__(
        self,
        model_name: str = 'microsoft/layoutlmv3-base',
        output_dir: str = './models/layoutlmv3-classification-finetuned',
        use_gpu: bool = True,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.max_length = max_length

        # Device
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("✓ Using CPU")

        # Load processor
        print(f"Loading processor from {model_name}...")
        self.processor = LayoutLMv3Processor.from_pretrained(
            model_name,
            apply_ocr=True
        )

        # Load model
        print(f"Loading model with {len(LABELS)} labels...")
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID
        )
        self.model.to(self.device)

    def load_ground_truth_data(
        self,
        ground_truth_file: str = 'ground_truth.json',
        documents_dir: str = 'documents'
    ) -> List[TrainingExample]:
        """
        Load and convert ground truth annotations to training examples

        Args:
            ground_truth_file: Path to ground truth JSON file
            documents_dir: Directory containing PDF files

        Returns:
            List of TrainingExample objects
        """
        print(f"\nLoading ground truth from {ground_truth_file}...")

        with open(ground_truth_file, 'r') as f:
            annotations = json.load(f)

        examples = []
        documents_dir = Path(documents_dir)

        for ann in annotations:
            file_path = documents_dir / ann['file_name']

            if not file_path.exists():
                continue

            # Only process PDFs (LayoutLMv3 needs images)
            if not str(file_path).lower().endswith('.pdf'):
                continue

            print(f"  Processing {ann['file_name']}...")

            # Convert PDF to images
            try:
                images = convert_from_path(str(file_path), dpi=200)
            except Exception as e:
                print(f"    ⚠ Error converting PDF: {e}")
                continue

            # Process each page
            for page_num, image in enumerate(images, 1):
                # Save image temporarily
                image_path = self.output_dir / 'temp_images' / f"{file_path.stem}_page{page_num}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(str(image_path))

                # Use processor to extract text and bboxes
                try:
                    encoding = self.processor(
                        image,
                        return_tensors="pt"
                    )

                    # Extract words and bboxes
                    if 'bbox' in encoding:
                        bboxes = encoding['bbox'][0].cpu().numpy()
                        tokens = self.processor.tokenizer.convert_ids_to_tokens(
                            encoding['input_ids'][0].cpu().numpy()
                        )

                        # Filter out special tokens
                        words = []
                        word_bboxes = []
                        for token, bbox in zip(tokens, bboxes):
                            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                                words.append(token.replace('##', ''))
                                # Denormalize bbox from [0, 1000] to pixel coords
                                x1 = int(bbox[0] * image.width / 1000)
                                y1 = int(bbox[1] * image.height / 1000)
                                x2 = int(bbox[2] * image.width / 1000)
                                y2 = int(bbox[3] * image.height / 1000)
                                word_bboxes.append((x1, y1, x2, y2))

                        # Create labels based on ground truth
                        # For now, use simple heuristic: check if word contains classification terms
                        labels = self._create_labels(words, ann)

                        if len(words) > 0:
                            example = TrainingExample(
                                image_path=str(image_path),
                                words=words,
                                bboxes=word_bboxes,
                                labels=labels,
                                page_num=page_num
                            )
                            examples.append(example)

                except Exception as e:
                    print(f"    ⚠ Error processing page {page_num}: {e}")
                    continue

        print(f"✓ Loaded {len(examples)} training examples")
        return examples

    def _create_labels(self, words: List[str], annotation: Dict) -> List[int]:
        """
        Create BIO labels for words based on ground truth annotation

        Args:
            words: List of words from OCR
            annotation: Ground truth annotation

        Returns:
            List of label IDs
        """
        labels = [LABEL2ID['O']] * len(words)  # Default: outside

        # Get expected classification level
        level = annotation.get('classification_level')
        has_markings = annotation.get('has_classification', False)

        if not has_markings or not level:
            return labels

        # Simple heuristic: mark words that match classification terms
        text = ' '.join(words).upper()

        # Classification levels
        if level == 'TOP SECRET':
            for i, word in enumerate(words):
                if 'TOP' in word.upper():
                    labels[i] = LABEL2ID['B-TOP_SECRET']
                elif 'SECRET' in word.upper() and i > 0 and labels[i-1] == LABEL2ID['B-TOP_SECRET']:
                    labels[i] = LABEL2ID['I-TOP_SECRET']

        elif level == 'SECRET':
            for i, word in enumerate(words):
                if 'SECRET' in word.upper() and 'TOP' not in words[i-1].upper() if i > 0 else True:
                    labels[i] = LABEL2ID['B-SECRET']

        elif level == 'CONFIDENTIAL':
            for i, word in enumerate(words):
                if 'CONFIDENTIAL' in word.upper():
                    labels[i] = LABEL2ID['B-CONFIDENTIAL']

        elif level == 'CUI' or level == 'CONTROLLED UNCLASSIFIED INFORMATION':
            for i, word in enumerate(words):
                if 'CUI' in word.upper() or 'CONTROLLED' in word.upper():
                    labels[i] = LABEL2ID['B-CUI']

        # Control markings
        control_terms = ['NOFORN', 'ORCON', 'IMCON', 'RELIDO', 'PROPIN', 'FISA']
        for i, word in enumerate(words):
            if any(term in word.upper() for term in control_terms):
                labels[i] = LABEL2ID['B-CONTROL']

        # Authority block
        authority_terms = ['CLASSIFIED', 'DERIVED', 'DECLASSIFY', 'REASON']
        for i, word in enumerate(words):
            if any(term in word.upper() for term in authority_terms):
                if i > 0 and labels[i-1] == LABEL2ID['B-AUTHORITY']:
                    labels[i] = LABEL2ID['I-AUTHORITY']
                else:
                    labels[i] = LABEL2ID['B-AUTHORITY']

        # Declassification
        declassif_terms = ['DECLASSIFIED', 'REDACTED', 'SANITIZED', 'FOIA']
        for i, word in enumerate(words):
            if any(term in word.upper() for term in declassif_terms):
                labels[i] = LABEL2ID['B-DECLASSIFICATION']

        return labels

    def train(
        self,
        train_examples: List[TrainingExample],
        val_examples: Optional[List[TrainingExample]] = None
    ):
        """
        Fine-tune the model

        Args:
            train_examples: Training examples
            val_examples: Optional validation examples
        """
        print(f"\n{'='*70}")
        print(f"Fine-Tuning LayoutLMv3 for Classification Marking Detection")
        print(f"{'='*70}\n")

        # Create datasets
        train_dataset = ClassificationMarkingDataset(
            train_examples,
            self.processor,
            self.max_length
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )

        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )

        print(f"Training Configuration:")
        print(f"  Training examples: {len(train_examples)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Total training steps: {total_steps}")
        print(f"  Device: {self.device}")
        print()

        # Training loop
        self.model.train()
        global_step = 0
        best_loss = float('inf')

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print("-" * 70)

            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training")

            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = epoch_loss / len(train_loader)
            print(f"  Average loss: {avg_loss:.4f}\n")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_model('best')
                print(f"  ✓ Saved best model (loss: {best_loss:.4f})\n")

        # Save final model
        self._save_model('final')
        print(f"\n✓ Training complete!")
        print(f"  Best loss: {best_loss:.4f}")
        print(f"  Models saved to: {self.output_dir}")

    def _save_model(self, checkpoint_name: str):
        """Save model checkpoint"""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)

        print(f"    Saved to: {checkpoint_dir}")


def main():
    """Main fine-tuning pipeline"""
    print("="*70)
    print("LayoutLMv3 Fine-Tuning for Classification Marking Detection")
    print("="*70)
    print()

    # Initialize finetuner
    finetuner = LayoutLMv3Finetuner(
        model_name='microsoft/layoutlmv3-base',
        output_dir='./models/layoutlmv3-classification-finetuned',
        use_gpu=True,
        learning_rate=5e-5,
        batch_size=4,
        num_epochs=10,
        warmup_steps=500,
        max_length=512
    )

    # Load ground truth data
    examples = finetuner.load_ground_truth_data(
        ground_truth_file='ground_truth.json',
        documents_dir='documents'
    )

    if len(examples) == 0:
        print("⚠ No training examples found!")
        print("Make sure ground_truth.json exists and contains PDF documents")
        return

    # Split into train/val
    train_examples, val_examples = train_test_split(
        examples,
        test_size=0.2,
        random_state=42
    )

    print(f"\nDataset split:")
    print(f"  Training: {len(train_examples)} examples")
    print(f"  Validation: {len(val_examples)} examples")
    print()

    # Fine-tune
    finetuner.train(train_examples, val_examples)

    print("\n" + "="*70)
    print("Fine-tuning complete!")
    print("="*70)


if __name__ == "__main__":
    main()
