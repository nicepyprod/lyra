#!/usr/bin/env python3
"""Main training script for Lyra-1 model.

This script handles the training pipeline including data loading,
model initialization, and the training loop with accelerate support.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse command-line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Lyra-1 model")

    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model or model identifier from HuggingFace Hub.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/accelerate/accelerate_config.yaml",
        help="Path to accelerate configuration file.",
    )

    # Data arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="Directory containing training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints and logs.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,  # lowered from 4 to avoid OOM on my 8GB GPU
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,  # lowered from 1e-4; felt too aggressive for fine-tuning
        help="Initial learning rate (after warmup).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for linear learning rate warmup.",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=25,  # more frequent logging to keep a closer eye on loss curves
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
