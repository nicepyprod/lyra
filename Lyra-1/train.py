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
        default=4,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
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
        default=50,
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
    )

    args = parser.parse_args()
    return args


def setup_logging(accelerator):
    """Configure logging for the training run."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if accelerator.is_local_main_process:
        logger.info("Logging initialized for main process.")


def main():
    """Main entry point for the training script."""
    args = parse_args()

    # Initialize accelerator
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_config=project_config,
    )

    setup_logging(accelerator)

    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    logger.info(f"Training configuration: {args}")
    logger.info(f"Accelerator state: {accelerator.state}")
    logger.info("Training script initialized. Implement model and data loading here.")


if __name__ == "__main__":
    main()
