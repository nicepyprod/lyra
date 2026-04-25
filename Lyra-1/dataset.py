"""Dataset utilities for Lyra-1 training.

Provides dataset classes and collation functions for loading and preprocessing
multimodal (text + audio) training data.
"""

import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any, Tuple


class AudioTextDataset(Dataset):
    """Dataset for paired audio and text samples.

    Loads audio files and their corresponding text transcriptions/captions
    from a JSON manifest file. Each entry in the manifest should have:
        - 'audio_path': relative or absolute path to the audio file
        - 'text': the associated text string
        - (optional) 'duration': audio duration in seconds
    """

    def __init__(
        self,
        manifest_path: str,
        tokenizer,
        feature_extractor,
        max_audio_length: float = 20.0,  # reduced from 30.0 - my data is mostly <20s and this saves memory
        max_text_length: int = 512,
        sample_rate: int = 16000,
        data_root: Optional[str] = None,
    ):
        """
        Args:
            manifest_path: Path to JSON lines manifest file.
            tokenizer: HuggingFace tokenizer for text encoding.
            feature_extractor: HuggingFace feature extractor for audio.
            max_audio_length: Maximum audio length in seconds to keep.
            max_text_length: Maximum token length for text sequences.
            sample_rate: Target sample rate for audio resampling.
            data_root: Optional root directory prepended to audio paths.
        """
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        self.data_root = data_root

        self.samples = self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: str) -> List[Dict[str, Any]]:
        """Load and filter samples from a JSON lines manifest."""
        samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                # Filter by duration if provided
                duration = entry.get("duration", None)
                if duration is not None and duration > self.max_audio_length:
                    continue
                samples.append(entry)
        return samples

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and resample audio to the target sample rate."""
        if self.data_root:
            audio_path = os.path.join(self.data_root, audio_path)

        waveform, orig_sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.R
