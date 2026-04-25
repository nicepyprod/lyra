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
        max_audio_length: float = 30.0,
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
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # Truncate to max length
        max_samples = int(self.max_audio_length * self.sample_rate)
        waveform = waveform[:, :max_samples]

        return waveform.squeeze(0)  # (T,)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        audio_path = sample["audio_path"]
        text = sample["text"]

        waveform = self._load_audio(audio_path)

        # Extract audio features
        audio_inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )

        # Tokenize text
        text_inputs = self.tokenizer(
            text,
            max_length=self.max_text_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_features": audio_inputs.input_features.squeeze(0),
            "input_ids": text_inputs.input_ids.squeeze(0),
            "attention_mask": text_inputs.attention_mask.squeeze(0),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate a list of dataset samples into a padded batch.

    Args:
        batch: List of dicts from AudioTextDataset.__getitem__.

    Returns:
        Dict of padded tensors ready for model input.
    """
    input_features = torch.stack([item["input_features"] for item in batch])

    max_text_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_text_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]

    return {
        "input_features": input_features,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
