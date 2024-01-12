import logging
from typing import List
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["duration"] = [item["duration"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    max_audio_size = max([item["audio"].size(1) for item in dataset_items])
    result_batch["audio"] = torch.cat([
        F.pad(item["audio"], (0, max_audio_size - item["audio"].size(1)))
        for item in dataset_items
    ])

    max_spectrogram_size = max([item["spectrogram"].size(2) for item in dataset_items])
    result_batch["spectrogram"] = torch.cat([
        F.pad(item["spectrogram"], (0, max_spectrogram_size - item["spectrogram"].size(2)))
        for item in dataset_items
    ])

    max_text_encoded_size = max([item["text_encoded"].size(1) for item in dataset_items])
    result_batch["text_encoded"] = torch.cat([
        F.pad(item["text_encoded"], (0, max_text_encoded_size - item["text_encoded"].size(1)))
        for item in dataset_items
    ])

    result_batch["text_encoded_length"] = torch.tensor([
        item["text_encoded"].size(1) for item in dataset_items
    ])

    result_batch["spectrogram_length"] = torch.tensor([
        item["spectrogram"].size(2) for item in dataset_items
    ])

    return result_batch