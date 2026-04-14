from collections.abc import Sequence
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset

from question_datasets.arc_easy import ARCEasyDataset
from question_datasets.base import BaseQuestionDataset
from question_datasets.hle import HLEDataset
from question_datasets.mmlu import MMLUDataset


DATASET_REGISTRY = {
    "arc_easy": ARCEasyDataset,
    "mmlu": MMLUDataset,
    "hle": HLEDataset,
}


class CombinedQuestionDataset(Dataset):
    def __init__(self, datasets: Sequence[BaseQuestionDataset], tokenizer: Any):
        self.datasets = list(datasets)
        self.tokenizer = tokenizer
        self.concat_dataset = ConcatDataset(self.datasets)

    def __len__(self) -> int:
        return len(self.concat_dataset)

    def __getitem__(self, idx: int):
        prompt_tokens, _local_idx, seq_len, metadata = self.concat_dataset[idx]
        metadata = {**metadata, "combined_dataset_idx": idx}
        return prompt_tokens, idx, seq_len, metadata

    def collate_fn(self, batch):
        max_seq_len = max(seq_len for _, _, seq_len, _ in batch)
        padded_prompt_tokens = [
            prompt_tokens + [self.tokenizer.pad_id] * (max_seq_len - seq_len)
            for prompt_tokens, _, seq_len, _ in batch
        ]
        return (
            torch.tensor(padded_prompt_tokens, dtype=torch.long),
            torch.tensor([idx for _, idx, _, _ in batch]),
            torch.tensor([seq_len for _, _, seq_len, _ in batch]),
            [metadata for _, _, _, metadata in batch],
        )


def build_combined_question_dataset(
    dataset_names: Sequence[str],
    tokenizer: Any,
    max_token_length: int,
    add_bos_token: bool = False,
) -> CombinedQuestionDataset:
    datasets = []
    for dataset_name in dataset_names:
        dataset_cls = DATASET_REGISTRY[dataset_name]
        datasets.append(
            dataset_cls(
                tokenizer=tokenizer,
                max_token_length=max_token_length,
                add_bos_token=add_bos_token,
            ),
        )
    return CombinedQuestionDataset(datasets=datasets, tokenizer=tokenizer)
