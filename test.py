import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader
from tqdm import tqdm

from question_datasets import (
    ARCEasyDataset,
    HLEDataset,
    MMLUDataset,
    build_combined_question_dataset,
)


@dataclass
class MockTokenizer:
    pad_id: int = 0

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        tokens = [len(token) % 32000 for token in text.split()]
        if bos:
            tokens = [1] + tokens
        if eos:
            tokens = tokens + [2]
        return tokens


def print_sample(dataset_name: str, dataset, sample_count: int) -> None:
    print(f"\n=== {dataset_name} | dataset_len={len(dataset)} ===")
    for idx in range(min(sample_count, len(dataset))):
        prompt_tokens, sample_idx, seq_len, metadata = dataset[idx]
        sample_preview = {
            "sample_idx": sample_idx,
            "seq_len": seq_len,
            "first_12_tokens": prompt_tokens[:12],
            "metadata": metadata,
        }
        print(json.dumps(sample_preview, indent=2, ensure_ascii=False))


def print_batch(dataset_name: str, dataset, batch_size: int) -> None:
    if len(dataset) == 0:
        print(f"\n--- {dataset_name} | first_batch_skipped ---")
        print("Dataset is empty after filtering.")
        return
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    batch_tokens, batch_indices, batch_seq_lens, batch_metadata = next(iter(dataloader))
    batch_preview = {
        "batch_tokens_shape": list(batch_tokens.shape),
        "batch_indices": batch_indices.tolist(),
        "batch_seq_lens": batch_seq_lens.tolist(),
        "first_metadata": batch_metadata[0] if batch_metadata else None,
    }
    print(f"\n--- {dataset_name} | first_batch ---")
    print(json.dumps(batch_preview, indent=2, ensure_ascii=False))


def print_dataset_summary(loaded_datasets: list[tuple[str, Any]]) -> None:
    print("\n=== DATASET SUMMARY ===")
    total = 0
    for dataset_name, dataset in loaded_datasets:
        dataset_len = len(dataset)
        total += dataset_len
        print(f"{dataset_name}: {dataset_len}")
    print(f"TOTAL_INDIVIDUAL: {total}")


def dump_datasets_to_csv(loaded_datasets: list[tuple[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        "dataset_name",
        "sample_idx",
        "seq_len",
        "prompt_text",
        "question_text",
        "difficulty_label",
        "source_dataset",
        "source_split",
        "source_id",
        "subject",
        "question_type",
        "choices",
        "gold_answer",
        "has_image",
        "token_count",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        total_rows = sum(len(dataset) for _, dataset in loaded_datasets)
        progress_bar = tqdm(total=total_rows, desc="Dumping CSV")

        for dataset_name, dataset in loaded_datasets:
            for idx in range(len(dataset)):
                _, sample_idx, seq_len, metadata = dataset[idx]
                writer.writerow(
                    {
                        "dataset_name": dataset_name,
                        "sample_idx": sample_idx,
                        "seq_len": seq_len,
                        "prompt_text": metadata.get("prompt_text"),
                        "question_text": metadata.get("question_text"),
                        "difficulty_label": metadata.get("difficulty_label"),
                        "source_dataset": metadata.get("source_dataset"),
                        "source_split": metadata.get("source_split"),
                        "source_id": metadata.get("source_id"),
                        "subject": metadata.get("subject"),
                        "question_type": metadata.get("question_type"),
                        "choices": json.dumps(metadata.get("choices"), ensure_ascii=False),
                        "gold_answer": metadata.get("gold_answer"),
                        "has_image": metadata.get("has_image"),
                        "token_count": metadata.get("token_count"),
                    },
                )
                progress_bar.update(1)

        progress_bar.close()

    print(f"\n=== CSV DUMP ===")
    print(f"csv_path: {csv_path}")
    print(f"rows_written: {total_rows}")


def try_build_dataset(dataset_name: str, dataset_cls: Any, tokenizer: MockTokenizer, max_token_length: int):
    try:
        dataset = dataset_cls(
            tokenizer=tokenizer,
            max_token_length=max_token_length,
            add_bos_token=False,
        )
        return dataset
    except Exception as exc:
        print(f"\n=== {dataset_name} | load_failed ===")
        print(f"{type(exc).__name__}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_count", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_token_length", type=int, default=192)
    parser.add_argument("--dump_csv", action="store_true")
    parser.add_argument("--csv_path", type=Path, default=Path("question_dataset_dump.csv"))
    args = parser.parse_args()

    tokenizer = MockTokenizer()

    dataset_builders = [
        ("ARC-Easy", ARCEasyDataset),
        ("MMLU", MMLUDataset),
        ("HLE", HLEDataset),
    ]

    available_dataset_names = []
    loaded_datasets = []

    for dataset_name, dataset_cls in dataset_builders:
        dataset = try_build_dataset(
            dataset_name=dataset_name,
            dataset_cls=dataset_cls,
            tokenizer=tokenizer,
            max_token_length=args.max_token_length,
        )
        if dataset is None:
            continue
        available_dataset_names.append(dataset_name)
        loaded_datasets.append((dataset_name, dataset))
        print_sample(dataset_name, dataset, args.sample_count)
        print_batch(dataset_name, dataset, args.batch_size)

    print_dataset_summary(loaded_datasets)

    if args.dump_csv:
        dump_datasets_to_csv(loaded_datasets, args.csv_path.resolve())

    combined_dataset_names = []
    if "ARC-Easy" in available_dataset_names:
        combined_dataset_names.append("arc_easy")
    if "MMLU" in available_dataset_names:
        combined_dataset_names.append("mmlu")
    if "HLE" in available_dataset_names:
        combined_dataset_names.append("hle")

    if combined_dataset_names:
        combined_dataset = build_combined_question_dataset(
            dataset_names=combined_dataset_names,
            tokenizer=tokenizer,
            max_token_length=args.max_token_length,
            add_bos_token=False,
            num_samples={"mmlu": 2200},
        )
        print_sample("COMBINED", combined_dataset, args.sample_count)
        print_batch("COMBINED", combined_dataset, args.batch_size)
        print("\n=== COMBINED SUMMARY ===")
        print(f"combined_dataset_len: {len(combined_dataset)}")
        for dataset_name, dataset in loaded_datasets:
            print(f"{dataset_name}_contribution: {len(dataset)}")
    else:
        print("\n=== COMBINED | skipped ===")
        print("No datasets were available to combine.")


if __name__ == "__main__":
    main()
