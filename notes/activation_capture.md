# Activation Capture: how it works (summary)

**Files referenced**
- [llama3_interpretability_sae/capture_activations.py](llama3_interpretability_sae/capture_activations.py)
- [llama3_interpretability_sae/llama_3/model_text_only.py](llama3_interpretability_sae/llama_3/model_text_only.py)

## Overview
This document describes how residual activations are captured from selected transformer layers during a forward pass and how they are saved to disk asynchronously.

## High-level flow
1. Distributed setup: the script initializes a torch distributed process group (NCCL). Each process uses a single GPU and a rank.
2. Dataset & dataloader: a dataset is created (OpenWebText or QA combined), wrapped in a `DistributedSampler`, and loaded with `DataLoader`.
3. Model loading: `Transformer` is constructed using `ModelArgs`, weights are loaded with `torch.load(..., weights_only=True)`, moved to GPU, and set to `eval()`.
4. Capture configuration: `store_layer_activ` lists the layer indices to capture (e.g., `[16, 22]`). These layers have their `capture_activation` flag set when `Transformer` is instantiated.
5. Asynchronous saving: a `multiprocessing.Queue` and a `Process` running `save_activations_process` are created so GPU capture can continue while disk writes happen in the background.
6. Batch processing: for each batch the script runs a forward pass and extracts per-layer residual activations, trims them to their true sequence lengths, and enqueues them for saving. When complete, a `None` sentinel is enqueued to stop the saver process.

## Where activations are captured
- In `llama_3/model_text_only.py` the `TransformerBlock` does:
  - `x_normalized = self.attention_norm(x)`
  - If `capture_activation` is True, `self.residual_activations = x_normalized.detach().clone()` (keeps tensor on the model device to avoid extra transfers during capture).
  - `self.sae_forward_fn` (optional) would be applied next if provided, but capturing happens before any SAE processing.
- The `Transformer.get_layer_residual_activs()` method returns a dict mapping layer id -> `residual_activations.cpu()` for all layers in `self.store_layer_activ`.

Notes on shapes and memory:
- `residual_activations` is a tensor shaped `(batch_size, seq_len, dim)`.
- The clone/detach keeps the GPU tensor until `get_layer_residual_activs()` moves it to CPU (`.cpu()`), minimizing frequent device transfers during the forward loop.

## capture_activations loop (in `capture_activations.py`)
1. For each dataloader batch:
   - Batch elements: `(batch, indices, seq_lens[, metadata])`.
   - Move `batch` to the GPU device.
   - Run `with torch.no_grad(): model(batch, start_pos=0)`.
   - Call `model.get_layer_residual_activs()` to receive a dict of CPU tensors for each tracked layer.
2. Trim activations per example:
   - For each layer tensor, iterate the first dimension (examples in the batch) and slice `act[:seq_len]` to remove padding, producing a list of variable-length tensors.
   - The code uses `zip(activations, seq_lens, strict=True)` to pair each example's activation with its true sequence length.
3. Enqueue the trimmed activations along with `indices` and optional `metadata` into the `save_activation_queue`.
4. When all batches assigned to the process are processed, `None` is enqueued as a stop signal for the saver process.

## save_activations_process
- Runs in a separate process and reads from the `Queue`.
- For each item `(layer_activations, indices, metadata_batch)`:
  - For each `(layer, activations)` pair, create (if not existing) `activation_out_dir/layer_{layer}` and save each example activation as `activations_l{layer}_idx{dataset_idx}.pt` using `torch.save`.
  - If `metadata` is present, append a JSON line to `metadata_rank{rank}.jsonl` with the `activation_path`, `capture_dataset_idx`, and `capture_layer` fields.
- Stops when it receives `None`.

## Important code details & considerations
- The capture happens right after layer normalization inside each transformer block (`attention_norm`), i.e., these are residual (pre-attention/FFN) activations.
- The code uses `.detach().clone()` to ensure the captured tensors are not part of the autograd graph and are a separate memory allocation.
- `get_layer_residual_activs()` moves captures to CPU for safe cross-process queuing and to reduce GPU memory pressure during the save step; the main process thus retains GPU availability for further batches.
- The saver process writes one file per example per layer. This is straightforward to load later, but results in many small files. Consider batching or alternative storage formats (e.g., sharded tensors, HDF5) if filesystem overhead becomes an issue.
- The `Queue` size is polled and the main loop sleeps if the queue grows beyond 10 items to avoid excessive memory growth.
- The forward call sets `start_pos=0` (so key-value caching is not being used to extend context across calls here). The model supports KV caching internally (`Attention.cache_k` / `cache_v`), but for the capture flow the sequence is handled per-batch.

## Where to look in code
- Main capture loop: [llama3_interpretability_sae/capture_activations.py](llama3_interpretability_sae/capture_activations.py)
- Capture logic inside transformer block: [llama3_interpretability_sae/llama_3/model_text_only.py](llama3_interpretability_sae/llama_3/model_text_only.py) (`TransformerBlock` and `Transformer.get_layer_residual_activs()`)
- Saver process: `save_activations_process` in `capture_activations.py`.

## Quick recommendations (optional)
- If you want fewer files: change saver to accumulate per-layer batched tensors and save per-batch/shard files.
- If you need activation alignment across distributed ranks: ensure consistent dataset indexing and optionally include global offsets in metadata.
- Consider compressing or using a binary container if activations become large on disk.

---
Generated from reading the capture script and the Transformer block capture points.
