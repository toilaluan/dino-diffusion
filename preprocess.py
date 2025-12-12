"""
Optimized preprocessing pipeline with:
1. Async file saving to avoid I/O bottlenecks
2. Dataset.map() for pretokenization (parallelized across CPU cores)
3. DataLoader prefetching for GPU utilization
"""
import torch
import asyncio
import numpy as np
import os
import io
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, Any
import argparse

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoTokenizer
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# Import your custom modules
from dino_diff.models.dino_sampling import DinoSampler
from dino_diff.models.text_embedder import T5TextEmbedder

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DINO_PRETRAINED = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINO_BREAK_AT_LAYER = 6
TEXT_PRETRAINED = "google/flan-t5-xl"


# ============================================================================
# STEP 1: Pretokenize dataset using dataset.map()
# ============================================================================

def get_tokenizer():
    """Get the T5 tokenizer."""
    return AutoTokenizer.from_pretrained(TEXT_PRETRAINED)


def pretokenize_batch(batch: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """
    Tokenize a batch of captions. Used with dataset.map().
    
    Args:
        batch: Batch from dataset with 'txt' column
        tokenizer: HuggingFace tokenizer
        
    Returns:
        Batch with added tokenized fields
    """
    captions = batch['txt']
    
    tokenized = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",  # Use numpy for dataset storage
    )
    
    return {
        **batch,
        "input_ids": tokenized["input_ids"].tolist(),
        "attention_mask": tokenized["attention_mask"].tolist(),
    }


def pretokenize_dataset(dataset, num_proc: int = 8):
    """
    Pretokenize the entire dataset using dataset.map() for parallelization.
    
    Args:
        dataset: HuggingFace dataset
        num_proc: Number of processes for parallel tokenization
        
    Returns:
        Dataset with tokenized fields added
    """
    print(f"Pretokenizing dataset with {num_proc} processes...")
    tokenizer = get_tokenizer()
    
    # Use partial to pass tokenizer to the mapping function
    tokenize_fn = partial(pretokenize_batch, tokenizer=tokenizer)
    
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,  # Process 1000 samples at a time
        num_proc=num_proc,
        desc="Tokenizing captions",
        remove_columns=[],  # Keep all original columns
    )
    
    print(f"Tokenization complete!")
    return tokenized_dataset


# ============================================================================
# STEP 2: Async file saving
# ============================================================================

class AsyncBatchSaver:
    """
    Handles async saving of batch data to disk.
    Uses a thread pool for non-blocking I/O.
    """
    
    def __init__(self, cache_dirs: Dict[str, str], max_workers: int = 4):
        """
        Args:
            cache_dirs: Dictionary with cache directory paths
            max_workers: Number of threads for async saving
        """
        self.cache_dirs = cache_dirs
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks = []
        self.loop = None
        
    def _save_numpy_sync(self, filepath: str, data: np.ndarray):
        """Synchronous numpy save (runs in thread pool)."""
        np.save(filepath, data)
        
    async def save_sample(self, idx: int, data: Dict[str, Any]):
        """
        Save a single sample asynchronously.
        
        Args:
            idx: Sample index
            data: Dictionary containing arrays to save
        """
        loop = asyncio.get_event_loop()
        
        # Create save tasks for each component
        tasks = []
        
        # Text embeddings
        tasks.append(loop.run_in_executor(
            self.executor,
            self._save_numpy_sync,
            os.path.join(self.cache_dirs["text_embeds"], f"{idx:06d}.npy"),
            data["text_embeds"]
        ))
        
        # Text inputs (tokenized) - save as dict
        text_input_data = {
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"]
        }
        tasks.append(loop.run_in_executor(
            self.executor,
            self._save_numpy_sync,
            os.path.join(self.cache_dirs["text_inputs"], f"{idx:06d}.npy"),
            text_input_data
        ))
        
        # DINO intermediate features
        tasks.append(loop.run_in_executor(
            self.executor,
            self._save_numpy_sync,
            os.path.join(self.cache_dirs["clean_inter"], f"{idx:06d}.npy"),
            data["clean_inter"]
        ))
        
        # DINO final features
        tasks.append(loop.run_in_executor(
            self.executor,
            self._save_numpy_sync,
            os.path.join(self.cache_dirs["clean_final"], f"{idx:06d}.npy"),
            data["clean_final"]
        ))
        
        await asyncio.gather(*tasks)
    
    async def save_batch(self, start_idx: int, batch_data: Dict[str, Any]):
        """
        Save an entire batch asynchronously.
        
        Args:
            start_idx: Starting index for this batch
            batch_data: Dictionary containing batched tensors
        """
        batch_size = batch_data["text_embeds"].shape[0]
        
        tasks = []
        for i in range(batch_size):
            sample_data = {
                "text_embeds": batch_data["text_embeds"][i],
                "input_ids": batch_data["input_ids"][i],
                "attention_mask": batch_data["attention_mask"][i],
                "clean_inter": batch_data["clean_inter"][i],
                "clean_final": batch_data["clean_final"][i],
            }
            tasks.append(self.save_sample(start_idx + i, sample_data))
        
        await asyncio.gather(*tasks)
    
    def schedule_batch_save(self, start_idx: int, batch_data: Dict[str, Any]):
        """
        Schedule a batch save to run in the background.
        Returns immediately, save happens asynchronously.
        """
        if self.loop is None:
            self.loop = asyncio.new_event_loop()
        
        task = asyncio.ensure_future(
            self.save_batch(start_idx, batch_data),
            loop=self.loop
        )
        self.pending_tasks.append(task)
        
    async def wait_all(self):
        """Wait for all pending save tasks to complete."""
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks)
            self.pending_tasks.clear()
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


# ============================================================================
# STEP 3: Dataset class for pretokenized data
# ============================================================================

class PretokenizedImageDataset(Dataset):
    """
    Dataset for pretokenized data. Images are processed on-the-fly,
    but text is already tokenized.
    """
    
    def __init__(self, dataset, image_processor):
        """
        Args:
            dataset: Pretokenized HuggingFace dataset
            image_processor: DINO image processor
        """
        self.dataset = dataset
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get image
        image = item['jpg']
        
        # Handle different image formats
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image)).convert('RGB')
        
        # Process image
        pixel_values = self.image_processor(
            images=image,
            return_tensors="pt",
        )["pixel_values"].squeeze(0)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


def collate_fn(batch):
    """Custom collate function."""
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


# ============================================================================
# MAIN ASYNC PROCESSING LOOP
# ============================================================================

async def process_and_save_async(
    dataloader,
    dino_sampler,
    text_embedder,
    saver: AsyncBatchSaver,
    device: str,
):
    """
    Main async processing loop. Processes batches and saves asynchronously.
    """
    sample_idx = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Move to GPU
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            current_batch_size = pixel_values.shape[0]
            
            # ==================== Text Embeddings ====================
            text_embeds = text_embedder(input_ids, attention_mask)
            
            # ==================== DINO Features ====================
            all_hidden_states, position_embeddings = dino_sampler.forward_features(pixel_values)
            clean_inter = all_hidden_states[DINO_BREAK_AT_LAYER]
            clean_final = dino_sampler(clean_inter, position_embeddings)
            
            # ==================== Prepare for async save ====================
            batch_data = {
                "text_embeds": text_embeds.cpu().numpy(),
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
                "clean_inter": clean_inter.cpu().numpy(),
                "clean_final": clean_final.cpu().numpy(),
            }
            
            # Schedule async save (non-blocking)
            await saver.save_batch(sample_idx, batch_data)
            
            sample_idx += current_batch_size
            
            # Clear GPU cache periodically
            if batch_idx % 50 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()
    
    return sample_idx


def main():
    parser = argparse.ArgumentParser(description="Async preprocessing pipeline")
    parser.add_argument("--dataset", type=str, default="BLIP3o/BLIP3o-Pretrain-Long-Caption")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=8)
    parser.add_argument("--tokenize_num_proc", type=int, default=8,
                        help="Number of processes for dataset.map() tokenization")
    parser.add_argument("--save_workers", type=int, default=4,
                        help="Number of threads for async saving")
    parser.add_argument("--text_cache_dir", type=str, default="cache/cache_text")
    parser.add_argument("--dino_cache_dir", type=str, default="cache/cache_dino")
    parser.add_argument("--save_tokenized", type=str, default=None,
                        help="Path to save pretokenized dataset for reuse")
    parser.add_argument("--load_tokenized", type=str, default=None,
                        help="Path to load pretokenized dataset")
    
    args = parser.parse_args()
    
    # Create cache directories
    cache_dirs = {
        "text_embeds": os.path.join(args.text_cache_dir, "text_embeds"),
        "text_inputs": os.path.join(args.text_cache_dir, "text_inputs"),
        "clean_inter": os.path.join(args.dino_cache_dir, "clean_inter"),
        "clean_final": os.path.join(args.dino_cache_dir, "clean_final"),
    }
    
    for dir_path in cache_dirs.values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Async Preprocessing Pipeline")
    print(f"{'='*60}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Tokenization processes: {args.tokenize_num_proc}")
    print(f"Async save workers: {args.save_workers}")
    print(f"{'='*60}\n")
    
    # ==================== Load and tokenize dataset ====================
    if args.load_tokenized and os.path.exists(args.load_tokenized):
        print(f"Loading pretokenized dataset from {args.load_tokenized}...")
        dataset = load_from_disk(args.load_tokenized)
    else:
        print(f"Loading dataset: {args.dataset}...")
        dataset = load_dataset(
            args.dataset, 
            split=args.dataset_split,
            data_files=[f"sa_00000{x}.tar" for x in range(10)]
        )
        
        # Pretokenize using dataset.map()
        dataset = pretokenize_dataset(dataset, num_proc=args.tokenize_num_proc)
        
        # Optionally save tokenized dataset
        if args.save_tokenized:
            print(f"Saving pretokenized dataset to {args.save_tokenized}...")
            dataset.save_to_disk(args.save_tokenized)
    
    print(f"Dataset ready: {len(dataset)} samples\n")
    
    # ==================== Initialize models ====================
    print("Initializing models...")
    
    image_processor = AutoImageProcessor.from_pretrained(DINO_PRETRAINED)
    
    dino_sampler = (
        DinoSampler.from_pretrained(DINO_PRETRAINED, break_at_layer=DINO_BREAK_AT_LAYER)
        .eval()
        .to(DEVICE)
    )
    
    text_embedder = T5TextEmbedder(pretrained_path=TEXT_PRETRAINED).eval().to(DEVICE)

    dino_sampler = torch.compile(dino_sampler, fullgraph=True, dynamic=False)
    text_embedder = torch.compile(text_embedder, fullgraph=True, dynamic=False)
    
    print("Models initialized!\n")
    
    # ==================== Create DataLoader ====================
    preprocess_dataset = PretokenizedImageDataset(dataset, image_processor)
    
    dataloader = DataLoader(
        preprocess_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    
    # ==================== Initialize async saver ====================
    saver = AsyncBatchSaver(cache_dirs, max_workers=args.save_workers)
    
    # ==================== Run async processing ====================
    print("Starting async processing...")
    
    total_samples = asyncio.run(
        process_and_save_async(dataloader, dino_sampler, text_embedder, saver, DEVICE)
    )
    
    # Wait for all saves to complete
    asyncio.run(saver.wait_all())
    saver.shutdown()
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total samples: {total_samples}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()