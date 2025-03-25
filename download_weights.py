#!/usr/bin/env python3
"""
Download pre-trained weights for TaskSolvingLLM model.
This script downloads model weights from the repository and prepares them for use.
"""

import argparse
import os
import sys
import time
import json
import hashlib
import requests
from tqdm import tqdm
import torch
import shutil
from pathlib import Path

# Configuration
DEFAULT_WEIGHTS_DIR = "weights"
MODELS_CONFIG = {
    "small": {
        "dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "url": "https://example.com/tasksolvingllm/weights/small.pt",
        "size_mb": 500,
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    },
    "medium": {
        "dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "url": "https://example.com/tasksolvingllm/weights/medium.pt",
        "size_mb": 1500,
        "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
    },
    "large": {
        "dim": 2048,
        "num_layers": 36,
        "num_heads": 32,
        "url": "https://example.com/tasksolvingllm/weights/large.pt",
        "size_mb": 5000,
        "sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
    },
}

# URLs for simulated files
MEMORY_TEMPLATE_URL = "https://example.com/tasksolvingllm/memory_template.json"
CONFIG_URL = "https://example.com/tasksolvingllm/config.json"


def calculate_file_hash(filepath, algorithm='sha256'):
    """Calculate hash of a file."""
    hash_func = getattr(hashlib, algorithm)()
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
            
    return hash_func.hexdigest()


def download_file(url, destination, expected_size_mb=None, timeout=30, verify_ssl=True):
    """
    Download a file with progress tracking.
    
    Args:
        url: URL to download
        destination: Path to save the file
        expected_size_mb: Expected size in MB (for progress bar)
        timeout: Connection timeout in seconds
        verify_ssl: Whether to verify SSL certificates
    
    Returns:
        bool: True if download was successful
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Start the request
        with requests.get(url, stream=True, timeout=timeout, verify=verify_ssl) as response:
            response.raise_for_status()
            
            # Get file size from headers or use expected size
            file_size = int(response.headers.get('content-length', 0))
            if file_size == 0 and expected_size_mb:
                file_size = expected_size_mb * 1024 * 1024
                
            # Convert to MB for display
            file_size_mb = file_size / (1024 * 1024)
            
            # Display progress bar
            desc = f"Downloading {os.path.basename(destination)}"
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=desc) as pbar:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"Downloaded {file_size_mb:.1f}MB to {destination}")
            return True
            
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        
        # Remove partial downloads
        if os.path.exists(destination):
            os.remove(destination)
            
        return False


def simulate_download_file(url, destination, expected_size_mb, timeout=30):
    """
    Simulate file download for demonstration purposes.
    In a real implementation, you would replace this with actual download_file.
    """
    print(f"Simulating download from {url}")
    print(f"This would download approximately {expected_size_mb}MB")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Display progress bar
    desc = f"Downloading {os.path.basename(destination)}"
    total_size = expected_size_mb * 1024 * 1024  # Convert MB to bytes
    chunk_size = total_size // 50  # 50 updates
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
        # Create a dummy file with the correct size
        with open(destination, 'wb') as f:
            # Simulate download chunks
            downloaded = 0
            while downloaded < total_size:
                # Simulate network delay
                time.sleep(0.1)
                
                # Calculate chunk size (last chunk may be smaller)
                current_chunk_size = min(chunk_size, total_size - downloaded)
                
                # Write zeros (in a real download this would be actual data)
                f.write(b'\0' * current_chunk_size)
                
                # Update progress bar
                pbar.update(current_chunk_size)
                downloaded += current_chunk_size
    
    print(f"Simulated download of {expected_size_mb:.1f}MB to {destination}")
    
    # For simulation, create a file with metadata so we can verify it later
    with open(f"{destination}.meta", 'w') as f:
        json.dump({
            "url": url,
            "size_mb": expected_size_mb,
            "simulated": True,
            "timestamp": time.time()
        }, f)
    
    return True


def verify_model_file(filepath, expected_hash=None):
    """
    Verify the integrity of a downloaded model file.
    
    Args:
        filepath: Path to the file to verify
        expected_hash: Expected SHA-256 hash
    
    Returns:
        bool: True if verification passed
    """
    # For simulated files, check if metadata exists
    if os.path.exists(f"{filepath}.meta"):
        print(f"Simulated file detected: {filepath}")
        print("Verification skipped for simulated file")
        return True
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return False
    
    if expected_hash:
        print(f"Verifying file integrity: {os.path.basename(filepath)}")
        file_hash = calculate_file_hash(filepath)
        
        if file_hash != expected_hash:
            print(f"Error: Hash verification failed for {filepath}")
            print(f"Expected: {expected_hash}")
            print(f"Actual: {file_hash}")
            return False
            
        print("File integrity verified successfully")
    
    return True


def convert_to_sharded_weights(model_path, output_dir, shard_size_mb=1000):
    """
    Convert a single model file into sharded weights for larger models.
    
    Args:
        model_path: Path to the model file
        output_dir: Directory to save sharded weights
        shard_size_mb: Size of each shard in MB
    """
    try:
        print(f"Processing model file: {model_path}")
        
        # For simulated files, create dummy shards
        if os.path.exists(f"{model_path}.meta"):
            print("Processing simulated model file")
            
            # Read metadata
            with open(f"{model_path}.meta", 'r') as f:
                metadata = json.load(f)
            
            # Calculate number of shards
            model_size_mb = metadata.get("size_mb", 500)
            num_shards = max(1, int(model_size_mb / shard_size_mb) + 1)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Create dummy shard files
            for i in range(num_shards):
                shard_path = os.path.join(output_dir, f"model_shard_{i:03d}.pt")
                
                # Last shard might be smaller
                if i == num_shards - 1:
                    current_shard_size = model_size_mb % shard_size_mb
                    if current_shard_size == 0:
                        current_shard_size = shard_size_mb
                else:
                    current_shard_size = shard_size_mb
                
                # Create a dummy shard file
                with open(shard_path, 'wb') as f:
                    f.write(b'\0' * int(current_shard_size * 1024 * 1024 / num_shards))
                
                print(f"Created shard {i+1}/{num_shards}: {shard_path}")
                
            # Create metadata file
            with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
                json.dump({
                    "version": "1.0",
                    "num_shards": num_shards,
                    "original_size_mb": model_size_mb,
                    "shards": [f"model_shard_{i:03d}.pt" for i in range(num_shards)],
                    "simulated": True,
                    "timestamp": time.time()
                }, f)
                
            print(f"Created metadata file: {os.path.join(output_dir, 'model_info.json')}")
            return True
        
        # For real files, load the model and split into shards
        print("Loading model weights...")
        model_weights = torch.load(model_path, map_location="cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Split into state dict into shards
        shards = []
        current_shard = {}
        current_shard_size = 0
        shard_index = 0
        
        # Process each parameter
        for key, tensor in model_weights.items():
            # Calculate tensor size in MB
            tensor_size_mb = tensor.element_size() * tensor.nelement() / (1024 * 1024)
            
            # If this tensor would exceed shard size, save current shard and start a new one
            if current_shard_size + tensor_size_mb > shard_size_mb and current_shard:
                # Save current shard
                shard_path = os.path.join(output_dir, f"model_shard_{shard_index:03d}.pt")
                torch.save(current_shard, shard_path)
                print(f"Saved shard {shard_index}: {shard_path} ({current_shard_size:.2f}MB)")
                
                # Add to list of shards
                shards.append(f"model_shard_{shard_index:03d}.pt")
                
                # Reset for next shard
                current_shard = {}
                current_shard_size = 0
                shard_index += 1
            
            # Add tensor to current shard
            current_shard[key] = tensor
            current_shard_size += tensor_size_mb
        
        # Save final shard if not empty
        if current_shard:
            shard_path = os.path.join(output_dir, f"model_shard_{shard_index:03d}.pt")
            torch.save(current_shard, shard_path)
            print(f"Saved shard {shard_index}: {shard_path} ({current_shard_size:.2f}MB)")
            shards.append(f"model_shard_{shard_index:03d}.pt")
        
        # Create metadata file
        with open(os.path.join(output_dir, "model_info.json"), 'w') as f:
            json.dump({
                "version": "1.0",
                "num_shards": len(shards),
                "original_model": os.path.basename(model_path),
                "shards": shards,
                "timestamp": time.time()
            }, f)
            
        print(f"Created metadata file: {os.path.join(output_dir, 'model_info.json')}")
        return True
        
    except Exception as e:
        print(f"Error converting model to sharded weights: {e}")
        return False


def create_config_file(model_size, output_dir, model_path):
    """
    Create configuration file for the model.
    
    Args:
        model_size: Size of the model (small, medium, large)
        output_dir: Directory to save config file
        model_path: Path to model weights file or directory
    """
    config = MODELS_CONFIG.get(model_size, MODELS_CONFIG["small"]).copy()
    
    # Remove download-specific keys
    for key in ["url", "size_mb", "sha256"]:
        config.pop(key, None)
    
    # Add model information
    config.update({
        "model_name": f"TaskSolvingLLM-{model_size}",
        "vocab_size": 50257,  # GPT-2 vocabulary size
        "max_seq_len": 8192,
        "use_quantization": True,
        "model_path": os.path.relpath(model_path, output_dir),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    # Save config file
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"Created configuration file: {config_path}")
    return config_path


def setup_memory_template(output_dir):
    """
    Download or create memory template file.
    
    Args:
        output_dir: Directory to save memory template
    """
    memory_dir = os.path.join(output_dir, "memory")
    os.makedirs(memory_dir, exist_ok=True)
    
    # Create empty memory template
    memory_path = os.path.join(memory_dir, "memory_template.json")
    with open(memory_path, 'w') as f:
        json.dump({
            "version": "1.0",
            "memories": [],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
        
    print(f"Created memory template file: {memory_path}")
    return memory_path


def download_model(model_size="small", output_dir=DEFAULT_WEIGHTS_DIR, force=False, simulate=True):
    """
    Download model weights and configuration.
    
    Args:
        model_size: Size of model to download (small, medium, large)
        output_dir: Directory to save model files
        force: Whether to force re-download even if files exist
        simulate: Whether to simulate the download (for demo/testing)
    
    Returns:
        bool: True if download was successful
    """
    if model_size not in MODELS_CONFIG:
        print(f"Error: Invalid model size '{model_size}'. Available options: {', '.join(MODELS_CONFIG.keys())}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model configuration
    model_config = MODELS_CONFIG[model_size]
    
    # Download model weights
    model_filename = f"tasksolvingllm_{model_size}.pt"
    model_path = os.path.join(output_dir, model_filename)
    
    if os.path.exists(model_path) and not force:
        print(f"Model file already exists: {model_path}")
        print("Use --force to re-download")
    else:
        if simulate:
            success = simulate_download_file(
                model_config["url"],
                model_path,
                model_config["size_mb"]
            )
        else:
            success = download_file(
                model_config["url"],
                model_path,
                model_config["size_mb"]
            )
            
        if not success:
            print("Failed to download model weights")
            return False
    
    # Verify model file
    if not simulate and not verify_model_file(model_path, model_config["sha256"]):
        print("Model verification failed")
        return False
    
    # Process model weights based on size
    if model_size in ["medium", "large"]:
        print(f"Converting {model_size} model to sharded weights...")
        sharded_dir = os.path.join(output_dir, f"{model_size}_shards")
        if not convert_to_sharded_weights(model_path, sharded_dir):
            print("Failed to convert model to sharded weights")
            return False
        model_location = sharded_dir
    else:
        model_location = model_path
    
    # Create configuration file
    config_path = create_config_file(model_size, output_dir, model_location)
    
    # Setup memory template
    memory_path = setup_memory_template(output_dir)
    
    print("\nModel download and setup complete!")
    print(f"Model weights: {model_location}")
    print(f"Configuration: {config_path}")
    print(f"Memory template: {memory_path}")
    print("\nTo use this model, initialize TaskSolvingLLM with the configuration file:")
    print(f"model = TaskSolvingLLM.from_config('{os.path.relpath(config_path)}')")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained weights for TaskSolvingLLM")
    parser.add_argument("--model", "-m", choices=list(MODELS_CONFIG.keys()), default="small",
                       help="Model size to download (default: small)")
    parser.add_argument("--output", "-o", default=DEFAULT_WEIGHTS_DIR,
                       help=f"Output directory (default: {DEFAULT_WEIGHTS_DIR})")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force re-download even if files exist")
    parser.add_argument("--no-simulate", action="store_true",
                       help="Perform actual download instead of simulation (requires internet connection)")
    
    args = parser.parse_args()
    
    print(f"Downloading TaskSolvingLLM ({args.model} model)")
    print(f"Output directory: {args.output}")
    
    if args.no_simulate:
        print("Performing actual download...")
    else:
        print("Simulating download (use --no-simulate to perform actual download)")
    
    if download_model(args.model, args.output, args.force, not args.no_simulate):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())