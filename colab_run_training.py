#!/usr/bin/env python3
"""
Wrapper script to run training on Colab GPU via colab-cli.
This script will be executed on a fresh Colab VM.
"""
import subprocess
import sys
import os

def run(cmd, desc=""):
    print(f"\n{'='*60}")
    if desc:
        print(f"STEP: {desc}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: {cmd} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    return result

def main():
    # Install dependencies
    run("pip install torch numpy pandas scikit-learn lifelines matplotlib seaborn -q",
        "Installing dependencies")
    
    # Clone repository
    run("cd /content && git clone https://github.com/HAHA1122344/breast-cancer-multimodal-transformer.git",
        "Cloning repository")
    
    # Start training
    os.chdir("/content/breast-cancer-multimodal-transformer/code")
    
    run("python train.py "
        "--train_npz ../data/train.npz "
        "--val_npz ../data/val.npz "
        "--output_dir ../checkpoints "
        "--epochs 150 "
        "--batch_size 128 "
        "--lr 1e-4 "
        "--gradient_accumulation_steps 2 "
        "--pathway_mask_path ../data/soft_module_mask.npz",
        "Training model")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
