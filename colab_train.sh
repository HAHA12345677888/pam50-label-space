#!/bin/bash
set -e

echo "=========================================="
echo "Installing dependencies..."
echo "=========================================="
pip install torch numpy pandas scikit-learn lifelines matplotlib seaborn -q

echo "=========================================="
echo "Cloning repository..."
echo "=========================================="
cd /content
git clone https://github.com/HAHA1122344/breast-cancer-multimodal-transformer.git
cd breast-cancer-multimodal-transformer

echo "=========================================="
echo "Starting training..."
echo "=========================================="
cd code
python train.py \
  --train_npz ../data/train.npz \
  --val_npz ../data/val.npz \
  --output_dir ../checkpoints \
  --epochs 150 \
  --batch_size 128 \
  --lr 1e-4 \
  --gradient_accumulation_steps 2 \
  --pathway_mask_path ../data/soft_module_mask.npz

echo "=========================================="
echo "Training complete!"
echo "=========================================="
