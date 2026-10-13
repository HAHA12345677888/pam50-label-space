#!/usr/bin/env python3
"""
Pretrain per-modality DenoisingAutoencoders on reconstruction loss only.

Usage:
    python scripts/pretrain_autoencoders.py \
        --data data/train.npz \
        --ae_latent_dims 64,256,256,256,128 \
        --epochs 100 \
        --out checkpoints/pretrained_ae.pt

Output: a single .pt file with state_dicts under keys 'ae_0'...'ae_{M-1}'.
"""
import sys, os, argparse
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'code'))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn

from models.autoencoder_module import DenoisingAutoencoder
from train import MultimodalNPZDataset


def pretrain_one_ae(model: nn.Module, loader: DataLoader, epochs: int, lr: float,
                    device: torch.device, noise_factor: float = 0.1,
                    name: str = "ae", log_every: int = 10):
    """Train a single autoencoder on reconstruction loss only."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            inputs = batch[0].to(device).float()
            # Inject noise explicitly at input level
            noisy = inputs + torch.randn_like(inputs) * noise_factor
            reconstructed, z = model(noisy, add_noise=False)
            loss = nn.functional.mse_loss(reconstructed, inputs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % log_every == 0 or epoch == 1:
            print(f"  [{name}] epoch {epoch:03d}/{epochs} recon_loss: {epoch_loss / n_batches:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/train.npz")
    parser.add_argument("--ae_latent_dims", type=str, default="64,256,256,256,128",
                        help="Comma-separated latent dims, one per modality")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise_factor", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="checkpoints/pretrained_ae.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    ds = MultimodalNPZDataset(args.data)
    latent_dims = [int(x) for x in args.ae_latent_dims.split(",")]
    assert len(latent_dims) == len(ds.modalities), \
        f"Expected {len(ds.modalities)} latent dims, got {len(latent_dims)}"

    print(f"[INFO] Modalities: {[m.shape[1] for m in ds.modalities]}")
    print(f"[INFO] Latent dims: {latent_dims}")

    # Per-modality dataloaders
    loaders = []
    for i, m in enumerate(ds.modalities):
        tds = TensorDataset(torch.from_numpy(m))
        loaders.append(DataLoader(tds, batch_size=args.batch_size, shuffle=True, num_workers=0))

    # Train each AE independently
    state_dicts = {}
    for i, (input_dim, ld, loader) in enumerate(zip(
            [m.shape[1] for m in ds.modalities], latent_dims, loaders)):
        print(f"\n[INFO] Pretraining AE {i}: {input_dim} -> {ld}")
        ae = DenoisingAutoencoder(input_dim=input_dim, latent_dim=ld, noise_factor=args.noise_factor)
        pretrain_one_ae(ae, loader, epochs=args.epochs, lr=args.lr, device=device,
                        noise_factor=args.noise_factor, name=f"ae_{i}", log_every=10)
        state_dicts[f"ae_{i}"] = ae.state_dict()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(state_dicts, args.out)
    print(f"\n[INFO] Pretrained autoencoders saved to {args.out}")


if __name__ == "__main__":
    main()
