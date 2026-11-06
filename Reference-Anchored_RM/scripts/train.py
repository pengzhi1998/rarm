import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import torch
from torch.utils.data import DataLoader
from video_clip_comp import VideoComparator
from utils.loss import SoftNearestNeighborsLoss
from utils.dataset import ContrastiveVideoDataset, collate_contrastive_batch
from tqdm import tqdm
import os
import json
from datetime import datetime
import wandb
from models.vit_dinov3 import DINO_REPOSITORY


def train_video_comparator(
    vit_checkpoint_path,
    data_dir,
    output_dir="checkpoints",
    num_epochs=200,
    batch_size=64,
    learning_rate=1e-4,
    num_frames=5,
    temperature=0.1,
    device="cuda",
    wandb_project="rarm-training",
    num_workers=12,
    wandb_mode="disabled",
):
    if num_frames != 5:
        raise ValueError("The original training sampler requires num_frames=5.")
    if num_epochs < 1 or batch_size < 1 or num_workers < 0:
        raise ValueError("epochs and batch_size must be positive; num_workers must be nonnegative.")
    if learning_rate <= 0 or temperature <= 0:
        raise ValueError("learning_rate and temperature must be positive.")
    if not Path(vit_checkpoint_path).is_file():
        raise FileNotFoundError(f"DINOv3 checkpoint not found: {vit_checkpoint_path}")
    if not (Path(data_dir) / "JPEGImages").is_dir():
        raise FileNotFoundError(f"Dataset JPEGImages directory not found: {data_dir}")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "vit_checkpoint_path": vit_checkpoint_path,
        "data_dir": data_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_frames": num_frames,
        "temperature": temperature,
        "device": device,
        "lora_r": 16,
        "embed_dim": 768,
        "hidden_dim": 512,
        "num_self_attn_layers": 2,
        "weight_decay": 0.01,
        "num_workers": num_workers,
        "dinov3_repository": DINO_REPOSITORY,
        "dinov3_local_override": os.environ.get("RARM_DINOV3_REPO"),
    }

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    wandb.init(project=wandb_project, name=run_name, config=config, mode=wandb_mode)

    model = VideoComparator(
        vit_checkpoint_path=vit_checkpoint_path,
        lora_r=16,
        embed_dim=768,
        hidden_dim=512,
        num_self_attn_layers=2,
        device=device
    )

    criterion = SoftNearestNeighborsLoss(temperature=temperature)

    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    dataset = ContrastiveVideoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        image_size=384
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_contrastive_batch
    )

    losses_log = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            # batch structure:
            # - anchors: [B, T, H, W, 3]
            # - positives: List of B lists, each with variable number of [T, H, W, 3]
            # - first_source_negatives: List of B lists
            # - second_source_negatives: List of B lists

            anchors = batch['anchors'].to(device)  # [B, T, H, W, 3]
            positives = batch['positives']  # List of lists
            first_negs = batch['first_source_negatives']
            second_negs = batch['second_source_negatives']

            B = anchors.size(0)
            T = anchors.size(1)

            # ─────────────────────────────────────────
            # Step 1: Encode all anchors
            # ─────────────────────────────────────────
            anchor_frames = anchors.reshape(-1, *anchors.shape[2:])  # [B*T, H, W, 3]
            anchor_embs_flat = model.vit.encode(anchor_frames)  # [B*T*576, D]

            H, W = anchor_frames.shape[1], anchor_frames.shape[2]
            patch_size = model.vit.vit.patch_size
            orig_patches = (H // patch_size) * (W // patch_size)
            D = anchor_embs_flat.shape[1]

            anchor_embs = anchor_embs_flat.reshape(B, T * orig_patches, D)
            anchor_embs = model.temporal_processor(anchor_embs)  # [B, T*64, D]

            # ─────────────────────────────────────────
            # Step 2: Encode all positives and negatives
            # ─────────────────────────────────────────
            def encode_clip_list(clip_list):
                """Encode a list of clips, returns list of embeddings."""
                if not clip_list:
                    return []
                clips_stacked = torch.stack(clip_list).to(device)  # [N, T, H, W, 3]
                N = clips_stacked.size(0)
                frames = clips_stacked.reshape(-1, *clips_stacked.shape[2:])
                embs_flat = model.vit.encode(frames)
                embs = embs_flat.reshape(N, T * orig_patches, D)
                embs = model.temporal_processor(embs)
                return [embs[i] for i in range(N)]

            # Encode all clips for each video
            pos_embs_list = [encode_clip_list(p) for p in positives]
            first_neg_embs_list = [encode_clip_list(n) for n in first_negs]
            second_neg_embs_list = [encode_clip_list(n) for n in second_negs]

            # ─────────────────────────────────────────
            # Step 3: Compute similarities for SNNL
            # ─────────────────────────────────────────
            pos_similarities = []
            first_neg_similarities = []
            second_neg_similarities = []

            N_tokens = anchor_embs.size(1)

            for i in range(B):
                anchor_i = anchor_embs[i:i+1]  # [1, N_tokens, D]

                # Positive similarities (always 2-5 positives)
                pos_stacked = torch.stack(pos_embs_list[i])  # [num_pos, N_tokens, D]
                anchor_exp = anchor_i.expand(pos_stacked.size(0), -1, -1)
                cross_out = model.cross_attn(anchor_exp, pos_stacked)
                pos_sims = model.comparison_head(cross_out).squeeze(-1)
                pos_similarities.append(pos_sims)

                # First-source negative similarities (could be empty for edge cases)
                if first_neg_embs_list[i]:
                    neg1_stacked = torch.stack(first_neg_embs_list[i])
                    anchor_exp = anchor_i.expand(neg1_stacked.size(0), -1, -1)
                    cross_out = model.cross_attn(anchor_exp, neg1_stacked)
                    neg1_sims = model.comparison_head(cross_out).squeeze(-1)
                    first_neg_similarities.append(neg1_sims)
                else:
                    first_neg_similarities.append(torch.tensor([], device=device))

                # Second-source negative similarities (always 2-5 reverse clips)
                neg2_stacked = torch.stack(second_neg_embs_list[i])
                anchor_exp = anchor_i.expand(neg2_stacked.size(0), -1, -1)
                cross_out = model.cross_attn(anchor_exp, neg2_stacked)
                neg2_sims = model.comparison_head(cross_out).squeeze(-1)
                second_neg_similarities.append(neg2_sims)

            # Third-source: anchor vs other anchors (cross-video)
            # Compute [B, B] similarity matrix, extract off-diagonal for each anchor
            anchor_Q = anchor_embs.unsqueeze(1).expand(B, B, N_tokens, D)
            anchor_KV = anchor_embs.unsqueeze(0).expand(B, B, N_tokens, D)
            Q_flat = anchor_Q.reshape(B * B, N_tokens, D)
            KV_flat = anchor_KV.reshape(B * B, N_tokens, D)

            cross_out = model.cross_attn(Q_flat, KV_flat)
            sims_flat = model.comparison_head(cross_out).squeeze(-1)
            sim_matrix = sims_flat.view(B, B)

            # Extract off-diagonal (other anchors as negatives)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            third_neg_similarities = sim_matrix[mask].view(B, B - 1)

            # ─────────────────────────────────────────
            # Step 4: Compute SNNL loss
            # ─────────────────────────────────────────
            loss = criterion(
                pos_similarities,
                first_neg_similarities,
                second_neg_similarities,
                third_neg_similarities
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        losses_log.append({"epoch": epoch + 1, "avg_loss": avg_loss})
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss})

        with open(os.path.join(run_dir, "losses.json"), "w") as f:
            json.dump(losses_log, f, indent=2)

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    wandb.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the Reference-Anchored Reward Model.")
    parser.add_argument("--data_dir", default="datasets/dataset_1500/ReferenceAnchoredRewardModelDataset",
                        help="Dataset root containing JPEGImages/<video_id>/.")
    parser.add_argument("--vit_checkpoint_path",
                        default="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--output_dir", default="checkpoints/dinov3_comparator")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_frames", type=int, choices=[5], default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--wandb_project", default="rarm-training")
    parser.add_argument("--wandb_mode", choices=["disabled", "offline", "online"], default="disabled")
    train_video_comparator(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
