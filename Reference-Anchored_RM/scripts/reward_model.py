import sys
from pathlib import Path

# Add parent directory to path so we can import from models, utils, etc.
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Add scripts directory to path for local imports
scripts_dir = Path(__file__).resolve().parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:
    import cv2
except ImportError as exc:
    raise ImportError("opencv-python is required for video loading.") from exc

from video_clip_comp import VideoComparator


class ClipRewardModel:
    """
    Reward model that compares rollout clips to demonstration clips and
    uses the highest similarity as the reward signal.
    """

    def __init__(
        self,
        vit_checkpoint_path,
        comparator_checkpoint_path,
        demo_path,
        num_frames=5,
        frame_stride=1,
        clip_stride=None,
        device="cuda"
    ):
        if num_frames < 1 or frame_stride < 1 or (clip_stride is not None and clip_stride < 1):
            raise ValueError("num_frames, frame_stride, and clip_stride must be positive.")
        if not Path(comparator_checkpoint_path).is_file():
            raise FileNotFoundError(f"RARM checkpoint not found: {comparator_checkpoint_path}")
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.clip_span = (num_frames - 1) * frame_stride + 1 + 2
        self.clip_stride = clip_stride or self.clip_span
        self.device = device
        self.rollout_buffer = []

        self.model = VideoComparator(
            vit_checkpoint_path=vit_checkpoint_path,
            lora_r=16,
            embed_dim=768,
            hidden_dim=512,
            num_self_attn_layers=2,
            device=device
        )

        checkpoint = torch.load(comparator_checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()

        demo_frames = self._load_frames_from_dir(demo_path)
        demo_clips = self._build_clips(demo_frames)
        if not demo_clips:
            raise ValueError("No demo clips available to compare.")
        self.demo_embs = self._encode_clips(demo_clips, squeeze=False)

    def _select_clip_indices(self, start_index):
        if self.num_frames == 1:
            return [start_index]

        step = (self.clip_span - 1) / (self.num_frames - 1)
        return [start_index + int(round(i * step)) for i in range(self.num_frames)]

    def _load_frames_from_dir(self, dir_path):
        frame_files = sorted(
            list(Path(dir_path).glob("*.jpg")) + list(Path(dir_path).glob("*.png"))
        )
        if not frame_files:
            raise ValueError(f"No .jpg or .png frames found in {dir_path}")

        frames = []
        for frame_path in frame_files:
            frame = Image.open(frame_path).convert("RGB")
            frame = TF.resize(frame, size=[384, 384], antialias=True)
            frame_tensor = TF.to_tensor(frame).permute(1, 2, 0)
            frames.append(frame_tensor)

        return torch.stack(frames)

    def _build_clips(self, frames):
        total_frames = frames.size(0)
        max_start = total_frames - self.clip_span + 1
        if max_start <= 0:
            return []

        clips = []
        for start in range(0, max_start, self.clip_stride):
            indices = self._select_clip_indices(start)
            if indices[-1] >= total_frames:
                break
            clip = frames[indices]
            clips.append(clip)

        return clips

    @torch.inference_mode()
    def _encode_clips(self, clips, *, squeeze=False):
        if isinstance(clips, list):
            clips_tensor = torch.stack(clips)
        else:
            clips_tensor = clips

        if clips_tensor.dim() == 4:
            clips_tensor = clips_tensor.unsqueeze(0)
        elif clips_tensor.dim() != 5:
            raise ValueError("Clips must be a [N, T, H, W, C] tensor.")

        N, T, height, width, channels = clips_tensor.shape

        patch_size = self.model.vit.vit.patch_size
        orig_patches = (height // patch_size) * (width // patch_size)
        expected_patches = self.model.temporal_processor.orig_num_patches
        if orig_patches != expected_patches:
            raise ValueError(
                f"Expected {expected_patches} patches per frame, got {orig_patches}. "
                "Resize frames to match training resolution."
            )

        clips_tensor = clips_tensor.to(self.device)
        frames = clips_tensor.reshape(-1, height, width, channels)

        embeddings = self.model.vit.encode(frames)
        expected_tokens = T * orig_patches
        if embeddings.size(0) != N * expected_tokens:
            raise ValueError("Encoded token count does not match expected clip size.")

        embeddings = embeddings.reshape(N, expected_tokens, embeddings.size(-1))
        embeddings = self.model.temporal_processor(embeddings)
        if squeeze and embeddings.size(0) == 1:
            return embeddings.squeeze(0)
        return embeddings

    def _encode_clip(self, clip):
        return self._encode_clips(clip, squeeze=True)

    def _prepare_frame(self, frame):
        if isinstance(frame, Image.Image):
            frame_pil = frame.convert("RGB")
        elif isinstance(frame, torch.Tensor):
            frame_tensor = frame
            if frame_tensor.dim() == 3 and frame_tensor.shape[0] in (1, 3):
                frame_tensor = frame_tensor.permute(1, 2, 0)
            if frame_tensor.shape[-1] != 3:
                raise ValueError("rollout_frame tensor must have 3 channels.")
            if frame_tensor.shape[0] != 384 or frame_tensor.shape[1] != 384:
                frame_tensor = TF.resize(frame_tensor.permute(2, 0, 1), size=[384, 384], antialias=True)
                frame_tensor = frame_tensor.permute(1, 2, 0)
            return frame_tensor
        else:
            try:
                import numpy as np
            except ImportError as exc:
                raise TypeError("rollout_frame must be a PIL.Image or torch.Tensor.") from exc

            if isinstance(frame, np.ndarray):
                if frame.ndim != 3 or frame.shape[-1] != 3:
                    raise ValueError("rollout_frame numpy array must be HxWx3.")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
            else:
                raise TypeError("rollout_frame must be a PIL.Image, torch.Tensor, or numpy array.")

        frame_pil = TF.resize(frame_pil, size=[384, 384], antialias=True)
        return TF.to_tensor(frame_pil).permute(1, 2, 0)

    @torch.inference_mode()
    def step(self, rollout_frame):
        frame_tensor = self._prepare_frame(rollout_frame)
        self.rollout_buffer.append(frame_tensor)

        required = self.clip_span
        if len(self.rollout_buffer) < required:
            return -3.0

        start_idx = len(self.rollout_buffer) - required
        indices = self._select_clip_indices(start_idx)
        clip = torch.stack([self.rollout_buffer[i] for i in indices])

        rollout_emb = self._encode_clip(clip)
        demo_batch = self.demo_embs
        rollout_batch = rollout_emb.unsqueeze(0).expand(demo_batch.size(0), -1, -1)
        emb_a_vs_b = self.model.cross_attn(rollout_batch, demo_batch)
        sims = self.model.comparison_head(emb_a_vs_b)
        return sims.max().item()

    def reset_rollout(self):
        self.rollout_buffer = []


@torch.inference_mode()
def main():
    import argparse
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to render the clip-to-clip heatmap. "
            "Install it with: pip install matplotlib"
        ) from exc

    parser = argparse.ArgumentParser(
        description="Test reward model by comparing rollout frames against demo frames."
    )
    parser.add_argument(
        "--demo_dir",
        required=True,
        help="Directory with demo frames as .jpg or .png files."
    )
    parser.add_argument(
        "--rollout_dir",
        required=True,
        help="Directory with rollout frames as .jpg or .png files."
    )
    parser.add_argument(
        "--vit_checkpoint_path",
        default="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        help="Path to the DINOv3 ViT checkpoint."
    )
    parser.add_argument(
        "--comparator_checkpoint_path",
        default="checkpoints/checkpoint_epoch_65.pt",
        help="Path to the trained comparator checkpoint."
    )
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output_heatmap",
        default="clip_similarity_heatmap.png",
        help="Output path for the clip-to-clip similarity heatmap."
    )

    args = parser.parse_args()

    model = ClipRewardModel(
        vit_checkpoint_path=args.vit_checkpoint_path,
        comparator_checkpoint_path=args.comparator_checkpoint_path,
        demo_path=args.demo_dir,
        num_frames=args.num_frames,
        frame_stride=args.frame_stride,
        clip_stride=args.clip_stride,
        device=args.device
    )

    demo_frames = model._load_frames_from_dir(args.demo_dir)
    rollout_frames = model._load_frames_from_dir(args.rollout_dir)
    rewards = []
    for frame in rollout_frames:
        rewards.append(model.step(frame))

    max_reward = max(rewards) if rewards else 0.0
    print(f"Rollout frames processed: {len(rollout_frames)}")
    print(f"Max reward: {max_reward}")

    # Build clip-to-clip similarity matrix (rollout vs demo).
    rollout_clips = model._build_clips(rollout_frames)
    if not rollout_clips:
        raise ValueError("No rollout clips available to compare.")

    rollout_embs = model._encode_clips(rollout_clips, squeeze=False)
    demo_embs = model.demo_embs

    num_rollout = rollout_embs.size(0)
    num_demo = demo_embs.size(0)
    if num_rollout == 0 or num_demo == 0:
        raise ValueError("Clip embeddings are empty; check clip parameters.")

    import time
    start_time = time.perf_counter()

    rollout_pairs = rollout_embs.unsqueeze(1).expand(num_rollout, num_demo, -1, -1)
    demo_pairs = demo_embs.unsqueeze(0).expand(num_rollout, num_demo, -1, -1)

    q_pairs = rollout_pairs.reshape(-1, rollout_embs.size(1), rollout_embs.size(2))
    kv_pairs = demo_pairs.reshape(-1, demo_embs.size(1), demo_embs.size(2))

    emb_pairs = model.model.cross_attn(q_pairs, kv_pairs)
    sims = model.model.comparison_head(emb_pairs)
    sim_matrix = sims.reshape(num_rollout, num_demo).detach().cpu().float()

    elapsed = time.perf_counter() - start_time
    print(f"Heatmap similarity computation time: {elapsed:.3f}s")

    plt.figure(figsize=(6, 6))
    plt.imshow(sim_matrix, cmap="viridis", aspect="equal")
    plt.colorbar(label="Similarity")
    plt.xlabel("Demo clip index")
    plt.ylabel("Rollout clip index")
    plt.title("Clip-to-Clip Similarity")
    plt.tight_layout()
    output_path = Path(args.output_heatmap)
    if not output_path.suffix:
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved heatmap to {output_path}")


if __name__ == "__main__":
    main()
