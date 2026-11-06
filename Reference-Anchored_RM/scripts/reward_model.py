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
        device="cuda",
        similarity_threshold_offset: float = 0.5,
        use_cross_attn: bool = True,
        use_temporal_processor: bool = True,
    ):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.clip_span = (num_frames - 1) * frame_stride + 1  # Consecutive frames
        self.clip_stride = clip_stride or self.clip_span
        self.device = device
        # Per-clip threshold offset: threshold_i = max_self_sim_i - offset
        # Also used to normalise the reward gap (max_sim - threshold) / offset.
        self.similarity_threshold_offset = similarity_threshold_offset
        # Architecture toggle: if True, similarities = cross_attn → comparison_head (default).
        # If False, similarities = -||rollout - demo||_2 (no-cross-attn ablation).
        self.use_cross_attn = use_cross_attn
        # Architecture toggle: if False, skip the temporal_processor module entirely.
        # Embeddings keep their raw ViT shape [N, T*576, D] when bypassed.
        self.use_temporal_processor = use_temporal_processor
        self.rollout_buffer = []
        self.similarity_rows = []  # Store similarity rows for heatmap
        self.clip_start_indices = []  # Track which frame each clip started at
        self._last_stored_start_idx = -999  # Track last stored clip start index for stride control
        self.last_similarities = None  # Store last computed similarities for external access
        # Clip stride for heatmap storage (same as _build_clips uses)
        self._heatmap_stride = max(1, (num_frames - 1) // 2)

        self.model = VideoComparator(
            vit_checkpoint_path=vit_checkpoint_path,
            lora_r=16,
            embed_dim=768,
            hidden_dim=512,
            num_self_attn_layers=2,
            device=device,
            use_temporal_processor=use_temporal_processor,
        )

        checkpoint = torch.load(comparator_checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()

        demo_frames = self._load_frames_from_dir(demo_path)
        demo_clips, self.demo_clip_start_indices = self._build_clips(demo_frames)
        if not demo_clips:
            raise ValueError("No demo clips available to compare.")
        self.demo_embs = self._encode_clips(demo_clips, squeeze=False)
        # Store demo embeddings in float32 to avoid dtype issues
        self.demo_embs = self.demo_embs.float()

        # Precompute progress label for each demo clip (scale 0–10).
        # Middle frame of clip i = start_i + (clip_span - 1) // 2.
        self.total_ref_frames = demo_frames.size(0)
        mid_offset = (self.clip_span - 1) // 2
        self.demo_progress = [
            (start + mid_offset) / self.total_ref_frames * 10.0
            for start in self.demo_clip_start_indices
        ]

        # Compute per-clip self-similarity thresholds.
        # For each reference clip i, threshold_i = max_self_sim_i - similarity_threshold_offset,
        # where max_self_sim_i is the max similarity of clip i against all ref clips.
        print(f"Computing per-clip self-similarity thresholds (offset={self.similarity_threshold_offset})...")
        self.max_self_sim, _, per_clip_max, per_clip_best = self._compute_threshold()
        self.per_clip_thresholds = [ms - self.similarity_threshold_offset for ms in per_clip_max]
        print(f"  Global max_self_sim={self.max_self_sim:.4f}")
        print(f"  Per-ref-clip thresholds ({len(per_clip_max)} clips):")
        for i, (max_sim, best_ref, thresh) in enumerate(zip(per_clip_max, per_clip_best, self.per_clip_thresholds)):
            mid_frame = self.demo_clip_start_indices[i] + mid_offset
            print(f"    Frame {mid_frame:4d}: max_sim={max_sim:.4f}, best_ref_clip={best_ref}, threshold={thresh:.4f}")

        # Progress state: updated each time a confident match is found.
        self.previous_progress = 0.0
        self.last_raw_delta = 0.0

    def _select_clip_indices(self, start_index):
        if self.num_frames == 1:
            return [start_index]

        step = (self.clip_span - 1) / (self.num_frames - 1)
        return [start_index + int(round(i * step)) for i in range(self.num_frames)]

    @torch.inference_mode()
    def _compute_threshold(self):
        """
        Compare every demo clip against all demo clips to find the single
        maximum self-similarity value.  threshold = max_self_sim - 1.0.
        Returns (max_sim, threshold, per_clip_max, per_clip_best) where
        per_clip_max[i] and per_clip_best[i] are the max similarity value and
        best matching clip index for ref clip i.
        """
        N = self.demo_embs.size(0)
        max_sim = float("-inf")
        per_clip_max = []
        per_clip_best = []
        for i in range(N):
            query_batch = self.demo_embs[i].unsqueeze(0).expand(N, -1, -1)
            # Convert to model dtype for computation
            query_batch = query_batch.to(self.model.dtype)
            demo_batch = self.demo_embs.to(self.model.dtype)
            if self.use_cross_attn:
                emb_a_vs_b = self.model.cross_attn(query_batch, demo_batch)
                sims = self.model.comparison_head(emb_a_vs_b)
            else:
                sims = -torch.norm(query_batch - demo_batch, p=2, dim=-1).mean(dim=-1, keepdim=False)
            sims_float = sims.float()
            row_max = sims_float.max().item()
            per_clip_max.append(row_max)
            per_clip_best.append(sims_float.argmax().item())
            if row_max > max_sim:
                max_sim = row_max
        return max_sim, max_sim - 1.0, per_clip_max, per_clip_best

    def _load_frames_from_dir(self, dir_path):
        # Try .jpg first, then .png
        frame_files = sorted(Path(dir_path).glob("*.jpg"))
        if not frame_files:
            frame_files = sorted(Path(dir_path).glob("*.png"))
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
            return [], []

        loop_stride = self.clip_stride

        clips = []
        start_indices = []
        for start in range(0, max_start, loop_stride):
            indices = self._select_clip_indices(start)
            if indices[-1] >= total_frames:
                break
            clip = frames[indices]
            clips.append(clip)
            start_indices.append(start)

        return clips, start_indices

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
        if self.use_temporal_processor:
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
        if self.use_temporal_processor:
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
    def step(self, rollout_frame, store_similarities=True):
        """
        Process a single frame and return a step reward.

        Args:
            rollout_frame: Single frame (PIL.Image, torch.Tensor, or numpy array)
            store_similarities: If True, store similarity row for heatmap construction
                               (respects the same stride as _build_clips for consistency)

        Returns:
            reward: Similarity-weighted progress delta if the rollout clip's max
                    similarity to the best-matching demo clip >= that clip's threshold;
                    0.0 if below threshold (not confident — progress kept frozen);
                    None if not enough frames have been buffered yet (sentinel).
        """
        frame_tensor = self._prepare_frame(rollout_frame)
        self.rollout_buffer.append(frame_tensor)

        required = self.clip_span
        if len(self.rollout_buffer) < required:
            return None  # Sentinel: clip not yet formed

        start_idx = len(self.rollout_buffer) - required
        indices = self._select_clip_indices(start_idx)
        clip = torch.stack([self.rollout_buffer[i] for i in indices])

        rollout_emb = self._encode_clip(clip)
        demo_batch = self.demo_embs
        rollout_batch = rollout_emb.unsqueeze(0).expand(demo_batch.size(0), -1, -1)

        # Convert to the model's dtype (bfloat16) for computation
        rollout_batch = rollout_batch.to(self.model.dtype)
        demo_batch = demo_batch.to(self.model.dtype)

        if self.use_cross_attn:
            emb_a_vs_b = self.model.cross_attn(rollout_batch, demo_batch)
            sims = self.model.comparison_head(emb_a_vs_b)
        else:
            sims = -torch.norm(rollout_batch - demo_batch, p=2, dim=-1).mean(dim=-1, keepdim=False)

        # Store the current similarities for external access (convert to float32 to avoid BFloat16 issues)
        self.last_similarities = sims.detach().float().cpu().numpy()

        # Store similarity row for heatmap (move to CPU to save GPU memory)
        # Only store at the same stride intervals as _build_clips() uses
        if store_similarities:
            if start_idx == 0 or (start_idx - self._last_stored_start_idx) >= self._heatmap_stride:
                self.similarity_rows.append(sims.detach().cpu().float())
                self.clip_start_indices.append(start_idx)
                self._last_stored_start_idx = start_idx

        max_sim = sims.float().max().item()  # Convert to float32 for safe extraction
        best_demo_idx = sims.argmax().item()
        clip_threshold = self.per_clip_thresholds[best_demo_idx]

        if max_sim >= clip_threshold:
            current_progress = self.demo_progress[best_demo_idx]
            self.last_raw_delta = current_progress - self.previous_progress
            reward = self.last_raw_delta * (max_sim - clip_threshold) / self.similarity_threshold_offset
            self.previous_progress = current_progress
        else:
            self.last_raw_delta = 0.0
            reward = 0.0

        return reward

    def reset_rollout(self, initial_progress: float = 0.0):
        """
        Reset the rollout buffer and similarity storage for a new episode.

        Args:
            initial_progress: Starting progress value (0–10) matching the env's RSI reset
                              state.  The first step's reward is the delta from this
                              baseline, so it will naturally be ~0 when the agent has not
                              yet moved from the reset position.  Pass 0.0 for episodes
                              that start from the task's initial state.
        """
        self.rollout_buffer = []
        self.similarity_rows = []
        self.clip_start_indices = []
        self._last_stored_start_idx = -999
        self.previous_progress = initial_progress
        self.last_similarities = None  # Reset last similarities
        self.last_raw_delta = 0.0

    def get_similarity_matrix(self):
        """
        Get the similarity matrix built during step() calls.

        Returns:
            sim_matrix: [num_rollout_clips, num_demo_clips] tensor, or None if no clips computed
        """
        if not self.similarity_rows:
            return None
        return torch.stack(self.similarity_rows)

    def get_num_demo_clips(self):
        """Return number of demo clips."""
        return self.demo_embs.size(0)


if __name__ == "__main__":
    import argparse

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
        default=str(parent_dir.parent / "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
        help="Path to the DINOv3 ViT checkpoint."
    )
    parser.add_argument(
        "--comparator_checkpoint_path",
        default=str(parent_dir.parent / "checkpoints/checkpoint_epoch_65.pt"),
        help="Path to the trained comparator checkpoint."
    )
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--clip_stride", type=int, default=5)
    parser.add_argument("--device", default="cuda")

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

    import time

    rollout_frames = model._load_frames_from_dir(args.rollout_dir)
    print(f"Loaded {len(rollout_frames)} rollout frames")
    print(f"Demo clips: {model.get_num_demo_clips()}")

    # Process frames and build similarity matrix incrementally
    print("\nProcessing frames (building heatmap incrementally)...")
    start_time = time.perf_counter()

    rewards = []
    for i, frame in enumerate(rollout_frames):
        reward = model.step(frame, store_similarities=True)
        rewards.append(reward)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(rollout_frames)} frames, "
                  f"{len(model.similarity_rows)} clips computed")

    elapsed = time.perf_counter() - start_time
    valid_rewards = [r for r in rewards if r > -3.0]
    max_reward = max(valid_rewards) if valid_rewards else 0.0

    print(f"\nRollout frames processed: {len(rollout_frames)}")
    print(f"Clips computed: {len(model.similarity_rows)}")
    print(f"Max reward: {max_reward:.4f}")
    print(f"Processing time: {elapsed:.3f}s")

    # Get the similarity matrix built during step() calls
    sim_matrix = model.get_similarity_matrix()
    if sim_matrix is None:
        raise ValueError("No clips were computed. Check if rollout has enough frames.")

    num_rollout, num_demo = sim_matrix.shape
    print(f"\nSimilarity matrix: {num_rollout} x {num_demo}")
    print(f"  Max similarity: {sim_matrix.max().item():.4f}")
    print(f"  Min similarity: {sim_matrix.min().item():.4f}")
    print(f"  Mean similarity: {sim_matrix.mean().item():.4f}")

    # Print per-clip reward (max similarity across all demo clips for each rollout clip)
    clip_rewards_max = sim_matrix.max(dim=1)
    clip_rewards = clip_rewards_max.values
    best_demo_indices = clip_rewards_max.indices
    clip_span = model.clip_span
    print(f"\nPer-clip rewards (max over {num_demo} demo clips):")
    for i in range(num_rollout):
        r_start = model.clip_start_indices[i]
        r_end = r_start + clip_span - 1
        best_d = best_demo_indices[i].item()
        d_start = model.demo_clip_start_indices[best_d]
        d_end = d_start + clip_span - 1
        print(f"  Rollout clip {i:3d} (frames {r_start:4d}-{r_end:4d}): "
              f"reward={clip_rewards[i].item():.4f}  "
              f"best_demo_clip={best_d} (frames {d_start}-{d_end})")

    # Print detailed similarity information instead of saving heatmap
    print(f"\nDetailed similarity matrix ({num_rollout} x {num_demo}):")
    for i in range(min(num_rollout, 20)):  # Show first 20 rollout clips to avoid too much output
        similarities = sim_matrix[i]
        r_start = model.clip_start_indices[i]
        r_end = r_start + clip_span - 1
        print(f"  Rollout clip {i:3d} (frames {r_start:4d}-{r_end:4d}):")
        print(f"    All similarities: {similarities.numpy()}")
        print(f"    Max: {similarities.max().item():.4f}, Mean: {similarities.mean().item():.4f}, Min: {similarities.min().item():.4f}")

    if num_rollout > 20:
        print(f"  ... (showing only first 20 of {num_rollout} rollout clips)")

    print(f"\nSimilarity matrix summary:")
    print(f"  Shape: {sim_matrix.shape}")
    print(f"  Global max: {sim_matrix.max().item():.4f}")
    print(f"  Global min: {sim_matrix.min().item():.4f}")
    print(f"  Global mean: {sim_matrix.mean().item():.4f}")
    print(f"  Global std: {sim_matrix.std().item():.4f}")
