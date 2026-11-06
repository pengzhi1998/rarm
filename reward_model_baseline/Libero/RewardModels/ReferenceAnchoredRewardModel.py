"""LIBERO demonstration and RARM inference wrapper."""

import numpy as np
import os
import sys
import imageio
from pathlib import Path
from PIL import Image

# Get project root for checkpoint paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

# Add Reference_Anchored_RM to path for import
reference_rm_path = os.path.join(project_root, "Reference-Anchored_RM")
if reference_rm_path not in sys.path:
    sys.path.insert(0, reference_rm_path)

# Import ClipRewardModel
from scripts.reward_model import ClipRewardModel
from reward_model_baseline.Libero.utils.LIBERODemoRegister import LIBERODemoRegister


class ReferenceAnchoredRewardModel:
    """
    Singleton registry for Reference-Anchored Reward Model.
    """

    _global_instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of ReferenceAnchoredRewardModel can exist."""
        if cls._global_instance is None:
            cls._global_instance = super(ReferenceAnchoredRewardModel, cls).__new__(cls)
        return cls._global_instance

    @classmethod
    def get_global_instance(cls, task_id=0, vit_checkpoint_path=None, comparator_checkpoint_path=None,
                           num_frames=5, frame_stride=1, clip_stride=1, device="cuda",
                           demo_seed=None, similarity_threshold_offset: float = 0.5,
                           use_cross_attn: bool = True, use_temporal_processor: bool = True):
        """
        Get or create the global ReferenceAnchoredRewardModel instance.

        Args:
            task_id: The task ID for loading demo images
            vit_checkpoint_path: Path to the DINOv3 ViT checkpoint
            comparator_checkpoint_path: Path to the trained comparator checkpoint
            num_frames: Number of frames per clip
            frame_stride: Stride between frames in a clip
            clip_stride: Stride between clips
            device: Device to run the model on (cuda or cpu)
            demo_seed: Specific demo seed to use for reference images (None = first available)
        """
        if cls._global_instance is None:
            cls._global_instance = cls(
                task_id=task_id,
                vit_checkpoint_path=vit_checkpoint_path,
                comparator_checkpoint_path=comparator_checkpoint_path,
                num_frames=num_frames,
                frame_stride=frame_stride,
                clip_stride=clip_stride,
                device=device,
                demo_seed=demo_seed,
                similarity_threshold_offset=similarity_threshold_offset,
                use_cross_attn=use_cross_attn,
                use_temporal_processor=use_temporal_processor,
            )
        return cls._global_instance

    @classmethod
    def reset_global_instance(cls):
        """Reset the global instance. Useful for testing or reinitialization."""
        cls._global_instance = None
        cls._initialized = False

    def __init__(
        self,
        vit_checkpoint_path=None,
        comparator_checkpoint_path=None,
        num_frames=5,
        frame_stride=1,
        clip_stride=1,
        device="cuda",
        task_id=0,
        demo_seed=None,
        similarity_threshold_offset: float = 0.5,
        use_cross_attn: bool = True,
        use_temporal_processor: bool = True,
    ):
        """
        Initialize the Reference-Anchored Reward Model (only runs once due to singleton pattern).

        Args:
            vit_checkpoint_path: Path to the DINOv3 ViT checkpoint
            comparator_checkpoint_path: Path to the trained comparator checkpoint
            num_frames: Number of frames per clip
            frame_stride: Stride between frames in a clip
            clip_stride: Stride between clips
            device: Device to run the model on (cuda or cpu)
            task_id: The task ID for loading demo images
            demo_seed: Specific demo seed for reference images (None = first available)
        """
        if ReferenceAnchoredRewardModel._initialized:
            return

        # Set default paths if not provided
        if vit_checkpoint_path is None:
            vit_checkpoint_path = "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        if comparator_checkpoint_path is None:
            comparator_checkpoint_path = "checkpoints/checkpoint_epoch_65.pt"

        # Convert relative paths to absolute paths from project root
        if not os.path.isabs(vit_checkpoint_path):
            vit_checkpoint_path = os.path.join(project_root, vit_checkpoint_path)
        if not os.path.isabs(comparator_checkpoint_path):
            comparator_checkpoint_path = os.path.join(project_root, comparator_checkpoint_path)

        self.vit_checkpoint_path = vit_checkpoint_path
        self.comparator_checkpoint_path = comparator_checkpoint_path
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.clip_stride = clip_stride
        self.device = device
        self.model = None
        self.project_root = project_root
        self.task_id = task_id
        self.demo_register = LIBERODemoRegister.get_global_instance()
        self.demo_images_path = self.demo_register.get_demo_folder_path(self.task_id, seed=demo_seed)

        # Initialize the reward model with reference images path
        print(f"Loading reference images from: {self.demo_images_path}")
        self.model = ClipRewardModel(
            vit_checkpoint_path=self.vit_checkpoint_path,
            comparator_checkpoint_path=self.comparator_checkpoint_path,
            demo_path=self.demo_images_path,
            num_frames=self.num_frames,
            frame_stride=self.frame_stride,
            clip_stride=self.clip_stride,
            device=self.device,
            similarity_threshold_offset=similarity_threshold_offset,
            use_cross_attn=use_cross_attn,
            use_temporal_processor=use_temporal_processor,
        )

        ReferenceAnchoredRewardModel._initialized = True
        print("ReferenceAnchoredRewardModel initialized successfully (singleton)")

    def compute_reward_from_images(
        self,
        query_images: list[np.ndarray],
        save_videos: bool = False,
        initial_progress: float = 0.0,
        return_similarity_info: bool = False,
        save_chunk_frames: bool = True,
        chunk_id: str = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[list[float], dict] | list[float]:
        """
        Compute rewards from a sequence of images using a reference-anchored reward model.

        Args:
            query_images: List of query image frames as numpy arrays (H, W, C).
            save_videos: Whether to save query and reference videos to /temp folder.
            initial_progress: Starting progress value (0–10) matching the env's RSI reset
                              state.  Pass 0.0 for episodes that begin from the task's
                              initial state.  The first step's reward will be ~0 because
                              it is the delta from this baseline, not the baseline itself.
            return_similarity_info: If True, also return similarity information dict.
            save_chunk_frames: Whether to save individual chunk frames to disk.
            chunk_id: Identifier for this chunk (e.g., "env0_episode1_chunk0").
            verbose: If True, print per-frame similarity details and reward summary.

        Returns:
            If return_similarity_info is False:
                List of reward values for each query image (progress deltas or -0.5;
                leading frames before the first complete clip are filled with 0.0).
            If return_similarity_info is True:
                Tuple of (rewards list, similarity_info dict) where similarity_info contains:
                - max_similarities: List of maximum similarity values for each clip
                - best_ref_indices: List of best matching reference clip indices
                - all_similarities: List of all similarity values for each clip
        """
        # Filter out None values
        query_imgs = [img for img in query_images if img is not None]

        if not os.path.exists(self.demo_images_path):
            raise ValueError(f"Reference path does not exist: {self.demo_images_path}")
        if not query_imgs:
            raise ValueError("Query images cannot be empty")

        # Save videos to /temp folder for visualization
        video_output_dir = os.path.join(self.project_root, "temp")
        os.makedirs(video_output_dir, exist_ok=True)

        if query_imgs and save_videos:
            query_video_path = os.path.join(video_output_dir, "query_images.mp4")
            imageio.mimsave(query_video_path, query_imgs, fps=30)
            print(f"Query video saved to: {query_video_path}")

        # Save individual chunk frames if requested
        if query_imgs and save_chunk_frames and chunk_id:
            chunk_frames_dir = os.path.join(video_output_dir, "chunk_frames", chunk_id)
            os.makedirs(chunk_frames_dir, exist_ok=True)
            for frame_idx, frame in enumerate(query_imgs):
                frame_path = os.path.join(chunk_frames_dir, f"frame_{frame_idx:06d}.jpg")
                # Convert numpy array to PIL Image and save
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                img_pil = Image.fromarray(frame)
                img_pil.save(frame_path)
            print(f"Chunk frames saved to: {chunk_frames_dir} ({len(query_imgs)} frames)")

        reference_image_files = sorted(Path(self.demo_images_path).glob("frame_*.jpg"))
        if reference_image_files and save_videos:
            reference_imgs = [np.array(Image.open(img_path)) for img_path in reference_image_files]
            reference_video_path = os.path.join(video_output_dir, "reference_images.mp4")
            imageio.mimsave(reference_video_path, reference_imgs, fps=30)
            print(f"Reference video saved to: {reference_video_path}")

        # Reset rollout with the episode's starting progress before processing frames.
        self.model.reset_rollout(initial_progress=initial_progress)

        # Compute rewards for each query image.
        # step() returns None for the first (clip_span - 1) frames (sentinel),
        # then either a progress delta (uncapped) or -0.5.
        rewards = []
        max_similarities = []
        best_ref_indices = []
        all_similarities = [] # Store all similarity values for detailed analysis
        image_progress = [] # Save the progress information

        # Optional: save above-threshold clips
        positives_dir = kwargs.get("positives_dir")
        env_id = kwargs.get("env_id", 0)
        episode_id = kwargs.get("episode_id", 0)

        for frame_idx, frame in enumerate(query_imgs):
            prev_progress = self.model.previous_progress  # Capture before step()
            # Don't store similarities in the model to save memory
            reward = self.model.step(frame, store_similarities=False)
            rewards.append(reward)

            if return_similarity_info:
                hw = getattr(self.model, 'max_reached_progress', self.model.previous_progress)
                image_progress.append(hw / 10.0)  # normalized to [0, 1]

            # Save above-threshold clips to positives directory
            if positives_dir and self.model.last_similarities is not None:
                sims = self.model.last_similarities
                ms = float(np.max(sims))
                br = int(np.argmax(sims))
                if ms >= self.model.per_clip_thresholds[br]:
                    clip_start = frame_idx - self.model.clip_span + 1
                    sim_str = f"{ms:.4f}".replace('.', '').replace('-', 'n')
                    clip_name = f"env{env_id}_ep{episode_id}_roll{clip_start}_ref{br}_sim{sim_str}"
                    clip_dir = os.path.join(positives_dir, clip_name)
                    os.makedirs(clip_dir, exist_ok=True)
                    for ci in range(self.model.num_frames):
                        fi = clip_start + ci
                        if 0 <= fi < len(query_imgs):
                            img = query_imgs[fi]
                            if img.dtype != np.uint8:
                                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                            Image.fromarray(img).save(os.path.join(clip_dir, f"frame_{fi:04d}.jpg"))

            # Collect similarity info if requested
            if return_similarity_info and hasattr(self.model, 'last_similarities') and self.model.last_similarities is not None:
                similarities = self.model.last_similarities
                max_sim = np.max(similarities)
                best_ref_idx = np.argmax(similarities)
                max_similarities.append(max_sim)
                best_ref_indices.append(int(best_ref_idx))
                all_similarities.append(similarities.copy())

                if verbose:
                    ref_start = self.model.demo_clip_start_indices[int(best_ref_idx)]
                    ref_frames = self.model._select_clip_indices(ref_start)
                    clip_threshold = self.model.per_clip_thresholds[int(best_ref_idx)]
                    confidence = (max_sim - clip_threshold) / 0.5
                    prev_prog = prev_progress * 10  # Scale 0-10 → 0-100
                    progress = self.model.previous_progress * 10
                    delta = self.model.last_raw_delta * 10
                    # Show middle frame of the rollout clip instead of buffer index
                    mid_offset = (self.model.clip_span - 1) // 2
                    mid_frame = frame_idx - (self.model.clip_span - 1) + mid_offset
                    print(f"  Frame {mid_frame:3d}: max_sim={max_sim:.4f}, "
                          f"best_ref_clip={best_ref_idx} (ref frames {ref_frames}), "
                          f"prev={prev_prog:.1f}%, progress={progress:.1f}%, "
                          f"delta={delta:.2f}, confidence={confidence:.4f}")

        # Replace leading None sentinels with 0.0.
        # These correspond to frames before the first complete clip can be formed;
        # no progress information is available yet, so reward = 0.
        for i in range(len(rewards)):
            if rewards[i] is None:
                rewards[i] = 0.0
            else:
                break

        if verbose:
            max_reward = max(rewards) if rewards else 0.0
            print(f"Computed {len(rewards)} rewards, max reward: {max_reward:.4f}")

        if return_similarity_info and max_similarities:
            # Create comprehensive dict with all similarity info
            similarity_info = {
                'max_similarities': max_similarities,
                'best_ref_indices': best_ref_indices,
                'all_similarities': all_similarities,
                'image_progress': image_progress,
            }

            if verbose:
                avg_max_sim = np.mean(max_similarities)
                max_max_sim = np.max(max_similarities)
                max_sim_idx = np.argmax(max_similarities)
                best_ref_at_max = best_ref_indices[max_sim_idx] if max_sim_idx < len(best_ref_indices) else -1
                most_common_ref = max(set(best_ref_indices), key=best_ref_indices.count) if best_ref_indices else -1

                print(f"  Max similarity: {max_max_sim:.4f} (at frame {max_sim_idx}, matched ref clip {best_ref_at_max})")
                print(f"  Avg similarity: {avg_max_sim:.4f}, Most matched ref clip: {most_common_ref}")

            return rewards, similarity_info
        elif return_similarity_info:
            return rewards, {'max_similarities': [], 'best_ref_indices': [], 'image_progress': image_progress}
        else:
            return rewards
