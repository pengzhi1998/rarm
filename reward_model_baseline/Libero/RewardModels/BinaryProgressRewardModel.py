import os
import sys

import numpy as np

# Ensure Reference-Anchored_RM is on sys.path for ClipRewardModel import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
reference_rm_path = os.path.join(project_root, "Reference-Anchored_RM")
if reference_rm_path not in sys.path:
    sys.path.insert(0, reference_rm_path)

from scripts.reward_model import ClipRewardModel
from reward_model_baseline.Libero.RewardModels.ReferenceAnchoredRewardModel import (
    ReferenceAnchoredRewardModel,
)


class BinaryProgressClipRewardModel(ClipRewardModel):
    """ClipRewardModel variant returning +1 for forward progress, 0 otherwise.

    Tracks a max_reached_progress high-water mark that only updates upward.
    A reward of +1 is assigned when:
      1. The rollout clip's similarity to its best-matching demo clip >= that clip's threshold
      2. The matched demo clip's progress > max_reached_progress
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_reached_progress = 0.0
        self._max_demo_progress = max(self.demo_progress)

    @classmethod
    def from_existing(cls, base_model):
        """Create from an existing ClipRewardModel, sharing VideoComparator and demo data."""
        instance = object.__new__(cls)
        # Shared references (read-only after initialization)
        for attr in (
            'num_frames', 'frame_stride', 'clip_span', 'clip_stride', 'device',
            'model', 'demo_embs', 'demo_clip_start_indices', 'total_ref_frames',
            'demo_progress', 'max_self_sim', 'per_clip_thresholds', '_heatmap_stride',
            'similarity_threshold_offset', 'use_cross_attn', 'use_temporal_processor',
        ):
            setattr(instance, attr, getattr(base_model, attr))
        # Own mutable state
        instance.rollout_buffer = []
        instance.similarity_rows = []
        instance.clip_start_indices = []
        instance._last_stored_start_idx = -999
        instance.last_similarities = None
        instance.previous_progress = 0.0
        instance.last_raw_delta = 0.0
        instance.max_reached_progress = 0.0
        instance._max_demo_progress = max(instance.demo_progress)
        return instance

    def step(self, rollout_frame, store_similarities=True):
        """Process a single frame and return a binary progress reward.

        Returns:
            1.0 if forward progress is detected (new high-water mark),
            0.0 if no progress or below threshold (or max progress already reached),
            None if not enough frames buffered yet (sentinel).
        """
        if self.max_reached_progress >= self._max_demo_progress:
            return 0.0

        parent_reward = super().step(rollout_frame, store_similarities=store_similarities)
        if parent_reward is None:
            return None
        if self.last_similarities is None:
            return 0.0

        max_sim = float(np.max(self.last_similarities))
        best_demo_idx = int(np.argmax(self.last_similarities))
        clip_threshold = self.per_clip_thresholds[best_demo_idx]

        if max_sim >= clip_threshold:
            current_progress = self.demo_progress[best_demo_idx]
            if current_progress > self.max_reached_progress:
                self.max_reached_progress = current_progress
                return 1.0

        return 0.0

    def reset_rollout(self, initial_progress=0.0):
        super().reset_rollout(initial_progress)
        self.max_reached_progress = initial_progress


class BinaryProgressRewardModel(ReferenceAnchoredRewardModel):
    """Singleton wrapper using BinaryProgressClipRewardModel.

    Shares the VideoComparator from ReferenceAnchoredRewardModel to avoid
    loading model weights twice.  Inherits compute_reward_from_images() which
    calls self.model.step() — using BinaryProgressClipRewardModel's overridden
    step() for binary progress rewards.
    """

    _global_instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._global_instance is None:
            cls._global_instance = object.__new__(cls)
        return cls._global_instance

    @classmethod
    def get_global_instance(
        cls,
        task_id=0,
        task_description=None,
        demo_seed=None,
        vit_checkpoint_path=None,
        comparator_checkpoint_path=None,
        device="cuda",
        similarity_threshold_offset: float = 0.5,
        use_cross_attn: bool = True,
        use_temporal_processor: bool = True,
    ):
        if cls._global_instance is None:
            cls._global_instance = cls(
                task_id=task_id,
                demo_seed=demo_seed,
                vit_checkpoint_path=vit_checkpoint_path,
                comparator_checkpoint_path=comparator_checkpoint_path,
                device=device,
                similarity_threshold_offset=similarity_threshold_offset,
                use_cross_attn=use_cross_attn,
                use_temporal_processor=use_temporal_processor,
            )
        return cls._global_instance

    @classmethod
    def reset_global_instance(cls):
        cls._global_instance = None
        cls._initialized = False

    def __init__(
        self,
        task_id=0,
        demo_seed=None,
        vit_checkpoint_path=None,
        comparator_checkpoint_path=None,
        device="cuda",
        similarity_threshold_offset: float = 0.5,
        use_cross_attn: bool = True,
        use_temporal_processor: bool = True,
        **kwargs,
    ):
        if BinaryProgressRewardModel._initialized:
            return

        # Reuse existing ReferenceAnchoredRewardModel singleton (same GPU model).
        # Threshold offset and architecture toggles are baked into the ClipRewardModel here.
        base_rm = ReferenceAnchoredRewardModel.get_global_instance(
            task_id=task_id,
            demo_seed=demo_seed,
            vit_checkpoint_path=vit_checkpoint_path,
            comparator_checkpoint_path=comparator_checkpoint_path,
            device=device,
            similarity_threshold_offset=similarity_threshold_offset,
            use_cross_attn=use_cross_attn,
            use_temporal_processor=use_temporal_processor,
        )

        # Copy wrapper-level attributes needed by compute_reward_from_images()
        self.project_root = base_rm.project_root
        self.demo_images_path = base_rm.demo_images_path
        print(self.demo_images_path)

        # Create BinaryProgressClipRewardModel sharing the VideoComparator
        self.model = BinaryProgressClipRewardModel.from_existing(base_rm.model)

        BinaryProgressRewardModel._initialized = True
        print("BinaryProgressRewardModel initialized (shares model with ReferenceAnchoredRewardModel)")
