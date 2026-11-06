import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
from typing import List, Dict, Tuple


class ContrastiveVideoDataset(Dataset):
    """
    Dataset for Soft-Nearest Neighbors contrastive learning on video trajectories.

    For each video, samples:
    - One anchor clip from a randomly selected anchor region
    - Multiple positive clips from the same anchor region
    - First-source negatives: from non-anchor regions in the same video
    - Second-source negatives: time-reversed clips from anchor region
    - Third-source negatives: anchors from other videos (handled at batch level)

    Clip sampling uses 25%/50%/25% rule:
    - First frame from first 25% of region
    - Middle 3 frames from middle 50%
    - Last frame from last 25%
    """

    # Configuration by video length category
    CONFIG = {
        'short': {  # < 40 frames
            'anchor_size': (8, 12),
            'num_positives': 2,
            'num_neg_regions': 10,
            'neg_region_size': (7, 15),
            'num_reverse': 2,
        },
        'medium': {  # 40-99 frames
            'anchor_size': (12, 20),
            'num_positives': 3,
            'num_neg_regions': 15,
            'neg_region_size': (10, 25),
            'num_reverse': 3,
        },
        'long': {  # >= 100 frames
            'anchor_size': (20, 40),
            'num_positives': 5,
            'num_neg_regions': 25,
            'neg_region_size': (15, 50),
            'num_reverse': 5,
        },
    }

    def __init__(self, data_dir: str, num_frames: int = 5, image_size: int = 384):
        self.num_frames = num_frames
        self.image_size = image_size

        data_path = Path(data_dir)
        self.jpeg_dir = data_path / "JPEGImages"
        if not self.jpeg_dir.exists():
            raise ValueError(f"JPEGImages directory not found: {self.jpeg_dir}")

        self.videos = self._scan_directory()
        self._print_stats()

    def _scan_directory(self) -> List[Dict]:
        videos = []
        skipped = 0

        for video_dir in sorted(self.jpeg_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            frame_files = sorted(
                list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))
            )
            n_frames = len(frame_files)

            if n_frames < 30:
                skipped += 1
                continue

            category = 'short' if n_frames < 40 else ('medium' if n_frames < 100 else 'long')
            videos.append({
                'video_id': video_dir.name,
                'frame_files': frame_files,
                'num_frames': n_frames,
                'category': category,
            })

        if skipped > 0:
            print(f"Skipped {skipped} videos (< 30 frames)")
        if not videos:
            raise ValueError(f"No valid videos found in {self.jpeg_dir}")
        return videos

    def _print_stats(self):
        counts = {'short': 0, 'medium': 0, 'long': 0}
        for v in self.videos:
            counts[v['category']] += 1
        print(f"Loaded {len(self.videos)} videos: "
              f"short={counts['short']}, medium={counts['medium']}, long={counts['long']}")

    def __len__(self) -> int:
        return len(self.videos)

    def _sample_anchor_region(self, n_frames: int, size_range: Tuple[int, int]) -> Tuple[int, int]:
        """Sample anchor region [start, end] (inclusive)."""
        min_size, max_size = size_range
        region_size = random.randint(min_size, min(max_size, n_frames))
        start = random.randint(0, n_frames - region_size)
        return start, start + region_size - 1

    def _sample_negative_regions(self, n_frames: int, anchor_start: int, anchor_end: int,
                                  num_regions: int, min_size: int, max_size: int) -> List[Tuple[int, int]]:
        """
        Sample negative regions from pre-anchor and post-anchor spaces.
        Regions are split proportionally between spaces based on available size.
        Region sizes are clamped to fit within each space.
        """
        pre_space = anchor_start  # frames 0 to anchor_start-1
        post_space = n_frames - anchor_end - 1  # frames anchor_end+1 to n_frames-1

        # Determine which spaces can fit minimum region size
        pre_valid = pre_space >= min_size
        post_valid = post_space >= min_size

        if not pre_valid and not post_valid:
            return []

        # Split regions proportionally between valid spaces
        if pre_valid and post_valid:
            total_valid_space = pre_space + post_space
            num_pre = round(num_regions * pre_space / total_valid_space)
            num_post = num_regions - num_pre
        elif pre_valid:
            num_pre = num_regions
            num_post = 0
        else:
            num_pre = 0
            num_post = num_regions

        regions = []

        # Sample regions from pre-anchor space (frames 0 to anchor_start-1)
        for _ in range(num_pre):
            actual_max = min(max_size, pre_space)
            region_size = random.randint(min_size, actual_max)
            start = random.randint(0, anchor_start - region_size)
            regions.append((start, start + region_size - 1))

        # Sample regions from post-anchor space (frames anchor_end+1 to n_frames-1)
        for _ in range(num_post):
            actual_max = min(max_size, post_space)
            region_size = random.randint(min_size, actual_max)
            post_start = anchor_end + 1
            start = random.randint(post_start, n_frames - region_size)
            regions.append((start, start + region_size - 1))

        return regions

    def _sample_clip_25_50_25(self, start: int, end: int) -> List[int]:
        """Sample 5 frames using 25%/50%/25% rule from region [start, end]."""
        size = end - start + 1
        q1 = int(0.25 * size)
        q3 = int(0.75 * size)

        first = random.randint(start, start + q1 - 1)
        middle = sorted(random.sample(range(start + q1, start + q3), 3))
        last = random.randint(start + q3, end)

        return [first] + middle + [last]

    def _sample_clip_uniform(self, start: int, end: int) -> List[int]:
        """Sample 5 frames uniformly from region [start, end]."""
        size = end - start + 1
        step = (size - 1) / (self.num_frames - 1)
        return [start + int(i * step) for i in range(self.num_frames)]

    def _sample_clip_reverse(self, start: int, end: int) -> List[int]:
        """Sample 5 frames in reverse order from region [start, end]."""
        all_frames = list(range(start, end + 1))
        return sorted(random.sample(all_frames, self.num_frames), reverse=True)

    def _load_frames(self, frame_files: List[Path], indices: List[int]) -> torch.Tensor:
        """Load frames at specified indices, returns [T, H, W, 3]."""
        frames = []
        for idx in indices:
            idx = min(idx, len(frame_files) - 1)
            img = Image.open(frame_files[idx]).convert('RGB')
            img = TF.resize(img, [self.image_size, self.image_size], antialias=True)
            frames.append(TF.to_tensor(img).permute(1, 2, 0))
        return torch.stack(frames)

    def _apply_flip(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply horizontal flip to all frames. frames: [T, H, W, C]"""
        return torch.flip(frames, dims=[2])

    def _apply_augmentation(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply RandomResizedCrop + ColorJitter consistently to all frames."""
        # frames: [T, H, W, C] -> [T, C, H, W] for torchvision ops
        frames = frames.permute(0, 3, 1, 2)

        # Get crop params using image dimensions
        H, W = frames.shape[2], frames.shape[3]
        scale = random.uniform(0.9, 1.0)
        crop_size = int(H * scale)
        i = random.randint(0, H - crop_size)
        j = random.randint(0, W - crop_size)

        # Color jitter params
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        saturation = random.uniform(0.8, 1.2)
        hue = random.uniform(-0.1, 0.1)

        # Apply same augmentation to all frames
        augmented = []
        for f in frames:
            aug = TF.resized_crop(f, i, j, crop_size, crop_size, (self.image_size, self.image_size))
            aug = TF.adjust_brightness(aug, brightness)
            aug = TF.adjust_contrast(aug, contrast)
            aug = TF.adjust_saturation(aug, saturation)
            aug = TF.adjust_hue(aug, hue)
            augmented.append(aug)

        # [T, C, H, W] -> [T, H, W, C]
        return torch.stack(augmented).permute(0, 2, 3, 1)

    def __getitem__(self, idx: int) -> Dict:
        video = self.videos[idx]
        frame_files = video['frame_files']
        n_frames = video['num_frames']
        cfg = self.CONFIG[video['category']]

        # Stage 1: Sample anchor region
        anchor_start, anchor_end = self._sample_anchor_region(n_frames, cfg['anchor_size'])

        # Stage 2: Sample anchor clip (with augmentation)
        anchor_indices = self._sample_clip_25_50_25(anchor_start, anchor_end)
        anchor = self._load_frames(frame_files, anchor_indices)
        anchor = self._apply_augmentation(anchor)

        # Flip decision shared by anchor and positives
        should_flip = random.random() < 0.5
        if should_flip:
            anchor = self._apply_flip(anchor)

        # Stage 3: Sample positive clips (augmented, same flip)
        positives = []
        for _ in range(cfg['num_positives']):
            pos_indices = self._sample_clip_25_50_25(anchor_start, anchor_end)
            pos = self._load_frames(frame_files, pos_indices)
            pos = self._apply_augmentation(pos)
            if should_flip:
                pos = self._apply_flip(pos)
            positives.append(pos)

        # Stage 4: Sample first-source negatives from non-anchor regions
        min_neg_size, max_neg_size = cfg['neg_region_size']
        neg_regions = self._sample_negative_regions(
            n_frames, anchor_start, anchor_end,
            cfg['num_neg_regions'], min_neg_size, max_neg_size
        )

        first_source_negs = []
        for neg_start, neg_end in neg_regions:
            neg_indices = self._sample_clip_uniform(neg_start, neg_end)
            neg = self._load_frames(frame_files, neg_indices)
            neg = self._apply_augmentation(neg)
            if random.random() < 0.5:
                neg = self._apply_flip(neg)
            first_source_negs.append(neg)

        # Stage 5: Sample second-source negatives (reverse clips from anchor region)
        second_source_negs = []
        for _ in range(cfg['num_reverse']):
            rev_indices = self._sample_clip_reverse(anchor_start, anchor_end)
            rev = self._load_frames(frame_files, rev_indices)
            rev = self._apply_augmentation(rev)
            if random.random() < 0.5:
                rev = self._apply_flip(rev)
            second_source_negs.append(rev)

        return {
            'video_id': video['video_id'],
            'anchor': anchor,
            'positives': positives,
            'first_source_negatives': first_source_negs,
            'second_source_negatives': second_source_negs,
            'category': video['category'],
        }


def collate_contrastive_batch(batch: List[Dict]) -> Dict:
    """Collate function handling variable numbers of positives and negatives."""
    return {
        'video_ids': [item['video_id'] for item in batch],
        'anchors': torch.stack([item['anchor'] for item in batch]),
        'positives': [item['positives'] for item in batch],
        'first_source_negatives': [item['first_source_negatives'] for item in batch],
        'second_source_negatives': [item['second_source_negatives'] for item in batch],
        'categories': [item['category'] for item in batch],
    }
