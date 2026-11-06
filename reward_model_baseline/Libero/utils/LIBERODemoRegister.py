"""Load LIBERO HDF5 demonstrations and materialize reference frames on demand."""

import os
import re
import h5py
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from reward_model_baseline.Libero.constants.libero_task_mapping import (
    TASK_ID_TO_ENV_NAME,
    ENV_NAME_TO_TASK_ID,
    SUITE,
)
from reward_model_baseline.Libero.constants.env_constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    AGENTVIEW_CAMERA_NAME,
    RENDER_INTERVAL,
)

class LIBERODemoRegister:

    _global_instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        # Ensure only one instance of LIBERODemoRegister exists (singleton pattern)
        if cls._global_instance is None:
            cls._global_instance = super(LIBERODemoRegister, cls).__new__(cls)
        return cls._global_instance

    @classmethod
    def get_global_instance(cls):
        """
        Get the global instance of LIBERODemoRegister.
        """
        if cls._global_instance is None:
            cls._global_instance = cls()
        return cls._global_instance

    @classmethod
    def reset_global_instance(cls):
        """
        Reset the global instance of LIBERODemoRegister.
        Useful for testing or if you need to reload demo data.
        """
        cls._global_instance = None
        cls._initialized = False

    def __init__(self, demos_path:Optional[str] = None):
        """ Intialize the LIBERODemoRegister (this only runs once due to singleton pattern)

        Args:
            demos_path: Path to the data/LIBERO_datasets directory
        """
        if LIBERODemoRegister._initialized:
            return  # Already initialized, skip re-initialization

        # Get project root (3 levels up from utils/)
        utils_path = os.path.dirname(os.path.abspath(__file__))
        self.project_path = os.path.dirname(os.path.dirname(os.path.dirname(utils_path)))

        if demos_path is not None:
            self.demos_path = demos_path
        else:
            self.demos_path = os.path.join(
                self.project_path, "data", "LIBERO_datasets"
            )

        self._demo_cache: Dict[int,Dict[int, Dict[str,Any]]] = {};
    
        LIBERODemoRegister._initialized = True;
        
    def load_all_demos(self, task_id: int) -> Dict[int, Dict[str, Any]]:
        """
        Load all demos for a task from HDF5 into memory. Results are cached.

        Args:
            task_id: Task index (0-9 for LIBERO-10).

        Returns:
            Dict mapping demo_id -> demo dict:
            {task_id, task_name, seed, images, states, num_frames}
        """
        # Check cache first
        if task_id in self._demo_cache and self._demo_cache[task_id]:
            return self._demo_cache[task_id]

        if task_id not in TASK_ID_TO_ENV_NAME:
            raise ValueError(f"task_id={task_id} is not a valid {SUITE} task (valid: 0-{len(TASK_ID_TO_ENV_NAME)-1})")

        # Load the hdf5 file into demos
        hdf5_path = os.path.join(self.demos_path, SUITE,
                                TASK_ID_TO_ENV_NAME[task_id] + '_demo.hdf5')

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 demo file not found: {hdf5_path}")

        f = h5py.File(hdf5_path, 'r')

        # Loop through the open file and parse it into a dict
        demos = {}
        for key in f['data'].keys():
            demo_id = int(key.split('_')[1])  # "demo_3" -> 3
            images = np.array(f[f'data/{key}/obs/agentview_rgb'])  # (T, H, W, C)
            states = np.array(f[f'data/{key}/states'])              # (T, state_dim)
            actions = np.array(f[f'data/{key}/actions'])            # (T, act_dim)
            demos[demo_id] = {
                'task_id': task_id,
                'task_name': TASK_ID_TO_ENV_NAME[task_id],
                'seed': demo_id,
                'images': images,
                'states': states,
                'actions': actions,
                'num_frames': len(images),
            }
        f.close()

        # Cache all demos
        self._demo_cache[task_id] = demos

        return demos
    
    def load_demo(self, task_id: int, demo_id: int = 0) -> Dict[str,Any]:
        """
        Load a single demo by task and demo index.

        Args:
            task_id: Task index.
            demo_id: Demo index (matches demo_N in HDF5). Defaults to 0.

        Returns:
            Demo dict with keys: task_id, task_name, seed, images, states, num_frames.
        """
        all_demos = self.load_all_demos(task_id)
        return all_demos[demo_id]
    
    def get_demo_image(
        self, task_id: int, frame_index: int, seed: Optional[int] = 0
    ) -> np.ndarray:
        """
        Get a single image frame from a demo. Used by ReferenceAnchoredRewardModel.

        Args:
            task_id: Task index.
            frame_index: Frame index (0-based).
            seed: Demo index. Defaults to 0.

        Returns:
            Image as numpy array (H, W, C) uint8. Raw from HDF5, not flipped.
        """
        demo = self.load_demo(task_id, demo_id=seed)
        if frame_index >= demo['num_frames']:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {demo['num_frames']})"
            )
        return demo['images'][frame_index]
    
    def get_demo_state(
        self, task_id: int, frame_index: int, seed: Optional[int] = 0
    ) -> Dict[str, Any]:
        """
        Get a single MuJoCo state from a demo. Used by LIBEROEnvWrapper for random state intialisation.
        (Reference State Initialization — start episode from mid-demo state).

        Args:
            task_id: Task index.
            frame_index: Frame index (0-based).
            seed: Demo index. Defaults to 0.

        Returns:
            State vector (state_dim,) float64.
        """
        demo = self.load_demo(task_id, demo_id=seed)
        if frame_index >= demo['num_frames']:
            raise IndexError(
                f"Frame index {frame_index} out of range [0, {demo['num_frames']})"
            )
        return demo['states'][frame_index]

    def get_demo_folder_path(self, task_id: int, demo_id: int = 0, seed: Optional[int] = None) -> str:
        """
        Get path to JPEG frames folder for a demo.

        Replays demo actions through the simulator to capture native 384x384 images.
        Results are cached — if the folder already exists with frames, returns immediately.

        Args:
            task_id: Task index.
            demo_id: Demo index. Defaults to 0.
            seed: Alias for demo_id.

        Returns:
            Path to folder containing frame_0000.jpg, frame_0001.jpg, ...
        """
        if seed is not None:
            demo_id = seed

        out_dir = os.path.join(self.project_path, 'data', f'{SUITE}_{TASK_ID_TO_ENV_NAME[task_id]}_demo{demo_id}_frames_render_interval_{RENDER_INTERVAL}')

        # Return cached if already rendered
        if os.path.exists(out_dir) and len(glob(os.path.join(out_dir, 'frame_*.jpg'))) > 0:
            return out_dir

        os.makedirs(out_dir, exist_ok=True)

        # Lazy import to avoid circular dependency
        from reward_model_baseline.Libero.envs.libero_env_wrapper import LiberoEnvWrapper

        demo = self.load_demo(task_id, demo_id=demo_id)


        env = LiberoEnvWrapper(
            task_id=task_id,
            use_demo_init=False,
            demo_init_prob=0.0,
            render_interval=RENDER_INTERVAL,
        )
        env.env.reset()  # Skip LiberoEnvWrapper.reset() to avoid applying init noise during frame collection
        imgs = []
        for state_idx in range(0,len(demo['states']), RENDER_INTERVAL):
            env.env.set_init_state(demo['states'][state_idx])
            img = env.env.env.sim.render(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, camera_name=AGENTVIEW_CAMERA_NAME)
            imgs.append(img[::-1])  # Flip vertically 
        env.close()

        # Save frames as JPEGs
        for i, frame in enumerate(imgs):
            Image.fromarray(frame).save(os.path.join(out_dir, f'frame_{i:04d}.jpg'))

        print(f"Rendered {len(imgs)} native 384x384 demo frames to {out_dir}")
        return out_dir
    
    def get_available_task_ids(self) -> List[int]:
        """
        Get list of task IDs that have demos available.

        Returns:
            List of available task IDs.
        """
        available = []
        for task_id in TASK_ID_TO_ENV_NAME.keys():
            available.append(task_id)
        return sorted(available)
    
    def get_available_seeds(self, task_id: int) -> List[int]:
        """
        Get list of available seeds for a given task ID.

        Args:
            task_id: The task ID.

        Returns:
            Sorted list of available seeds.
        """
        return sorted(self.load_all_demos(task_id).keys());
    
    def clear_cache(self):
        """Clear the demo cache to free memory."""
        self._demo_cache.clear()