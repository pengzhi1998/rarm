from __future__ import annotations

from typing import Any, Optional

import numpy as np
import gymnasium as gym

from libero.libero import benchmark
from libero.libero.envs.env_wrapper import ControlEnv

from reward_model_baseline.Libero.constants.env_constants import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    AGENTVIEW_CAMERA_NAME,
    WRIST_CAMERA_NAME,
    DEMO_INIT_START_RATIO,
    DEMO_INIT_END_RATIO,
    DEMO_INIT_PROB,
    ACTION_BUFFER_LENGTH,
    ACTION_BUFFER_NOISE,
    ACTION_BUFFER_PROB,
    INIT_ARM_NOISE_STD,
    INIT_OBJ_NOISE_STD,
    POLICY_IMG_SIZE,
    RENDER_INTERVAL,
    SUCCESS_DELAY_STEPS,
)
from reward_model_baseline.Libero.constants.libero_task_mapping import SUITE


class LiberoEnvWrapper(gym.Env):
    def __init__(
        self,
        task_id,
        render_interval=RENDER_INTERVAL,
        use_demo_init=False,
        demo_init_prob=DEMO_INIT_PROB,
        is_eval_env=False,
        n_eval_episodes=10,
        demo_seed=None,
    ):
        """Wrap a LIBERO OffScreenRenderEnv as a gymnasium-compatible env.

        Args:
            task_id: Index into the LIBERO suite.
            render_interval: Capture RM image into info every N steps.
            use_demo_init: Enable Reference State Initialization (RSI) on reset.
            demo_init_prob: Probability of applying RSI when use_demo_init is True.
            is_eval_env: If True, use scheduled RSI progression across eval episodes.
            n_eval_episodes: Number of eval episodes (used for scheduled RSI spacing).
            demo_seed: Which demo index to use for RSI. Independent of the environment RNG seed.
        """
        bench = benchmark.get_benchmark_dict()[SUITE]()
        bddl_path = bench.get_task_bddl_file_path(task_id)

        self.task_id = task_id
        self.render_interval = render_interval
        self.use_demo_init = use_demo_init
        self.demo_init_prob = demo_init_prob
        self.is_eval_env = is_eval_env
        self.n_eval_episodes = n_eval_episodes
        self._step_count = 0
        self._render_enabled = True
        self._wrist_render_enabled = False
        self._policy_images_enabled = False
        self._demo_init_progress = 0.0
        self._demo_register = None
        self._demo_data = None
        self._demo_seed = demo_seed
        self._action_buffer: list = []
        self._eval_episode_count = 0
        self._success_delay_counter = 0

        self.env = ControlEnv(
            bddl_file_name=bddl_path,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            horizon=500,
        )

        _dummy_obs = self.env.reset()
        _flat = self.flatten_obs(_dummy_obs)
        self._obs_dim = _flat.shape[0]

    @property
    def observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float64)

    @property
    def action_space(self):
        low, high = self.env.env.action_spec
        return gym.spaces.Box(low=low, high=high, dtype=np.float64)

    def set_demo_init_prob(self, prob: float):
        self.demo_init_prob = float(np.clip(prob, 0.0, 1.0))

    def set_render_enabled(self, enabled: bool):
        self._render_enabled = enabled

    def set_wrist_render_enabled(self, enabled: bool):
        self._wrist_render_enabled = enabled

    def set_policy_images_enabled(self, enabled: bool):
        """Enable per-step 84×84 dual-camera renders. Called by drqv2_runner; off by default."""
        self._policy_images_enabled = enabled

    def _load_demo_data(self) -> Optional[dict]:
        if self._demo_data is not None:
            return self._demo_data
        try:
            from reward_model_baseline.Libero.utils.LIBERODemoRegister import LIBERODemoRegister
            self._demo_register = LIBERODemoRegister.get_global_instance()
            self._demo_data = self._demo_register.load_all_demos(self.task_id)
            return self._demo_data
        except (ImportError, FileNotFoundError) as e:
            print(f"Warning: Could not load demo data: {e}")
            return None

    def _get_demo_state_at_random_step(self) -> Optional[Any]:
        """Sample a MuJoCo state from a demo trajectory for RSI.

        Sampling range: [DEMO_INIT_START_RATIO, DEMO_INIT_END_RATIO] of the trajectory.
        Fills self._action_buffer with the next ACTION_BUFFER_LENGTH demo actions + noise
        with probability ACTION_BUFFER_PROB.

        For eval envs, steps through evenly-spaced progress ratios across episodes.
        """
        all_demos = self._load_demo_data()
        if all_demos is None or len(all_demos) == 0:
            return None

        chosen_seed = self._demo_seed if (self._demo_seed is not None and self._demo_seed in all_demos) else 0
        demo_data = all_demos[chosen_seed]
        states = demo_data['states']   # (T, state_dim)
        actions = demo_data['actions'] # (T, act_dim)
        n_states = len(states)

        if n_states == 0:
            return None

        if self.is_eval_env:
            # Evenly-spaced progress ratios: 0, 1/2n, 2/2n, ..., (n-1)/2n
            ratio = (self._eval_episode_count % self.n_eval_episodes) / (2 * self.n_eval_episodes)
            random_step_idx = int(n_states * ratio)
            self._eval_episode_count += 1
        else:
            start_index = int(n_states * DEMO_INIT_START_RATIO)
            end_index   = int(n_states * DEMO_INIT_END_RATIO)
            # Reject terminal states (task already complete)
            for _ in range(10):
                random_step_idx = np.random.randint(start_index, end_index)
                self.env.set_init_state(states[random_step_idx])
                if not self.env.check_success():
                    break

        self._demo_init_progress = (random_step_idx + 1) / n_states * 10.0

        # Action buffer: queue next ACTION_BUFFER_LENGTH demo actions + Gaussian noise
        # Disabled for eval envs — eval measures pure agent performance
        if not self.is_eval_env and np.random.random() < ACTION_BUFFER_PROB:
            end = min(random_step_idx + ACTION_BUFFER_LENGTH, n_states)
            act_dim = actions.shape[1]
            self._action_buffer = [
                actions[i] + np.random.normal(0, ACTION_BUFFER_NOISE, size=act_dim)
                for i in range(random_step_idx, end)
            ]
        else:
            self._action_buffer = []

        print(
            f"\n[RSI] demo={chosen_seed} frame={random_step_idx}/{n_states} "
            f"({random_step_idx/n_states:.0%}) progress={self._demo_init_progress:.2f}/10 "
            f"action_buffer={len(self._action_buffer)}"
        )
        return states[random_step_idx]

    def _apply_init_noise(self):
        """Add Gaussian noise to arm joints and movable object positions on normal (non-RSI) resets.

        Arm: noise on qpos[0:7] (7-DOF Panda joints) in radians.
        Objects: XY noise on all movable objects (env.objects) — includes manipulated
        objects and movable goal containers. Fixed fixtures are unaffected.
        """
        sim = self.env.env.sim

        # Arm joint noise (7-DOF Panda arm)
        arm_noise = np.random.normal(0, INIT_ARM_NOISE_STD, size=7)
        sim.data.qpos[0:7] += arm_noise
        print(f"\n[NOISE] arm joints: {np.array2string(arm_noise, precision=6)}")

        # Object XY noise — look up each object's free-joint qpos address by name
        try:
            for obj in self.env.env.objects:
                joint_name = obj.joints[-1]
                addr = sim.model.get_joint_qpos_addr(joint_name)
                # addr is a tuple (start, end) for multi-dof joints
                start = addr[0] if isinstance(addr, tuple) else int(addr)
                dx = np.random.normal(0, INIT_OBJ_NOISE_STD)
                dy = np.random.normal(0, INIT_OBJ_NOISE_STD)
                sim.data.qpos[start]     += dx
                sim.data.qpos[start + 1] += dy
                print(f"[NOISE] {obj.name}: x={dx:+.6f}m y={dy:+.6f}m")
        except Exception:
            pass  # Skip object noise if object joints are not accessible

        sim.forward()

    def _get_policy_images(self):
        """Render 84×84 frames from agentview and wrist cameras for DRQ-v2.

        Returns:
            (agentview, wrist): both HxWx3 uint8 numpy arrays.
        """
        sim = self.env.env.sim
        agentview = sim.render(
            camera_name=AGENTVIEW_CAMERA_NAME,
            width=POLICY_IMG_SIZE,
            height=POLICY_IMG_SIZE,
        )[::-1]
        wrist = sim.render(
            camera_name=WRIST_CAMERA_NAME,
            width=POLICY_IMG_SIZE,
            height=POLICY_IMG_SIZE,
        )[::-1]
        return agentview, wrist

    def flatten_obs(self, obs):
        arrays = [
            v for k, v in obs.items()
            if not k.endswith("_image") and not k.endswith("_depth")
        ]
        return np.concatenate(arrays)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self, *, seed=None, options=None):
        """Reset the environment.

        RSI start: sets state from demo frame at sampled progress + action buffer. No noise.
        Normal start: sets state from demo frame 0 (anchors object positions to RM reference),
        then adds Gaussian noise on arm joints and objects for variety.
        """
        if seed is not None:
            self.seed(seed)
        self._step_count = 0
        self._action_buffer = []
        self._success_delay_counter = 0
        obs = self.env.reset()

        print(f"\n[RESET] demo_init_prob={self.demo_init_prob:.3f}")

        rsi_applied = False
        if self.use_demo_init and (self.is_eval_env or np.random.random() < self.demo_init_prob):
            state = self._get_demo_state_at_random_step()
            if state is not None:
                obs = self.env.set_init_state(state)
                rsi_applied = True

        if not rsi_applied:
            # Anchor to demo frame 0 so object positions match the RM reference,
            # then add noise for variety. Without this, LIBERO's placement sampler
            # places objects at random positions within the region (up to ±2.5cm),
            # which causes RM early-clip mismatches during eval.
            all_demos = self._load_demo_data()
            if all_demos is not None and len(all_demos) > 0:
                chosen_seed = self._demo_seed if (self._demo_seed is not None and self._demo_seed in all_demos) else 0
                states = all_demos[chosen_seed]['states']
                if len(states) > 0:
                    obs = self.env.set_init_state(states[0])
            self._apply_init_noise()

        return self.flatten_obs(obs), {}

    def step(self, action):
        """Step the environment.

        If an action buffer is loaded (RSI demo replay), overrides the agent's action
        with the next buffered demo action.

        info keys:
            success: bool
            demo_init_progress: float (0-10 scale)
            image: HxWx3 uint8, present every render_interval steps
            policy_images: (agentview_84, wrist_84), present every step
        """
        if self._action_buffer:
            action = self._action_buffer.pop(0)
            info_pre = {"executed_action": action}
        else:
            info_pre = {}

        obs, reward, done, info2 = self.env.step(action)
        success = self.env.check_success()
        robosuite_done = self.env.env.done
        if success and self._success_delay_counter == 0:
            self._success_delay_counter = SUCCESS_DELAY_STEPS + 1
        if self._success_delay_counter > 0:
            self._success_delay_counter -= 1
        terminated = bool(success and self._success_delay_counter == 0)
        truncated = bool(robosuite_done and not terminated)

        info = {
            "success": success,
            "demo_init_progress": self._demo_init_progress,
            **info_pre,
        }

        if self._policy_images_enabled:
            info["policy_images"] = self._get_policy_images()

        if self._step_count % self.render_interval == 0 and self._render_enabled:
            sim = self.env.env.sim
            info["image"] = sim.render(
                camera_name=AGENTVIEW_CAMERA_NAME,
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
            )[::-1]
            if self._wrist_render_enabled:
                info["wrist_image"] = sim.render(
                    camera_name=WRIST_CAMERA_NAME,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                )[::-1]

        self._step_count += 1
        return self.flatten_obs(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()
