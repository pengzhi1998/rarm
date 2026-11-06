"""DrQ-v2 training runner for LIBERO with optional image-based reward model.

Uses deferred episode rewards for ``reward_type=1``, an optional reward-decay
curriculum, demo-action overrides, and the original clip-to-step alignment.
The policy is an independently initialized DrQ-v2 agent in one LIBERO environment.

Image pipeline:
    * 84×84 RGB policy frame is produced every env step by ``LiberoEnvWrapper``
      and stashed in ``info["policy_images"]`` as (agentview, wrist) tuple.
    * 384×384 RGB frame is produced every ``RENDER_INTERVAL`` steps by the
      env and stashed in ``info["image"]`` (kept only for reward_type=1 RM
      post-processing — never stored in the replay buffer).
    * A 3-frame stack ([18, 84, 84] uint8) is written to the replay buffer.
      18 = 3 frames × 6 channels (agentview RGB + wrist RGB).

Checkpoints are saved as ``.pt`` under ``model_dir``.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from reward_model_baseline.Libero.agents.drqv2 import (
    DrQV2Agent,
    ReplayBufferStorage,
    make_replay_loader,
    utils,
)
from reward_model_baseline.Libero.constants.env_constants import (
    BINARY_REWARD_SCALER,
    DEMO_INIT_PROB,
    POLICY_IMG_SIZE,
    RENDER_INTERVAL,
    STEP_COST,
)


# ----------------------------------------------------------------------------
# dm_env-style helpers (minimal, kept local to avoid a dm_env dependency)
# ----------------------------------------------------------------------------

@dataclass
class _NamedArraySpec:
    name: str
    shape: tuple
    dtype: np.dtype


class _DictTimeStep(dict):
    """Dict with a ``last()`` accessor so ``ReplayBufferStorage.add`` works."""

    def __init__(self, *, is_last: bool, **fields) -> None:
        super().__init__(**fields)
        self._is_last = is_last

    def last(self) -> bool:
        return self._is_last


class FrameStack:
    """3-frame ring buffer producing ``[18, 84, 84]`` uint8 observations.

    Each frame is 6 channels: agentview RGB (3) + wrist RGB (3).
    """

    def __init__(self, num_frames: int = 3, frame_size: int = POLICY_IMG_SIZE) -> None:
        self.num_frames = num_frames
        self._stack = np.zeros(
            (num_frames * 6, frame_size, frame_size), dtype=np.uint8
        )

    def _concat(self, agentview_hwc: np.ndarray, wrist_hwc: np.ndarray) -> np.ndarray:
        """Convert two HxWx3 frames to a 6xHxW array."""
        a = np.transpose(agentview_hwc, (2, 0, 1)).astype(np.uint8, copy=False)
        w = np.transpose(wrist_hwc, (2, 0, 1)).astype(np.uint8, copy=False)
        return np.concatenate([a, w], axis=0)

    def reset(self, agentview_hwc: np.ndarray, wrist_hwc: np.ndarray) -> np.ndarray:
        frame = self._concat(agentview_hwc, wrist_hwc)
        for i in range(self.num_frames):
            self._stack[i * 6 : (i + 1) * 6] = frame
        return self._stack.copy()

    def push(self, agentview_hwc: np.ndarray, wrist_hwc: np.ndarray) -> np.ndarray:
        frame = self._concat(agentview_hwc, wrist_hwc)
        self._stack = np.roll(self._stack, shift=-6, axis=0)
        self._stack[-6:] = frame
        return self._stack.copy()


# ----------------------------------------------------------------------------
# Per-episode RARM reward computation
# ----------------------------------------------------------------------------

def _compute_episode_rewards(
    *,
    rm,
    images,
    num_steps: int,
    init_progress: float,
    success_flags,
    num_timesteps: int,
    total_timesteps: int,
    rm_ratio: float,
    rm_decay: bool,
    rm_verbose: bool,
    positives_dir: Optional[str],
    episode_id: int,
):
    """Run the RM on ``images`` and interpolate rewards onto all steps."""
    print()  # blank line between episodes
    chunk_imgs = [img for _, img in images]
    chunk_id = f"env0_ts{num_timesteps}"
    chunk_rewards, similarity_info = rm.compute_reward_from_images(
        chunk_imgs,
        save_videos=False,
        initial_progress=init_progress,
        return_similarity_info=True,
        save_chunk_frames=rm_verbose,
        chunk_id=chunk_id,
        verbose=rm_verbose,
        positives_dir=positives_dir,
        env_id=0,
        episode_id=episode_id,
    )
    del chunk_imgs

    # Logging to track training progress
    if similarity_info and similarity_info.get("max_similarities"):
        sims = similarity_info["max_similarities"]
        refs = similarity_info["best_ref_indices"]
        if sims:
            max_sim_idx = int(np.argmax(sims))
            best_ref = refs[max_sim_idx] if max_sim_idx < len(refs) else -1
            most_common = max(set(refs), key=refs.count) if refs else -1
            print(f"  Episode ({len(images)} imgs, {num_steps} steps):")
            print(
                f"    Max reward: {max(chunk_rewards):.4f}, "
                f"Avg(pos): {np.mean([r for r in chunk_rewards if r > 0] or [0]):.4f}, "
                f"Sum: {sum(chunk_rewards):.4f}"
            )
            print(
                f"    Max sim: {max(sims):.4f} (img {max_sim_idx}, ref {best_ref}), "
                f"Avg sim: {np.mean(sims):.4f}, Top ref: {most_common}"
            )

    if rm_decay:
        denominator = total_timesteps * rm_ratio
        binary_coef = num_timesteps / denominator if denominator > 0 else 1.0
        binary_coef = max(0.0, min(1.0, binary_coef))
        rm_coef = 1.0 - binary_coef
    else:
        rm_coef = 1.0

    half_span = (rm.model.num_frames - 1) // 2
    img_steps = [s for s, _ in images]
    step_reward_map = {}
    for reward_idx, reward_val in enumerate(chunk_rewards):
        middle_img_idx = max(0, reward_idx - half_span)
        step_reward_map[img_steps[middle_img_idx]] = reward_val

    if not step_reward_map:
        return None
    
    sorted_steps = sorted(step_reward_map.keys())
    sorted_rewards = [step_reward_map[s] for s in sorted_steps]
    sorted_steps_arr = np.array(sorted_steps)

    init_prog_normalized = init_progress / 10.0
    raw_image_progress = similarity_info.get("image_progress", [0.0] * len(chunk_rewards))

    # Attribute each progress jump to the middle image of the clip that caused it, then carry
    # forward. When clip ending at image i fires, the evidence spans images [i-clip_span+1, i];
    # the middle is i-half_span. This mirrors the reward attribution done above.
    # Initialised at RSI floor so pre-clip frames correctly inherit init_progress, not 0.
    half_span = (rm.model.num_frames - 1) // 2
    n = len(raw_image_progress)
    image_progress = [init_prog_normalized] * n
    for i in range(n):
        mid_idx = max(0, i - half_span)
        image_progress[mid_idx] = max(image_progress[mid_idx], raw_image_progress[i])
    for i in range(1, n):
        image_progress[i] = max(image_progress[i], image_progress[i - 1])

    img_steps_arr = np.array(img_steps)
    progress_per_step = []
    for local_step in range(num_steps):
        idx = max(0, int(np.searchsorted(img_steps_arr, local_step, side="right")) - 1)
        progress_per_step.append(np.array([float(image_progress[idx])], dtype=np.float32))

    if rm_verbose:
        pvals = [p[0] for p in progress_per_step]
        print(f"\n  [DEBUG progress] init_progress={init_progress:.1f} ({init_prog_normalized:.3f} normalised), half_span={half_span}")
        print(f"  [DEBUG progress] raw_image_progress={[round(v, 3) for v in raw_image_progress]}")
        print(f"  [DEBUG progress] image_progress (mid-shift+carry)={[round(v, 3) for v in image_progress]}")
        print(f"  [DEBUG progress] progress_per_step[0]={pvals[0]:.3f}  "
              f"max={max(pvals):.3f}  min={min(pvals):.3f}")
        for s in range(1, len(pvals)):
            if pvals[s] != pvals[s - 1]:
                print(f"  [DEBUG progress] first change at step {s}: {pvals[s-1]:.3f} -> {pvals[s]:.3f}")
                break
        floor_ok = all(v >= init_prog_normalized - 1e-6 for v in pvals)
        mono_ok = all(pvals[s] >= pvals[s - 1] - 1e-6 for s in range(1, len(pvals)))
        init_ok = abs(pvals[0] - init_prog_normalized) < 1e-6
        print(f"  [DEBUG progress] floor>={init_prog_normalized:.3f}: {floor_ok} | "
              f"monotone: {mono_ok} | step0==init: {init_ok}")
        if not (floor_ok and mono_ok and init_ok):
            print("  [DEBUG progress] *** INVARIANT VIOLATION ***")

    for i, r in enumerate(sorted_rewards):
        if r == 1:
            sorted_rewards[max(0, i - 2) : i] = [1.0] * min(2, i)
    rewards = []
    for local_step in range(num_steps):
        idx = max(
            0, int(np.searchsorted(sorted_steps_arr, local_step, side="right")) - 1
        )
        img_reward = float(sorted_rewards[idx])
        binary_reward = success_flags[local_step] * BINARY_REWARD_SCALER
        rewards.append(img_reward * rm_coef + binary_reward - STEP_COST)
    return rewards


# ----------------------------------------------------------------------------
# Checkpoint helpers — atomic write with one-version-back fallback.
# ----------------------------------------------------------------------------

CHECKPOINT_FILENAME = "checkpoint.pt"
CHECKPOINT_PREV_FILENAME = "checkpoint.pt.prev"


def _capture_rng_state(env, eval_env) -> dict:
    return {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "env_action_space": env.action_space.np_random.bit_generator.state,
        "eval_env_action_space": eval_env.action_space.np_random.bit_generator.state,
    }


def _restore_rng_state(state: dict, env, eval_env) -> None:
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    random.setstate(state["python"])
    env.action_space.np_random.bit_generator.state = state["env_action_space"]
    eval_env.action_space.np_random.bit_generator.state = state["eval_env_action_space"]


def _save_checkpoint(model_dir: str, state: dict) -> None:
    """Atomic write: tmp → fsync → rename. Previous checkpoint kept as .prev fallback."""
    final_path = os.path.join(model_dir, CHECKPOINT_FILENAME)
    prev_path = os.path.join(model_dir, CHECKPOINT_PREV_FILENAME)
    tmp_path = final_path + ".tmp"

    torch.save(state, tmp_path)
    with open(tmp_path, "rb") as f:
        os.fsync(f.fileno())

    # Rotate current → prev before overwriting, so a crash mid-rename still leaves one good file.
    if os.path.exists(final_path):
        os.replace(final_path, prev_path)
    os.replace(tmp_path, final_path)


def _load_checkpoint(model_dir: str) -> Optional[dict]:
    """Try checkpoint.pt, fall back to checkpoint.pt.prev. Returns None if neither readable."""
    for name in (CHECKPOINT_FILENAME, CHECKPOINT_PREV_FILENAME):
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            continue
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if name == CHECKPOINT_PREV_FILENAME:
                print(f"[resume] WARNING: loaded {CHECKPOINT_PREV_FILENAME} (current was unreadable)")
            else:
                print(f"[resume] loaded {path}")
            return state
        except Exception as e:
            print(f"[resume] failed to load {path}: {e}")
    return None


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def run(
    *,
    task_id: int,
    reward_type: int,
    algorithm: str = "drqv2",
    rm_ratio: float,
    total_timesteps: int,
    seed: int,
    use_demo_init: bool,
    use_eval_rsi: bool = False,
    use_demo_decay: bool = False,
    rm_decay: bool,
    rm_verbose: bool,
    log_dir: str,
    model_dir: str,
    demo_seed: Optional[int] = None,
    positives_dir: Optional[str] = None,
    # --- DrQ-v2 hyperparameters (progressor defaults) ---
    frame_stack: int = 3,
    num_seed_frames: int = 4000,
    update_every_steps: int = 2,
    nstep: int = 3,
    batch_size: int = 256,
    discount: float = 0.99,
    lr: float = 1e-4,
    feature_dim: int = 50,
    hidden_dim: int = 1024,
    critic_target_tau: float = 0.01,
    num_expl_steps: int = 2000,
    stddev_schedule: str = "linear(1.0,0.1,500000)",
    stddev_clip: float = 0.3,
    replay_buffer_size: int = 750_000,
    replay_buffer_num_workers: int = 4,
    fetch_every: int = 1000,
    save_snapshot: bool = True,
    save_freq: int = 50_000,
    checkpoint_freq: int = 25_000,
    log_every_steps: int = 1000,
    eval_freq: int = 20_000,
    n_eval_episodes: int = 10,
    device: str = "cuda",
    resume: bool = False,
    steps_this_link: int = 0,
    similarity_threshold_offset: float = 0.5,
    vit_checkpoint_path: str | None = None,
    comparator_checkpoint_path: str | None = None,
) -> None:
    if algorithm != "drqv2":
        raise ValueError("This release supports only algorithm=drqv2")
    utils.set_seed_everywhere(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from reward_model_baseline.Libero.envs.libero_env_wrapper import LiberoEnvWrapper

    # ------------- env -------------
    env = LiberoEnvWrapper(
        task_id=task_id,
        render_interval=RENDER_INTERVAL,
        use_demo_init=use_demo_init,
        demo_init_prob=DEMO_INIT_PROB,
        demo_seed=demo_seed,
    )
    env.set_policy_images_enabled(True)
    if reward_type in (0, 2):
        env.set_render_enabled(False)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    # ------------- eval env -------------
    eval_env = LiberoEnvWrapper(
        task_id=task_id,
        render_interval=RENDER_INTERVAL,
        use_demo_init=use_eval_rsi,
        demo_init_prob=DEMO_INIT_PROB if use_eval_rsi else 0.0,
        is_eval_env=use_eval_rsi,
        n_eval_episodes=n_eval_episodes,
        demo_seed=demo_seed,
    )
    eval_env.set_policy_images_enabled(True)
    eval_env.set_render_enabled(False)
    eval_env.reset(seed=seed)
    eval_env.action_space.seed(seed)

    action_dim = int(env.action_space.shape[0])
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    # ------------- reward model (lazy) -------------
    rm = None
    if reward_type == 1:
        from reward_model_baseline.Libero.RewardModels.BinaryProgressRewardModel import BinaryProgressRewardModel
        rm = BinaryProgressRewardModel.get_global_instance(
            task_id=task_id,
            demo_seed=demo_seed if demo_seed is not None else seed,
            vit_checkpoint_path=vit_checkpoint_path,
            comparator_checkpoint_path=comparator_checkpoint_path,
            similarity_threshold_offset=similarity_threshold_offset,
            device=device,
        )

    # ------------- replay buffer -------------
    obs_shape = (frame_stack * 6, POLICY_IMG_SIZE, POLICY_IMG_SIZE)
    data_specs = (
        _NamedArraySpec("observation", obs_shape, np.uint8),
        _NamedArraySpec("action", (action_dim,), np.float32),
        _NamedArraySpec("reward", (1,), np.float32),
        _NamedArraySpec("discount", (1,), np.float32),
    )
    replay_dir = Path(log_dir).parent / "replay"
    storage = ReplayBufferStorage(data_specs, replay_dir)
    replay_loader, _ = make_replay_loader(
        replay_dir,
        replay_buffer_size,
        batch_size,
        replay_buffer_num_workers,
        save_snapshot,
        nstep,
        discount,
        fetch_every=fetch_every,
    )
    replay_iter = None

    # ------------- agent -------------
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=(action_dim,),
        device=device,
        lr=lr,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        critic_target_tau=critic_target_tau,
        num_expl_steps=num_expl_steps,
        update_every_steps=update_every_steps,
        stddev_schedule=stddev_schedule,
        stddev_clip=stddev_clip,
        use_tb=True,
    )

    # ------------- logging -------------
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
    except ImportError:
        pass

    stacker = FrameStack(frame_stack)

    # ------------- episode helpers -------------

    def _new_episode():
        if use_demo_decay and use_demo_init and total_timesteps > 0:
            decayed_prob = max(0.0, 1.0 - num_timesteps / total_timesteps)
            env.set_demo_init_prob(decayed_prob)
            if writer is not None:
                writer.add_scalar("rollout/demo_init_prob", decayed_prob, num_timesteps)
        env.reset()
        agentview, wrist = env._get_policy_images()
        stack_local = stacker.reset(agentview, wrist)
        init_progress_local = float(env._demo_init_progress)
        storage.add(
            _DictTimeStep(
                observation=stack_local,
                action=np.zeros(action_dim, dtype=np.float32),
                reward=0.0,
                discount=1.0,
                is_last=False,
            )
        )
        return stack_local, init_progress_local

    def _flush_pending_rm(pending, pending_images, init_progress_local, episode_id):
        if not pending:
            return 0.0
        rewards = None
        if rm is not None and len(pending_images) < rm.model.num_frames:
            print(
                f"  [RM skipped] only {len(pending_images)} imgs < {rm.model.num_frames} needed; "
                f"falling back to binary reward"
            )
        if rm is not None and len(pending_images) >= rm.model.num_frames:
            rewards = _compute_episode_rewards(
                rm=rm,
                images=pending_images,
                num_steps=len(pending),
                init_progress=init_progress_local,
                success_flags=[t["success"] for t in pending],
                num_timesteps=num_timesteps,
                total_timesteps=total_timesteps,
                rm_ratio=rm_ratio,
                rm_decay=rm_decay,
                rm_verbose=rm_verbose,
                positives_dir=positives_dir,
                episode_id=episode_id,
            )
        total_r = 0.0
        for i, t in enumerate(pending):
            if rewards is not None:
                r = float(rewards[i])
            else:
                r = float(t["success"]) * BINARY_REWARD_SCALER - STEP_COST

            total_r += r
            storage.add(
                _DictTimeStep(
                    observation=t["stack"],
                    action=t["action"],
                    reward=r,
                    discount=t["discount"],
                    is_last=t["done"],
                )
            )
        return total_r

    # ------------- eval helper -------------

    def _evaluate(current_step: int) -> dict:
        was_training = agent.training
        agent.train(False)

        eval_stacker = FrameStack(frame_stack)
        ep_rewards, ep_lengths, ep_successes = [], [], []
        try:
            for _ in range(n_eval_episodes):
                eval_env.reset()
                agentview, wrist = eval_env._get_policy_images()
                eval_stack = eval_stacker.reset(agentview, wrist)

                ep_reward = 0.0
                ep_len = 0
                success = False
                done = False
                while not done:
                    with torch.no_grad():
                        action = agent.act(eval_stack, step=current_step, eval_mode=True)
                    action = np.clip(
                        np.asarray(action, dtype=np.float32), action_low, action_high
                    )
                    _, reward, terminate, truncate, info = eval_env.step(action)
                    ep_reward += float(reward)
                    ep_len += 1
                    if int(info.get("success", 0)):
                        success = True
                    done = bool(terminate or truncate)
                    if done:
                        break
                    agentview, wrist = info.get("policy_images", (None, None))
                    eval_stack = eval_stacker.push(agentview, wrist)
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_len)
                ep_successes.append(float(success))
        finally:
            agent.train(was_training)

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "success_rate": float(np.mean(ep_successes)),
            "mean_ep_length": float(np.mean(ep_lengths)),
        }

    # ------------- main loop -------------
    num_timesteps = 0
    episode_count = 0
    episode_env_reward_sum = 0.0
    episode_success_steps = 0
    total_success_steps = 0
    current_reward_type = reward_type
    pending_switch_to_binary = False

    last_save = 0
    last_eval = 0
    last_checkpoint = 0
    best_success_rate = -1.0
    next_log = log_every_steps

    # --- resume from checkpoint (caller has already verified existence in task_training.py) ---
    if resume:
        ckpt = _load_checkpoint(model_dir)
        if ckpt is None:
            raise RuntimeError(
                f"--resume requested but checkpoint at {model_dir} could not be loaded"
            )
        agent.load_state_dict(ckpt["agent"])
        num_timesteps = ckpt["num_timesteps"]
        episode_count = ckpt["episode_count"]
        total_success_steps = ckpt["total_success_steps"]
        last_save = ckpt["last_save"]
        last_eval = ckpt["last_eval"]
        last_checkpoint = ckpt["last_checkpoint"]
        best_success_rate = ckpt["best_success_rate"]
        next_log = ckpt["next_log"]
        _restore_rng_state(ckpt["rng"], env, eval_env)
        # Demo-init prob is a deterministic function of num_timesteps (see _new_episode), so it
        # gets recomputed correctly on the next episode reset — nothing to restore explicitly.
        print(
            f"[resume] continuing from step {num_timesteps} "
            f"(episode {episode_count}, best success rate {best_success_rate:.2f})"
        )

    stack, init_progress = _new_episode()
    pending = []
    pending_images = []


    last_log_time = time.time()

    # Per-link budget: number of NEW env steps this process is allowed to train before clean-exit.
    # steps_this_link=0 means no per-link cap (train to total_timesteps in one process).
    link_step_start = num_timesteps
    link_step_limit = num_timesteps + steps_this_link if steps_this_link > 0 else total_timesteps

    print(
        f"Starting DrQ-v2 training | task_id={task_id} "
        f"reward_type={reward_type} total_timesteps={total_timesteps} "
        f"seed={seed} use_demo_init={use_demo_init} rm_decay={rm_decay} "
        f"eval_freq={eval_freq} n_eval_episodes={n_eval_episodes} "
        f"resume={resume} link_step_limit={link_step_limit}"
    )

    try:
      while num_timesteps < total_timesteps:
        # --- curriculum switch trigger (applied at next episode boundary) ---
        if (
            rm_decay
            and current_reward_type == 1
            and not pending_switch_to_binary
            and total_timesteps > 0
            and num_timesteps / total_timesteps >= rm_ratio
        ):
            pending_switch_to_binary = True

        # --- action selection ---
        if num_timesteps < num_seed_frames:
            action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        else:
            with torch.no_grad(), utils.eval_mode(agent.encoder, agent.actor):
                action = agent.act(stack, step=num_timesteps, eval_mode=False)
            action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, action_low, action_high)

        _, reward, terminate, truncate, info = env.step(action)
        num_timesteps += 1

        # Use expert action if wrapper overrode with buffered demo action
        action = np.clip(
            np.asarray(info.pop("executed_action", action), dtype=np.float32),
            action_low, action_high,
        )

        agentview, wrist = info.pop("policy_images", (None, None))
        stack = stacker.push(agentview, wrist)

        success = int(info.get("success", 0))
        total_success_steps += success
        episode_success_steps += success
        episode_env_reward_sum += float(reward)

        done = bool(terminate or truncate)
        env_discount = 0.0 if terminate else 1.0

        if current_reward_type == 1:
            if "image" in info:
                step_idx = len(pending)
                pending_images.append((step_idx, info["image"]))
        info.pop("image", None)
        info.pop("wrist_image", None)

        # --- write to replay buffer ---
        if current_reward_type == 0:
            r = float(reward) - STEP_COST
            storage.add(
                _DictTimeStep(
                    observation=stack,
                    action=action,
                    reward=r,
                    discount=env_discount,
                    is_last=done,
                )
            )
        elif current_reward_type == 2:
            r = float(success) * BINARY_REWARD_SCALER - STEP_COST
            storage.add(
                _DictTimeStep(
                    observation=stack,
                    action=action,
                    reward=r,
                    discount=env_discount,
                    is_last=done,
                )
            )
        else:
            pending.append(
                {
                    "stack": stack.copy(),
                    "action": action.copy(),
                    "success": float(success),
                    "discount": env_discount,
                    "done": done,
                }
            )

        # --- episode end ---
        if done:
            rm_reward_sum = 0.0
            if current_reward_type == 1:
                rm_reward_sum = _flush_pending_rm(
                    pending, pending_images, init_progress, episode_count
                )
                pending = []
                pending_images = []


            episode_count += 1
            print(
                f"[ep {episode_count}] steps={num_timesteps} "
                f"ep_env_rew={episode_env_reward_sum:.2f} "
                f"ep_success_steps={episode_success_steps} "
                f"total_success_steps={total_success_steps}"
                + (f" rm_ep_rew={rm_reward_sum:.2f}" if current_reward_type == 1 else "")
            )
            if writer is not None:
                writer.add_scalar("rollout/ep_env_reward", episode_env_reward_sum, num_timesteps)
                writer.add_scalar("rollout/ep_success_steps", episode_success_steps, num_timesteps)
                writer.add_scalar("rollout/total_success_steps", total_success_steps, num_timesteps)
                if current_reward_type == 1:
                    writer.add_scalar("rollout/rm_ep_reward", rm_reward_sum, num_timesteps)

            episode_env_reward_sum = 0.0
            episode_success_steps = 0

            if pending_switch_to_binary:
                print(f"[curriculum] switching reward_type 1 -> 2 at step {num_timesteps}")
                current_reward_type = 2
                pending_switch_to_binary = False
                env.set_render_enabled(False)

            # --- checkpoint at episode boundary (cadence: checkpoint_freq env steps) ---
            if num_timesteps - last_checkpoint >= checkpoint_freq:
                ckpt_state = {
                    "agent": agent.state_dict(),
                    "num_timesteps": num_timesteps,
                    "episode_count": episode_count,
                    "total_success_steps": total_success_steps,
                    "last_save": last_save,
                    "last_eval": last_eval,
                    "last_checkpoint": num_timesteps,
                    "best_success_rate": best_success_rate,
                    "next_log": next_log,
                    "rng": _capture_rng_state(env, eval_env),
                }
                _save_checkpoint(model_dir, ckpt_state)
                last_checkpoint = num_timesteps
                print(f"[checkpoint] saved at step {num_timesteps}")

            # --- per-link budget: clean-exit when this process has trained its share ---
            if steps_this_link > 0 and num_timesteps >= link_step_limit:
                trained_this_link = num_timesteps - link_step_start
                print(
                    f"[link budget] reached {trained_this_link} steps this link "
                    f"(limit {steps_this_link}); clean-exit at step {num_timesteps}. "
                    f"Resubmit with --resume to continue."
                )
                # Force a final checkpoint write so the next link picks up exactly here.
                ckpt_state = {
                    "agent": agent.state_dict(),
                    "num_timesteps": num_timesteps,
                    "episode_count": episode_count,
                    "total_success_steps": total_success_steps,
                    "last_save": last_save,
                    "last_eval": last_eval,
                    "last_checkpoint": num_timesteps,
                    "best_success_rate": best_success_rate,
                    "next_log": next_log,
                    "rng": _capture_rng_state(env, eval_env),
                }
                _save_checkpoint(model_dir, ckpt_state)
                return

            stack, init_progress = _new_episode()

        # --- agent update ---
        if (
            num_timesteps >= num_seed_frames
            and storage._num_episodes > 0
            and num_timesteps % update_every_steps == 0
        ):
            if replay_iter is None:
                replay_iter = iter(replay_loader)
            batch = next(replay_iter)
            metrics = agent.update(batch, step=num_timesteps)
            if writer is not None and num_timesteps >= next_log:
                for k, v in metrics.items():
                    writer.add_scalar(f"train/{k}", float(v), num_timesteps)
                now = time.time()
                sps = log_every_steps / max(1e-6, now - last_log_time)
                writer.add_scalar("train/steps_per_sec", sps, num_timesteps)
                last_log_time = now
                next_log = num_timesteps + log_every_steps

        # --- checkpointing ---
        if num_timesteps - last_save >= save_freq:
            ckpt_path = os.path.join(model_dir, f"drqv2_step_{num_timesteps}.pt")
            torch.save({"agent": agent.state_dict(), "step": num_timesteps}, ckpt_path)
            last_save = num_timesteps

        # --- evaluation ---
        if num_timesteps - last_eval >= eval_freq:
            last_eval = num_timesteps
            print(f"\n[eval @ step {num_timesteps}] running {n_eval_episodes} episodes...")
            eval_metrics = _evaluate(num_timesteps)
            print(
                f"  mean_reward={eval_metrics['mean_reward']:.2f} "
                f"success_rate={eval_metrics['success_rate']:.2f} "
                f"mean_ep_length={eval_metrics['mean_ep_length']:.1f}\n"
            )
            if writer is not None:
                writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], num_timesteps)
                writer.add_scalar("eval/success_rate", eval_metrics["success_rate"], num_timesteps)
                writer.add_scalar("eval/mean_ep_length", eval_metrics["mean_ep_length"], num_timesteps)
            if eval_metrics["success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]
                best_path = os.path.join(model_dir, "drqv2_best.pt")
                torch.save(
                    {
                        "agent": agent.state_dict(),
                        "step": num_timesteps,
                        "success_rate": best_success_rate,
                    },
                    best_path,
                )
                print(f"  new best success_rate={best_success_rate:.2f} -> saved {best_path}")

      # Final checkpoint
      final_path = os.path.join(model_dir, "drqv2_final.pt")
      torch.save({"agent": agent.state_dict(), "step": num_timesteps}, final_path)
      print(f"Training complete. Final checkpoint: {final_path}")
    finally:
        if writer is not None:
            writer.close()
        try:
            env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass
        # Note: replay_dir is INTENTIONALLY not removed — it persists across resubmissions so
        # the next --resume link can rebuild the buffer cache from the .npz files on disk.
