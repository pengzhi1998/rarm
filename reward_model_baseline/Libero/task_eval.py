import os

# Configure MuJoCo rendering backend for optimal performance
# Set these BEFORE importing any MuJoCo-related packages
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_GL_PRELOAD', '1')

import argparse
import glob
import numpy as np
import imageio
import torch
from PIL import Image

from reward_model_baseline.Libero.agents.drqv2.utils import set_seed_everywhere

from reward_model_baseline.Libero.constants.env_constants import (
    CAMERA_HEIGHT, CAMERA_WIDTH, AGENTVIEW_CAMERA_NAME, SEED, RENDER_INTERVAL, POLICY_IMG_SIZE,
)
from reward_model_baseline.Libero.constants.libero_task_mapping import (
    TASK_ID_TO_ENV_NAME,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate trained Libero models")
    parser.add_argument("--task_id", type=int, choices=sorted(TASK_ID_TO_ENV_NAME), default=0, help="Task ID for Libero environment")
    parser.add_argument("--experiment_run", type=int, default=0, help="Experiment version number")
    parser.add_argument("--training_run", type=str, default=None,
                        help="Training run subfolder name (e.g., 'rt_1_rm_ratio_1.0_seed_1_20260316_150936'). None for latest.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (also selects demo trajectory for RM)")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable RM reward computation with per-frame similarity output, and save frames to disk")
    parser.add_argument("--use_demo_init", action="store_true",
                        help="Use demo-based random state initialization (RSI) for evaluation")
    parser.add_argument("--demo_init_prob", type=float, default=0.7,
                        help="Probability of using demo init per episode (default: 0.7)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Specific model filename to load (e.g., 'drqv2_best.pt'). None for auto-detect.")
    parser.add_argument("--vit_checkpoint_path", default=None, help="DINOv3 checkpoint (default: checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth)")
    parser.add_argument("--comparator_checkpoint_path", default=None, help="RARM checkpoint (default: checkpoints/checkpoint_epoch_45_generic.pt)")
    parser.add_argument("--device", default=None, help="PyTorch device; default: CUDA if available, otherwise CPU")
    args = parser.parse_args(argv)
    if args.num_episodes <= 0:
        parser.error("num_episodes must be positive")
    return args


def find_model_path(model_dir, model_name=None):
    """Find best available model file in model_dir.

    Prefer best, then final, then latest step checkpoint.
    """
    if model_name is not None:
        path = os.path.join(model_dir, model_name)
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_dir}")

    # DrQ-v2 candidates (.pt). For step checkpoints, pick the latest by step.
    drq_candidates = ["drqv2_best.pt", "drqv2_final.pt"]
    for name in drq_candidates:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    step_paths = sorted(
        glob.glob(os.path.join(model_dir, "drqv2_step_*.pt")),
        key=lambda p: int(os.path.basename(p).split("_")[-1].split(".")[0]),
    )
    if step_paths:
        return step_paths[-1]

    raise FileNotFoundError(f"No DrQ-v2 policy checkpoint found in {model_dir}")


def load_drqv2_agent(model_path, env, device="cuda"):
    """Build a DrQV2Agent and load its weights from a .pt checkpoint."""
    from reward_model_baseline.Libero.agents.drqv2 import DrQV2Agent

    action_dim = int(env.action_space.shape[0])
    frame_stack = 3
    obs_shape = (frame_stack * 6, POLICY_IMG_SIZE, POLICY_IMG_SIZE)  # dual-camera: 18ch
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    use_progress = ckpt["agent"].get("use_progress", False)
    agent = DrQV2Agent(
        obs_shape=obs_shape,
        action_shape=(action_dim,),
        device=device,
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule="linear(1.0,0.1,500000)",
        stddev_clip=0.3,
        use_tb=False,
        use_progress=use_progress,
    )
    agent.load_state_dict(ckpt["agent"])
    agent.train(False)
    return agent, frame_stack


def main():
    args = parse_args()
    from reward_model_baseline.Libero.envs.libero_env_wrapper import LiberoEnvWrapper

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed_everywhere(args.seed)

    task_id = args.task_id
    task_name = TASK_ID_TO_ENV_NAME[task_id]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    output_base = os.path.join(project_root, "output")

    experiment_dir = os.path.join(output_base, "Libero", f"{task_id}_{task_name}_{args.experiment_run}")
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    if args.training_run is None:
        training_run_dirs = sorted(glob.glob(os.path.join(experiment_dir, "rt_*")))
        if not training_run_dirs:
            raise FileNotFoundError(f"No training runs found in: {experiment_dir}")
        task_dir = training_run_dirs[-1]
    else:
        task_dir = os.path.join(experiment_dir, args.training_run)
        if not os.path.exists(task_dir):
            raise FileNotFoundError(f"Training run not found: {task_dir}")

    model_dir = os.path.join(task_dir, "models")
    model_path = find_model_path(model_dir, args.model_name)

    experiment_name = f"{task_id}_{task_name}_{args.experiment_run}"
    training_run_name = os.path.basename(task_dir)
    eval_output_dir = os.path.join(project_root, "temp", f"eval_{experiment_name}_{training_run_name}")
    os.makedirs(eval_output_dir, exist_ok=True)

    algo_name = "DrQ-v2"

    print(f"Task: {task_name} (ID: {task_id})")
    print(f"Algorithm: {algo_name}")
    print(f"Model: {model_path}")
    print(f"Training run: {task_dir}")
    print(f"Seed: {args.seed}")
    print(f"Verbose: {args.verbose}")
    print(f"Demo init: {args.use_demo_init} (prob: {args.demo_init_prob})")

    rm = None
    if args.verbose:
        from reward_model_baseline.Libero.RewardModels.BinaryProgressRewardModel import (
            BinaryProgressRewardModel,
        )
        rm = BinaryProgressRewardModel.get_global_instance(
            task_id=task_id, demo_seed=args.seed,
            vit_checkpoint_path=args.vit_checkpoint_path,
            comparator_checkpoint_path=args.comparator_checkpoint_path,
            device=device,
        )
        print(f"Reward model loaded (demo seed: {args.seed})")

    env = LiberoEnvWrapper(
        task_id=task_id,
        render_interval=RENDER_INTERVAL,
        use_demo_init=args.use_demo_init,
        demo_init_prob=args.demo_init_prob,
        is_eval_env=args.use_demo_init,
        n_eval_episodes=args.num_episodes,
        demo_seed=args.seed,
    )
    env.seed(args.seed)

    env.set_policy_images_enabled(True)
    if not args.verbose:
        env.set_render_enabled(False)

    if args.use_demo_init:
        print(f"RSI schedule: {[f'{i/(2*args.num_episodes):.0%}' for i in range(args.num_episodes)]}")

    drq_agent = None
    drq_frame_stack = None
    drq_agent, drq_frame_stack = load_drqv2_agent(model_path, env, device=device)

    print(f"\nRunning {args.num_episodes} evaluation episodes...\n")

    from reward_model_baseline.Libero.drqv2_runner import FrameStack
    stacker = FrameStack(num_frames=drq_frame_stack)
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    successes = []
    for episode in range(args.num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        done = False
        step = 0
        episode_images = []

        demo_init_progress = 0.0
        success = False

        agentview, wrist = env._get_policy_images()
        stack = stacker.reset(agentview, wrist)

        while not done:
            with torch.no_grad():
                action = drq_agent.act(stack, step=step, eval_mode=True)
            action = np.clip(
                np.asarray(action, dtype=np.float32),
                action_low,
                action_high,
            )

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated
            if info.get("success", False):
                success = True
                done = True

            if args.verbose and "image" in info:
                episode_images.append((step, info["image"]))
            if "demo_init_progress" in info:
                demo_init_progress = info["demo_init_progress"]

            if not done:
                policy_images = info.get("policy_images")
                if policy_images is not None:
                    agentview, wrist = policy_images
                else:
                    agentview, wrist = env._get_policy_images()
                stack = stacker.push(agentview, wrist)

        successes.append(float(success))
        init_info = f", RSI progress: {demo_init_progress:.1f}" if demo_init_progress > 0 else ""
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}, Success = {success}{init_info}")

        if args.verbose and rm is not None and episode_images:
            episode_frame_dir = os.path.join(eval_output_dir, f"episode_{episode + 1}")
            os.makedirs(episode_frame_dir, exist_ok=True)
            frames_uint8 = []
            for frame_idx, (_, img) in enumerate(episode_images):
                frame_path = os.path.join(episode_frame_dir, f"frame_{frame_idx:06d}.jpg")
                if img.dtype != np.uint8:
                    img_save = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                else:
                    img_save = img
                Image.fromarray(img_save).save(frame_path)
                frames_uint8.append(img_save)
            print(f"  Saved {len(episode_images)} frames to {episode_frame_dir}")

            video_path = os.path.join(eval_output_dir, f"episode_{episode + 1}.gif")
            imageio.mimwrite(video_path, frames_uint8, fps=20)
            print(f"  Saved video to {video_path}")

            imgs = [img for _, img in episode_images]
            if len(imgs) >= rm.model.num_frames:
                print()
                chunk_id = f"eval_ep{episode + 1}"
                rewards, similarity_info = rm.compute_reward_from_images(
                    imgs,
                    save_videos=False,
                    initial_progress=demo_init_progress,
                    return_similarity_info=True,
                    save_chunk_frames=False,
                    chunk_id=chunk_id,
                    verbose=True,
                )

                if similarity_info and similarity_info.get('max_similarities'):
                    max_sims = similarity_info['max_similarities']
                    ref_indices = similarity_info['best_ref_indices']
                    if max_sims:
                        max_sim = np.max(max_sims)
                        max_sim_idx = int(np.argmax(max_sims))
                        best_ref_at_max = ref_indices[max_sim_idx] if max_sim_idx < len(ref_indices) else -1
                        most_common_ref = max(set(ref_indices), key=ref_indices.count) if ref_indices else -1

                        print(f"\n  Eval Episode {episode + 1} ({len(imgs)} images, {step} steps):")
                        print(f"    Max reward: {max(rewards):.4f}, "
                              f"Avg(pos): {np.mean([r for r in rewards if r > 0] or [0]):.4f}, "
                              f"Sum: {sum(rewards):.4f}")
                        print(f"    Max similarity: {max_sim:.4f} (at img {max_sim_idx}, ref {best_ref_at_max})")
                        print(f"    Avg similarity: {np.mean(max_sims):.4f}, Most matched ref clip: {most_common_ref}")
                        print(f"    Success: {success}")
            else:
                print(f"  Not enough frames for RM ({len(imgs)} < {rm.model.num_frames})")

            print()

    env.close()

    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {args.num_episodes} episodes")
    print(f"  Success rate: {np.mean(successes):.0%} ({int(sum(successes))}/{args.num_episodes})")
    print(f"  Output: {eval_output_dir}")
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
