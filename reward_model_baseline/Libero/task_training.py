"""Train an unpretrained DrQ-v2 policy on LIBERO using RARM rewards."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_GL_PRELOAD", "1")
# Required by deterministic CUDA matrix multiplication on supported CUDA versions.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from datetime import datetime
from pathlib import Path

from reward_model_baseline.Libero.constants.env_constants import SEED
from reward_model_baseline.Libero.constants.libero_task_mapping import TASK_ID_TO_ENV_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task_id", type=int, choices=sorted(TASK_ID_TO_ENV_NAME), default=0)
    parser.add_argument("--algorithm", choices=["drqv2"], default="drqv2")
    parser.add_argument("--reward_type", type=int, choices=[0, 1, 2], default=1,
                        help="0: environment reward; 1: RARM progress + success; 2: success only")
    parser.add_argument("--reward_ratio", type=float, default=0.3,
                        help="Fraction of total training over which --rm_decay fades RARM")
    parser.add_argument("--use_demo_init", action="store_true", help="Use reference state initialization (RSI)")
    parser.add_argument("--use_demo_decay", action="store_true", help="Decay RSI probability from 1 to 0")
    parser.add_argument("--use_eval_rsi", action="store_true", help="Use scheduled RSI in evaluation")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=SEED, help="RNG seed and reference demonstration index")
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--rm_decay", action="store_true", help="Fade RARM to success-only rewards")
    parser.add_argument("--verbose", action="store_true", help="Log detailed RM similarities and save frames")
    parser.add_argument("--save_similar_clips", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoint.pt in the deterministic run directory")
    parser.add_argument("--steps_this_link", type=int, default=0,
                        help="Stop at an episode boundary after this many new steps; 0 means no cap")
    parser.add_argument("--similarity_threshold_offset", type=float, default=0.5)
    parser.add_argument("--vit_checkpoint_path", default=str(REPO_ROOT / "checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"))
    parser.add_argument("--comparator_checkpoint_path", default=str(REPO_ROOT / "checkpoints/checkpoint_epoch_45_generic.pt"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.total_timesteps <= 0 or args.steps_this_link < 0:
        parser.error("total_timesteps must be positive and steps_this_link nonnegative")
    if args.similarity_threshold_offset <= 0:
        parser.error("similarity_threshold_offset must be positive")
    if args.rm_decay and not 0 < args.reward_ratio <= 1:
        parser.error("reward_ratio must be in (0, 1] when rm_decay is enabled")
    return args


def get_dir(args):
    task_name = TASK_ID_TO_ENV_NAME[args.task_id]
    base = REPO_ROOT / "output" / "Libero" / f"{args.task_id}_{task_name}_{args.version}"
    name = f"rt_{args.reward_type}_rm_ratio_{args.reward_ratio}_seed_{args.seed}"
    if not (args.resume or args.steps_this_link > 0):
        name += f"_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = base / f"{name}_{args.algorithm}"
    model_dir, log_dir = run_dir / "models", run_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, model_dir


def main(argv=None):
    args = parse_args(argv)
    from reward_model_baseline.Libero.drqv2_runner import run

    if args.reward_type == 1:
        for name in ("vit_checkpoint_path", "comparator_checkpoint_path"):
            path = Path(getattr(args, name)).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name}: {path}. See README checkpoint preparation.")
            setattr(args, name, str(path))
    log_dir, model_dir = get_dir(args)
    exists = any((model_dir / name).is_file() for name in ("checkpoint.pt", "checkpoint.pt.prev"))
    if args.resume and not exists:
        raise SystemExit(f"--resume requested but checkpoint.pt is missing in {model_dir}")
    if not args.resume and exists:
        raise SystemExit(f"A checkpoint already exists in {model_dir}; use --resume or a new --version")
    positives_dir = None
    if args.save_similar_clips:
        positives_dir = log_dir.parent / "positives"
        positives_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {log_dir.parent}")
    print(f"Task: {TASK_ID_TO_ENV_NAME[args.task_id]}")
    for name, value in vars(args).items():
        print(f"  {name}: {value}")
    run(task_id=args.task_id, reward_type=args.reward_type, algorithm=args.algorithm,
        rm_ratio=args.reward_ratio, total_timesteps=args.total_timesteps, seed=args.seed,
        use_demo_init=args.use_demo_init, use_eval_rsi=args.use_eval_rsi,
        use_demo_decay=args.use_demo_decay, rm_decay=args.rm_decay, rm_verbose=args.verbose,
        log_dir=str(log_dir), model_dir=str(model_dir), demo_seed=args.seed,
        positives_dir=str(positives_dir) if positives_dir else None,
        resume=args.resume, steps_this_link=args.steps_this_link,
        similarity_threshold_offset=args.similarity_threshold_offset,
        vit_checkpoint_path=args.vit_checkpoint_path,
        comparator_checkpoint_path=args.comparator_checkpoint_path, device=args.device)


if __name__ == "__main__":
    main()
