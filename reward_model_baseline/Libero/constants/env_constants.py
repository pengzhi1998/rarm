import os

# Configure MuJoCo rendering backend for optimal performance
# Set these BEFORE importing any MuJoCo-related packages
# Does this break running on multiple GPUs on Clariden?
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('MUJOCO_GL_PRELOAD', '1')

CAMERA_HEIGHT = 384
CAMERA_WIDTH  = 384
AGENTVIEW_CAMERA_NAME = "agentview"
WRIST_CAMERA_NAME     = "robot0_eye_in_hand"
RENDER_INTERVAL  = 4
POLICY_IMG_SIZE  = 84
BINARY_REWARD_SCALER = 5
SEED = 1
HAND_INIT_NOISE = 0
INIT_ARM_NOISE_STD = 0.0005   # radians (joint-angle noise, ~= metres EEF perturbation)
INIT_OBJ_NOISE_STD = 0.0005   # metres, XY only

SUCCESS_DELAY_STEPS  = 1
STEP_COST            = 0.5
ACTION_BUFFER_LENGTH = 70
ACTION_BUFFER_NOISE  = 0.01
ACTION_BUFFER_PROB   = 0.5
DEMO_INIT_START_RATIO = 0.01
DEMO_INIT_END_RATIO   = 0.95
DEMO_INIT_PROB        = 0.95
