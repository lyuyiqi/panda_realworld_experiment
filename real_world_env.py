#!/usr/bin/env python3
"""
Standalone environment for real Franka Panda robot control with OpenPi policy inference.
This module provides a clean interface for robot control, camera capture,
observation collection, and policy inference.
"""

import numpy as np
import os
import sys
import cv2
from typing import Optional, List, Dict

# Setup paths for imports
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
panda_directory = os.path.join(parent_directory, 'panda')
if panda_directory not in sys.path:
    sys.path.append(panda_directory)

from panda_controller import PandaController
from realsense_camera import RealSenseCamera, reset_all_cameras

# OpenPi policy imports
from openpi.training import config as openpi_config
from openpi.policies import policy_config as openpi_policy_config
from openpi.shared import download as openpi_download

# Default configuration
DEFAULT_CONTROL_FREQUENCY = 20.0  # Hz

# Default policy configuration
DEFAULT_POLICY_CONFIG = "pi05_move_corn_pot_finetune"
DEFAULT_CHECKPOINT = "./checkpoints/pi05_move_corn_pot_finetune/my_experiment/19999"

# Default task instruction (must match the training data)
# From convert_move_corn_pot_data_to_lerobot.py
DEFAULT_TASK_INSTRUCTION = "Lift the yellow corn, then place it next to the kitchen pot with metallic round lid"

# Home position in Cartesian space (x, y, z) in meters
HOME_POSITION = np.array([0.3, 0.0, 0.28], dtype=np.float64)

# Camera resolution matching RoboTwin simulator
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Gripper width range (meters)
GRIPPER_MAX_WIDTH = 0.08  # fully open
GRIPPER_MIN_WIDTH = 0.0   # fully closed

# Action normalization statistics from norm_stats.json
# These are used to denormalize the model's output actions
# Actions format: [joint_velocity (7D), gripper_position (1D)]
ACTIONS_MEAN = np.array([
    -0.001297646202147007,    # joint_velocity[0]
    0.003484223037958145,     # joint_velocity[1]
    -0.001084090443328023,    # joint_velocity[2]
    0.00264749932102859,      # joint_velocity[3]
    -0.00011337778414599597,  # joint_velocity[4]
    0.001726333750411868,     # joint_velocity[5]
    -0.001657692831940949,    # joint_velocity[6]
    0.49949410557746887       # gripper_position (normalized, 1=closed, 0=open)
], dtype=np.float32)

ACTIONS_STD = np.array([
    0.007537253201007843,     # joint_velocity[0]
    0.03818218410015106,      # joint_velocity[1]
    0.007921707816421986,     # joint_velocity[2]
    0.01760072447359562,      # joint_velocity[3]
    0.0038679803255945444,    # joint_velocity[4]
    0.02735305391252041,      # joint_velocity[5]
    0.009863569401204586,     # joint_velocity[6]
    0.4866367280483246        # gripper_position
], dtype=np.float32)


class RealWorldEnv:
    """
    A standalone environment class for controlling a real Franka Panda robot.
    
    This class provides:
    - Robot control via high-level joint position commands
    - Camera image capture from multiple RealSense cameras
    - Gripper control
    - Observation collection in a standard format
    
    Observation format:
        - images: List of 3 RGB images [head_camera, right_camera, left_camera]
                  each with shape (H, W, 3), e.g., (240, 320, 3)
        - state: 8D numpy array [7 joints + 1 gripper width in meters]
    
    Action format:
        - action: numpy array of shape (8,) - [7 joint velocities, 1 gripper position]
    """
    
    def __init__(
        self,
        # Policy config
        policy_config_name: str = None,
        checkpoint_path: str = None,
        use_policy: bool = True,
        # Robot config
        control_frequency: float = None,
        use_robot: bool = True,
        # Camera config - serial numbers for the 3 cameras
        head_camera_serial: str = None,
        right_camera_serial: str = None,
        left_camera_serial: str = None,
    ):
        """
        Initialize the real world environment.
        
        Args:
            policy_config_name: Name of the OpenPi policy config (e.g., "pi0_fast_droid", "pi05_droid")
            checkpoint_path: Path or URL to the policy checkpoint (e.g., "s3://openpi-assets/checkpoints/pi0_fast_droid")
            use_policy: If True, load and use the policy for inference.
            control_frequency: Robot control frequency in Hz (default: DEFAULT_CONTROL_FREQUENCY)
            use_robot: If True, connect to real robot. If False, camera-only mode.
            head_camera_serial: Serial number for head camera (if None, auto-detect)
            right_camera_serial: Serial number for right wrist camera (if None, auto-detect)
            left_camera_serial: Serial number for left wrist camera (if None, auto-detect)
        """
        # Policy config
        self.policy_config_name = policy_config_name or DEFAULT_POLICY_CONFIG
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.use_policy = use_policy
        
        # Robot config
        self.control_frequency = control_frequency or DEFAULT_CONTROL_FREQUENCY
        self.control_period = 1.0 / self.control_frequency
        self.use_robot = use_robot
        
        # Camera serial numbers
        self.head_camera_serial = head_camera_serial
        self.right_camera_serial = right_camera_serial
        self.left_camera_serial = left_camera_serial
        
        # Initialize policy if enabled
        self.policy = None
        if self.use_policy:
            self._init_policy()
        
        # Initialize robot and cameras if enabled
        self.robot = None
        self.cameras = []  # List of [head, right, left] cameras
        self._gripper_available = False
        self._gripper_closed = False  # Track gripper state for boolean control
        
        if self.use_robot:
            self._init_robot()
            self._init_cameras()
    
    def _init_policy(self):
        """Initialize the OpenPi policy for inference."""
        print(f"\nInitializing policy...")
        print(f"  Config: {self.policy_config_name}")
        print(f"  Checkpoint: {self.checkpoint_path}")
        
        # Get the training config
        config = openpi_config.get_config(self.policy_config_name)
        
        # Download checkpoint if needed (handles s3://, gs://, or local paths)
        checkpoint_dir = openpi_download.maybe_download(self.checkpoint_path)
        
        # Create the trained policy
        self.policy = openpi_policy_config.create_trained_policy(config, checkpoint_dir)
        print("Policy initialized successfully!")
    
    def _init_robot(self):
        """Initialize robot connection and move to home position, with error recovery."""
        print(f"\nInitializing robot...")
        self.robot = PandaController()
        try:
            self.robot.connect()
        except RuntimeError as e:
            msg = str(e)
            if "manual error recovery required" in msg:
                print("Manual error recovery required! Attempting manual recovery...")
                try:
                    if hasattr(self.robot, "manual_error_recovery"):
                        self.robot.manual_error_recovery()
                    elif hasattr(self.robot._panda, "manual_error_recovery"):
                        self.robot._panda.manual_error_recovery()
                    else:
                        print("No manual_error_recovery method found on PandaController. Please recover manually.")
                    print("Manual error recovery attempted. Retrying connect...")
                    self.robot.connect()
                except Exception as e2:
                    print(f"Manual error recovery failed: {e2}")
                    raise
            else:
                raise
        
        # Connect gripper (optional, continue if fails)
        try:
            self.robot.connect_gripper()
            self._gripper_available = True
            print("Gripper connected.")
        except Exception as e:
            print(f"Warning: Could not connect gripper: {e}")
            self._gripper_available = False
        
        # Move robot to home position
        print(f"Moving robot to home position: {HOME_POSITION}")
        self.robot.move_to_position(HOME_POSITION, speed_factor=0.3, precise=True)
        print("Robot initialized at home position.")
    
    def _init_cameras(self):
        """Initialize RealSense cameras for observation."""
        import time
        from realsense_camera import list_connected_cameras
        
        print(f"\nInitializing cameras (resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
        
        # Reset all cameras first for clean state
        reset_all_cameras(wait_time=3.0)
        
        # Discover available cameras
        available_cameras = list_connected_cameras()
        print(f"Found {len(available_cameras)} RealSense camera(s):")
        for i, cam in enumerate(available_cameras):
            print(f"  [{i}] S/N: {cam['serial_number']}, Name: {cam['name']}")
        
        # Camera serial numbers in order: head, right wrist, left wrist
        camera_serials = [
            self.head_camera_serial,
            self.right_camera_serial, 
            self.left_camera_serial,
        ]
        camera_names = ["head", "right_wrist", "left_wrist"]
        
        # If serial numbers not specified, auto-assign from available cameras
        available_serials = [cam['serial_number'] for cam in available_cameras]
        for i, serial in enumerate(camera_serials):
            if serial is None and i < len(available_serials):
                camera_serials[i] = available_serials[i]
                print(f"  Auto-assigning camera {i} ({camera_names[i]}): S/N {available_serials[i]}")
        
        # Initialize each camera with specific serial number
        for i, (serial, name) in enumerate(zip(camera_serials, camera_names)):
            if serial is None:
                print(f"  Warning: No camera available for {name}")
                self.cameras.append(None)
                continue
                
            try:
                camera = RealSenseCamera(
                    serial_number=serial,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=30,
                    enable_color=True,
                    enable_depth=False,  # We only need color
                    align_to_color=False,
                )
                camera.start(warmup_timeout_ms=20000)  # Longer timeout for multi-camera
                self.cameras.append(camera)
                print(f"  Camera {i} ({name}): initialized (S/N: {serial})")
                
                # Add delay between camera starts to avoid USB contention
                if i < len(camera_serials) - 1:
                    time.sleep(1.0)
                    
            except Exception as e:
                print(f"  Warning: Failed to initialize {name} camera (S/N: {serial}): {e}")
                self.cameras.append(None)
        
        # Verify all cameras are initialized
        num_working = sum(1 for c in self.cameras if c is not None)
        print(f"Cameras initialized: {num_working}/3 working.")
    
    def reset(self, go_home: bool = True):
        """
        Reset the environment state.
        
        Args:
            go_home: If True and robot is connected, move robot to home position
        """
        # Reset robot if connected
        if self.use_robot and self.robot is not None and go_home:
            print(f"Moving robot to home position: {HOME_POSITION}")
            self.robot.move_to_position(HOME_POSITION, speed_factor=0.3, precise=True)
            # Open gripper
            if self._gripper_available:
                self.gripper_open()
        
        print("Environment reset.")
    
    # =========================================================================
    # Policy Inference
    # =========================================================================
    
    def get_action(self, prompt: str = None, observation: Dict = None, denormalize: bool = True) -> np.ndarray:
        """
        Run policy inference to get actions.
        
        Args:
            prompt: Natural language instruction for the task. If None, uses DEFAULT_TASK_INSTRUCTION.
            observation: Optional observation dict. If None, will capture from cameras/robot.
                        Expected format from get_observation() or manual dict with:
                        - 'images': List of 3 RGB images [head, right_wrist, left_wrist]
                        - 'state': 8D numpy array [7 joints + 1 gripper width in meters]
            denormalize: If True, denormalize the full action output:
                        - Joint velocities: denormalized using ACTIONS_MEAN/STD
                        - Gripper: converted from normalized (1=closed, 0=open) to meters (0=closed, 0.08=open)
        
        Returns:
            actions: Predicted action chunk as numpy array with shape (N, 8)
                     where N is the action horizon (typically 16)
                     Each action has format: [7 joint velocities, 1 gripper position]
                     If denormalize=True:
                        - Joint velocities are in rad/s (denormalized)
                        - Gripper is in meters (0 to 0.08)
                     If denormalize=False:
                        - All values are normalized as output by the model
        """
        if not self.use_policy or self.policy is None:
            raise RuntimeError("Policy not initialized. Set use_policy=True when creating the environment.")
        
        # Use default task instruction if not provided
        if prompt is None:
            prompt = DEFAULT_TASK_INSTRUCTION
        
        # Get observation if not provided
        if observation is None:
            observation = self.get_observation()
        
        # Build the policy input in the expected format
        # The policy expects specific observation keys based on the DROID format
        policy_input = self._build_policy_input(observation, prompt)
        
        # Run inference
        result = self.policy.infer(policy_input)
        actions = result["actions"].copy()
        
        # Denormalize actions if requested
        if denormalize:
            actions = self._denormalize_actions(actions)
        
        return actions
    
    def _normalize_gripper(self, gripper_width: float) -> float:
        """
        Normalize gripper width to policy input format.
        
        Physical width: 0.0 (closed) to 0.08 (open) meters
        Normalized: 1.0 (closed) to 0.0 (open) - INVERTED as per training data
        
        Args:
            gripper_width: Physical gripper width in meters (0 to 0.08)
        
        Returns:
            Normalized gripper value (1 = closed, 0 = open)
        """
        # Normalize to 0-1 range where 0=closed, 1=open
        normalized = np.clip(gripper_width / GRIPPER_MAX_WIDTH, 0.0, 1.0)
        # Invert so 1=closed, 0=open (matching training data format)
        return 1.0 - normalized
    
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Denormalize action array from policy output.
        
        Only the gripper dimension needs denormalization:
        - Joint velocities [0:7]: kept as-is (already in correct units from policy)
        - Gripper [7]: converted from normalized (1=closed, 0=open) to meters (0=closed, 0.08=open)
        
        Policy output format:
            - Joint velocities [0:7]: rad/s (no denormalization needed)
            - Gripper [7]: 1.0 (closed) to 0.0 (open)
        
        Output format:
            - Joint velocities [0:7]: rad/s (unchanged)
            - Gripper [7]: 0.0 (closed) to 0.08 (open) meters
        
        Args:
            actions: Action array with shape (N, 8)
        
        Returns:
            Actions with gripper converted to meters
        """
        actions = actions.copy()
        
        # Only convert gripper from normalized (0-1, 1=closed) to physical width (0-0.08m)
        # 1. Clip to 0-1 range (in case of outliers)
        # 2. Invert (1-x) so 0=closed, 1=open
        # 3. Scale to physical width
        gripper_normalized = np.clip(actions[..., -1], 0.0, 1.0)
        actions[..., -1] = (1.0 - gripper_normalized) * GRIPPER_MAX_WIDTH
        
        return actions
    
    def _resize_image_for_policy(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to match training data format: 240x320 -> 180x320
        
        This matches the preprocessing in convert_move_corn_pot_data_to_lerobot.py
        
        Args:
            img: Input image with shape (H, W, 3), typically (240, 320, 3)
        
        Returns:
            Resized image with shape (180, 320, 3)
        """
        from PIL import Image
        
        # Ensure uint8 format
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        
        # Resize to 180x320 (height x width) using PIL for consistency with training
        pil_img = Image.fromarray(img)
        resized = pil_img.resize((320, 180), resample=Image.BICUBIC)
        return np.array(resized)
    
    def _build_policy_input(self, observation: Dict, prompt: str) -> Dict:
        """
        Convert our observation format to the policy's expected input format.
        
        Preprocessing matches training data (convert_move_corn_pot_data_to_lerobot.py):
        - Images: Resized from 240x320 to 180x320
        - Gripper: Normalized and inverted (1=closed, 0=open)
        
        Our format:
            - images: List of 3 RGB images [head, right_wrist, left_wrist] (240, 320, 3)
            - state: 8D array [7 joints + 1 gripper width in meters]
        
        Policy format (DROID-style):
            - observation/exterior_image_1_left: (180, 320, 3) uint8 - head/left camera
            - observation/exterior_image_2_left: (180, 320, 3) uint8 - right camera  
            - observation/wrist_image_left: (180, 320, 3) uint8 - wrist camera
            - observation/joint_position: (7,) float32
            - observation/gripper_position: (1,) float32 - normalized (1=closed, 0=open)
            - prompt: str
        """
        images = observation['images']
        state = observation['state']
        
        # Extract state components (8D: 7 joints + 1 gripper)
        joint_position = state[:7]
        gripper_width = state[7]  # Physical width in meters
        
        # Normalize gripper to match training format (1=closed, 0=open)
        gripper_normalized = self._normalize_gripper(gripper_width)
        
        # Resize images to 180x320 as per training data format
        # images[0] = head camera (exterior_image_1_left / left_camera in training)
        # images[1] = right camera (exterior_image_2_left / right_camera in training)
        # images[2] = left wrist camera (wrist_image_left)
        exterior_image_1 = self._resize_image_for_policy(images[0])  # Head/left camera
        exterior_image_2 = self._resize_image_for_policy(images[1])  # Right camera
        wrist_image = self._resize_image_for_policy(images[2])       # Wrist camera
        
        policy_input = {
            "observation/exterior_image_1_left": exterior_image_1,
            "observation/exterior_image_2_left": exterior_image_2,
            "observation/wrist_image_left": wrist_image,
            "observation/joint_position": joint_position.astype(np.float32),
            "observation/gripper_position": np.array([gripper_normalized], dtype=np.float32),
            "prompt": prompt,
        }
        
        return policy_input
    
    def step(self, action: np.ndarray, speed_factor: float = 0.3) -> None:
        """
        Execute a single action on the robot using move_to_joint_position.
        
        This uses the high-level blocking motion command that handles
        trajectory planning internally, ensuring smooth motion.
        
        Args:
            action: Single action as numpy array with shape (8,)
                    Format: [7 joint deltas (rad), 1 gripper position in meters]
                    Joint deltas are added to current joint positions.
            speed_factor: Speed factor for motion (0.0-1.0, default 0.3)
        """
        if not self.use_robot or self.robot is None:
            return
        
        action = np.asarray(action).flatten()
        
        if action.shape != (8,):
            raise ValueError(f"action must have shape (8,), got {action.shape}")
        
        joint_deltas = action[:7]  # First 7 are joint deltas (velocities)
        gripper_value = action[7]  # Last one is gripper position in meters
        
        # Get current joint positions
        current_joints = self.get_robot_state()
        if current_joints is None:
            print("Warning: Could not get current joint state, skipping step")
            return
        
        # Compute target joint positions by adding deltas
        target_joints = current_joints + joint_deltas
        
        # Move to target joint position (blocking, handles trajectory internally)
        self.robot.move_to_joint_position(target_joints, speed_factor=speed_factor)
        
        # Handle gripper
        # gripper_value is in meters (0 = closed, 0.08 = open)
        if self._gripper_available:
            self._update_gripper(gripper_value)
    
    def execute_actions(
        self,
        actions: np.ndarray,
        num_steps: int = None,
        speed_factor: float = 0.3,
    ) -> int:
        """
        Execute multiple actions on the robot using move_to_joint_position.
        
        Args:
            actions: Actions array with shape (N, 8)
            num_steps: Number of steps to execute (default: all)
            speed_factor: Speed factor for motion (0.0-1.0, default 0.3)
        
        Returns:
            Number of actions executed
        """
        if not self.use_robot:
            print("Warning: Robot not enabled, simulating execution")
            return 0
        
        actions = np.asarray(actions)
        if len(actions.shape) == 1:
            actions = actions.reshape(1, -1)
        
        num_to_execute = num_steps if num_steps is not None else len(actions)
        num_to_execute = min(num_to_execute, len(actions))
        
        for i in range(num_to_execute):
            self.step(actions[i], speed_factor=speed_factor)
        
        return num_to_execute
    
    # =========================================================================
    # Gripper Control
    # =========================================================================
    
    def _update_gripper(self, gripper_value: float, threshold: float = 0.04) -> None:
        """
        Update gripper based on target width value using boolean control.
        
        Only calls gripper functions when state changes (open <-> closed).
        Uses a threshold to decide between open/close states.
        
        Args:
            gripper_value: Target gripper width in meters (0 = closed, 0.08 = open)
            threshold: Width threshold for open/close decision (default: 0.04 = half open)
        """
        if not self._gripper_available:
            return
        
        # Determine target state: closed if below threshold, open otherwise
        should_close = gripper_value < threshold
        
        # Only act if state changes
        if should_close and not self._gripper_closed:
            # Transition to closed state
            try:
                self.robot.gripper.grasp(0.0, speed=0.1, force=10.0, epsilon_inner=0.08, epsilon_outer=0.08)
                self._gripper_closed = True
            except Exception as e:
                print(f"Warning: Could not close gripper: {e}")
        elif not should_close and self._gripper_closed:
            # Transition to open state
            try:
                self.robot.gripper.move(0.08, speed=0.3)
                self._gripper_closed = False
            except Exception as e:
                print(f"Warning: Could not open gripper: {e}")
    
    def get_gripper_width(self) -> float:
        """
        Get the current gripper width in meters.
        
        Returns:
            Gripper width in meters (0.0 = fully closed, 0.08 = fully open)
        """
        if not self.use_robot or self.robot is None or not self._gripper_available:
            return GRIPPER_MAX_WIDTH  # Default to open if gripper not available
        
        try:
            gripper_state = self.robot.gripper.read_once()
            width = gripper_state.width  # Width in meters (0 to 0.08)
            return float(np.clip(width, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH))
        except Exception as e:
            print(f"Warning: Could not read gripper state: {e}")
            return GRIPPER_MAX_WIDTH
    
    def get_gripper_state(self) -> float:
        """
        Get the current gripper state as physical width in meters.
        
        This is an alias for get_gripper_width() for backward compatibility.
        
        Returns:
            Gripper width in meters (0.0 = fully closed, 0.08 = fully open)
        """
        return self.get_gripper_width()
    
    def move_gripper(self, width: float, speed: float = 0.1) -> None:
        """
        Move the gripper to a specific width.
        
        Args:
            width: Target width in meters (0.0 to 0.08)
        """
        if not self.use_robot or self.robot is None or not self._gripper_available:
            return
        
        # Clamp width to valid range
        width = np.clip(width, GRIPPER_MIN_WIDTH, GRIPPER_MAX_WIDTH)
        
        try:
            self.robot.gripper.move(width, speed)  # speed = 0.1 m/s
        except Exception as e:
            print(f"Warning: Could not move gripper: {e}")
    
    def move_gripper_normalized(self, normalized_width: float) -> None:
        """
        Move the gripper using a normalized value.
        
        Args:
            normalized_width: Target width normalized to [0, 1]
                             0 = fully closed, 1 = fully open
        """
        width = np.clip(normalized_width, 0.0, 1.0) * GRIPPER_MAX_WIDTH
        self.move_gripper(width)
    
    def gripper_open(self, width: float = 0.08):
        """Open the gripper."""
        self.move_gripper(width)
    
    def gripper_close(self, width: float = 0.0, force: float = 10.0):
        """Close the gripper."""
        if self.use_robot and self.robot is not None and self._gripper_available:
            try:
                self.robot.gripper.grasp(width, speed=0.1, force=force)
            except Exception as e:
                print(f"Warning: Could not close gripper: {e}")
                self.move_gripper(width)
    
    # =========================================================================
    # Robot State Access
    # =========================================================================
    
    def get_robot_state(self) -> Optional[np.ndarray]:
        """
        Get current robot joint positions.
        
        Returns:
            Joint positions (7,) or None if robot not connected
        """
        if not self.use_robot or self.robot is None:
            return None
        return self.robot.get_joint_positions()
    
    def get_robot_pose(self) -> Optional[np.ndarray]:
        """
        Get current robot end-effector pose.
        
        Returns:
            4x4 pose matrix or None if robot not connected
        """
        if not self.use_robot or self.robot is None:
            return None
        return self.robot.get_pose()
    
    # =========================================================================
    # Observation Interface
    # =========================================================================
    
    def get_camera_frames(self) -> List[np.ndarray]:
        """
        Get color frames from all three cameras.
        
        Returns:
            List of 3 RGB images [head, right_wrist, left_wrist]
            Each image has shape (240, 320, 3) in RGB format
        """
        frames = []
        for i, camera in enumerate(self.cameras):
            if camera is not None:
                try:
                    color, _ = camera.get_frames()
                    if color is not None:
                        # Convert BGR to RGB (RealSense returns BGR)
                        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                        frames.append(rgb)
                    else:
                        # Return black image if capture failed
                        frames.append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))
                except Exception as e:
                    print(f"Warning: Failed to get frame from camera {i}: {e}")
                    frames.append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))
            else:
                # Camera not initialized, return black image
                frames.append(np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8))
        
        return frames
    
    def get_observation(self) -> Dict:
        """
        Get a complete observation from the real-world environment.
        
        Returns:
            Dictionary with keys:
            - 'images': List of 3 RGB images [head_camera, right_camera, left_wrist_camera]
                        each with shape (240, 320, 3)
            - 'state': 8D numpy array [7 joints + 1 gripper width in meters]
                       Gripper width is in meters (0.0 = closed, 0.08 = open)
        """
        # 1. Get camera frames (head, right, left_wrist)
        images = self.get_camera_frames()
        
        # 2. Get robot state
        # Get current joint positions (7D)
        qpos = self.get_robot_state()
        if qpos is None:
            # Fallback to zeros if robot state unavailable
            qpos = np.zeros(7, dtype=np.float64)
        
        # Get gripper width in meters (0.0 to 0.08)
        gripper_width = self.get_gripper_state()
        
        # Build 8D state: [7 joint positions, 1 gripper width in meters]
        state = np.concatenate([qpos, [gripper_width]])
        
        return {
            'images': images,
            'state': state,
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_action_dim(self) -> int:
        """Return the action dimension (8 = 7 joint velocities + 1 gripper)."""
        return 8
    
    def close(self):
        """Clean up resources, stop cameras, and disconnect from robot."""
        # Stop cameras
        for i, camera in enumerate(self.cameras):
            if camera is not None:
                try:
                    camera.stop(hardware_reset=False)
                    print(f"Camera {i} stopped.")
                except Exception as e:
                    print(f"Warning: Failed to stop camera {i}: {e}")
        self.cameras = []
        
        # Move robot to home and disconnect
        if self.use_robot and self.robot is not None:
            print("Moving robot to home position before closing...")
            try:
                self.robot.move_to_start()
                self.robot.gripper.move(0.08, speed=0.3)  # Open gripper
            except Exception as e:
                print(f"Warning: Failed to move robot to home: {e}")
            print("Disconnecting from robot...")
            self.robot.disconnect()
            self.robot = None
        print("Environment closed.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - resets and closes the environment."""
        self.close()
        return False


# =============================================================================
# Testing Functions
# =============================================================================

def create_fake_observation(
    image_shape: tuple = (240, 320, 3),
    seed: int = None
) -> tuple:
    """
    Create fake observation data for testing.
    
    Args:
        image_shape: Shape of each camera image (H, W, 3), default (240, 320, 3)
        seed: Random seed for reproducibility
    
    Returns:
        images: List of 3 fake RGB images [head, right, left_wrist]
        state: 8D robot state vector [7 joints + 1 gripper width in meters]
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create 3 fake camera images (head, right, left_wrist)
    images = [
        np.random.randint(0, 256, image_shape, dtype=np.uint8)
        for _ in range(3)
    ]
    
    # Create fake 8D state vector: [7 joint positions, 1 gripper width in meters]
    # Gripper width is in meters (0.0 to 0.08), 0.08 = fully open
    state = np.array([
        0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.08  # typical neutral joints + gripper open
    ], dtype=np.float64)
    
    return images, state


def test_with_robot():
    """
    Test with real robot connection.
    """
    print("=" * 60)
    print("Testing RealWorldEnv with Robot")
    print("=" * 60)
    
    try:
        # Create environment with robot (no policy for basic test)
        env = RealWorldEnv(
            use_robot=True,
            use_policy=False,
        )
        
        # Reset (moves robot to home)
        env.reset(go_home=True)
        
        # Get observation
        print("\nGetting observation...")
        obs = env.get_observation()
        print(f"Images: {len(obs['images'])} images, shapes: {[img.shape for img in obs['images']]}")
        print(f"State: shape={obs['state'].shape}")
        print(f"State values: {obs['state']}")
        
        # Test gripper
        print("\nTesting gripper...")
        print(f"Gripper state: {env.get_gripper_state()}")
        
        env.close()
        
        print("\n" + "=" * 60)
        print("ROBOT TEST PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_without_robot():
    """
    Test environment initialization without robot (for debugging).
    """
    print("=" * 60)
    print("Testing RealWorldEnv (No Robot, No Policy)")
    print("=" * 60)
    
    try:
        # Create environment without robot or policy
        env = RealWorldEnv(use_robot=False, use_policy=False)
        
        # Create fake observation
        images, state = create_fake_observation(seed=42)
        
        print(f"\nFake observation created:")
        print(f"Images: {len(images)} images, shapes: {[img.shape for img in images]}")
        print(f"State: shape={state.shape}")
        
        env.close()
        
        print("\n" + "=" * 60)
        print("TEST PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_only():
    """
    Test policy inference without robot connection.
    """
    print("=" * 60)
    print("Testing Policy Inference (No Robot)")
    print("=" * 60)
    
    try:
        # Create environment with policy but no robot
        # Uses default config: pi05_move_corn_pot_finetune
        env = RealWorldEnv(
            use_robot=False,
            use_policy=True,
            # Uses DEFAULT_POLICY_CONFIG and DEFAULT_CHECKPOINT
        )
        
        # Create fake observation
        images, state = create_fake_observation(seed=42)
        observation = {"images": images, "state": state}
        
        print(f"\nInput observation:")
        print(f"  Images: {len(images)} images, shapes: {[img.shape for img in images]}")
        print(f"  State: shape={state.shape}")
        print(f"  Joint positions: {state[:7]}")
        print(f"  Gripper width: {state[7]:.4f}m")
        
        # Show preprocessed input
        prompt = DEFAULT_TASK_INSTRUCTION
        policy_input = env._build_policy_input(observation, prompt)
        print(f"\nPreprocessed policy input:")
        print(f"  exterior_image_1_left shape: {policy_input['observation/exterior_image_1_left'].shape}")
        print(f"  exterior_image_2_left shape: {policy_input['observation/exterior_image_2_left'].shape}")
        print(f"  wrist_image_left shape: {policy_input['observation/wrist_image_left'].shape}")
        print(f"  joint_position: {policy_input['observation/joint_position']}")
        print(f"  gripper_position (normalized, 1=closed): {policy_input['observation/gripper_position']}")
        
        # Run inference using default task instruction
        print(f"\nRunning inference with prompt: '{prompt}'")
        actions = env.get_action(prompt, observation, denormalize=True)
        
        print(f"\nOutput actions (denormalized):")
        print(f"  Shape: {actions.shape}")
        print(f"  First action - joint velocities (rad/s): {actions[0, :7]}")
        print(f"  First action - gripper width (meters): {actions[0, 7]:.4f}m")
        
        # Also show raw (normalized) output
        actions_raw = env.get_action(prompt, observation, denormalize=False)
        print(f"\nOutput actions (raw/normalized):")
        print(f"  First action - joint velocities (normalized): {actions_raw[0, :7]}")
        print(f"  First action - gripper (normalized, 1=closed): {actions_raw[0, 7]:.4f}")
        
        env.close()
        
        print("\n" + "=" * 60)
        print("POLICY TEST PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real World Environment")
    parser.add_argument("--robot", action="store_true", help="Test with real robot")
    parser.add_argument("--policy", action="store_true", help="Test policy inference only")
    args = parser.parse_args()
    
    if args.robot:
        test_with_robot()
    elif args.policy:
        test_policy_only()
    else:
        test_without_robot()
