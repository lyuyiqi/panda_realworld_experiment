"""
Basic robot control wrapper for Franka Panda using panda-py.
Designed for 20-50 Hz Cartesian end-effector control.
"""
import logging
import time
from typing import Optional, Tuple, List
from contextlib import contextmanager

import numpy as np

import panda_py
from panda_py import controllers, libfranka

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PandaController:
    """
    A controller class for controlling Franka Panda robot.
    Optimized for 20-50 Hz Cartesian end-effector control.
    """

    HOSTNAME = "192.168.18.10"

    def __init__(
        self,
        default_speed_factor: float = 0.2,
        default_stiffness: Optional[List[float]] = None,
    ):
        """
        Initialize the Panda robot wrapper.

        Args:
            default_speed_factor: Default speed factor for movements (0.0-1.0)
            default_stiffness: Default joint stiffness values (7 values)
        """
        self.default_speed_factor = default_speed_factor
        self.default_stiffness = default_stiffness or [600, 600, 600, 600, 250, 150, 50]

        self._panda: Optional[panda_py.Panda] = None
        self._gripper: Optional[libfranka.Gripper] = None
        self._controller = None  # Can be CartesianImpedance or JointPosition
        self._is_streaming = False
        self._streaming_precise = False  # True if using JointPosition controller

    def connect(self) -> None:
        """
        Connect to the robot.
        
        Note: Unlock brakes and activate FCI manually via Desk before calling this.
        """
        logger.info(f"Connecting to Panda at {self.HOSTNAME}")
        self._panda = panda_py.Panda(self.HOSTNAME)
        self._panda.set_default_behavior()
        logger.info("Robot connected successfully")

    def connect_gripper(self) -> None:
        """Connect to the Franka gripper."""
        logger.info("Connecting to gripper")
        self._gripper = libfranka.Gripper(self.HOSTNAME)

    def disconnect(self) -> None:
        """Disconnect from the robot and clean up."""
        self.stop_streaming()
        if self._panda is not None:
            logger.info("Disconnecting from robot")
            self._panda = None
        if self._gripper is not None:
            self._gripper = None

    @property
    def panda(self) -> panda_py.Panda:
        """Get the underlying Panda object."""
        if self._panda is None:
            raise RuntimeError("Robot not connected. Call connect() first.")
        return self._panda

    @property
    def gripper(self) -> libfranka.Gripper:
        """Get the gripper object."""
        if self._gripper is None:
            raise RuntimeError("Gripper not connected. Call connect_gripper() first.")
        return self._gripper

    # =========================================================================
    # State Retrieval
    # =========================================================================

    def get_pose(self) -> np.ndarray:
        """
        Get current end-effector pose as 4x4 homogeneous transformation matrix.

        Returns:
            4x4 numpy array representing the pose
        """
        return self.panda.get_pose()

    def get_position(self) -> np.ndarray:
        """
        Get current end-effector position (x, y, z).

        Returns:
            3D position vector
        """
        return self.panda.get_position()

    def get_orientation(self) -> np.ndarray:
        """
        Get current end-effector orientation as quaternion.
        
        NOTE: panda-py returns quaternion in (x, y, z, w) order (scalar-last),
        NOT the (w, x, y, z) order that some libraries use.

        Returns:
            Quaternion as numpy array in (x, y, z, w) format
        """
        return self.panda.get_orientation()

    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.

        Returns:
            7D joint position vector
        """
        return np.array(self.panda.q)

    def get_joint_velocities(self) -> np.ndarray:
        """
        Get current joint velocities.

        Returns:
            7D joint velocity vector
        """
        return np.array(self.panda.dq)

    def get_state(self):
        """Get the full robot state."""
        return self.panda.get_state()

    # =========================================================================
    # Basic Motion Commands
    # =========================================================================

    def move_to_start(self, speed_factor: Optional[float] = None) -> None:
        """Move robot to the default start position."""
        sf = speed_factor or self.default_speed_factor
        logger.info("Moving to start position")
        self.panda.move_to_start(speed_factor=sf)

    def move_to_joint_position(
        self,
        q: np.ndarray,
        speed_factor: Optional[float] = None,
        stiffness: Optional[List[float]] = None,
    ) -> None:
        """
        Move to a target joint position.

        Args:
            q: Target joint positions (7D)
            speed_factor: Speed factor (0.0-1.0)
            stiffness: Joint stiffness values
        """
        sf = speed_factor or self.default_speed_factor
        stiff = stiffness or self.default_stiffness
        self.panda.move_to_joint_position(q, speed_factor=sf, stiffness=stiff)

    def move_to_pose(
        self,
        pose: np.ndarray,
        speed_factor: Optional[float] = None,
        impedance: Optional[np.ndarray] = None,
        success_threshold: float = 0.001,
    ) -> None:
        """
        Move to a target Cartesian pose.

        Args:
            pose: Target pose as 4x4 matrix or list of poses
            speed_factor: Speed factor (0.0-1.0)
            impedance: 6x6 Cartesian impedance matrix (default: diag([800,800,800,40,40,40]))
            success_threshold: Position error threshold in meters (default: 1mm)
        """
        sf = speed_factor or self.default_speed_factor
        if impedance is not None:
            self.panda.move_to_pose(pose, speed_factor=sf, impedance=impedance, success_threshold=success_threshold)
        else:
            self.panda.move_to_pose(pose, speed_factor=sf, success_threshold=success_threshold)

    def move_to_position(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        speed_factor: Optional[float] = None,
        precise: bool = False,
    ) -> None:
        """
        Move to a target position in WORLD FRAME, optionally with orientation.

        Args:
            position: Target position (x, y, z) in world frame
            orientation: Target orientation as quaternion (x, y, z, w), or None to keep current
            speed_factor: Speed factor (0.0-1.0)
            precise: If True, use IK + joint motion for precise positioning (~1mm error).
                     If False, use Cartesian impedance control (~10mm error but compliant).
        """
        # Build target pose matrix
        pose = self.get_pose().copy()
        pose[:3, 3] = np.asarray(position).flatten()
        if orientation is not None:
            pose[:3, :3] = self._quaternion_to_rotation_matrix(np.asarray(orientation).flatten())
        
        sf = speed_factor or self.default_speed_factor
        
        if precise:
            # Use IK + joint motion for precise positioning (like tutorial Code Block 3)
            q_target = panda_py.ik(pose)
            if q_target is None:
                raise RuntimeError(f"IK failed for position {position}")
            self.panda.move_to_joint_position(q_target, speed_factor=sf)
        else:
            # Use Cartesian impedance control (compliant but ~10mm error)
            self.panda.move_to_pose(pose, speed_factor=sf)

    def move(
        self,
        direction: np.ndarray,
        speed_factor: Optional[float] = None,
        precise: bool = True,
    ) -> None:
        """
        Move the end-effector in a direction (relative motion) in WORLD FRAME.

        This is a convenience method that moves relative to the current position
        while maintaining the current orientation.

        Args:
            direction: [dx, dy, dz] displacement in meters (world frame)
            speed_factor: Speed factor (0.0-1.0)
            precise: If True, use IK + joint motion for precise positioning (~1mm error).
                     If False, use Cartesian impedance control (~10mm error but compliant).

        Example:
            robot.move([0.1, 0.0, 0.0])  # Move 10cm in world +X direction
            robot.move([0.0, 0.0, -0.05])  # Move 5cm down (world -Z)
        """
        direction = np.asarray(direction)
        if direction.shape != (3,):
            raise ValueError("Direction must be a 3D vector [dx, dy, dz]")
        
        current_position = self.get_position()
        target_position = current_position + direction
        self.move_to_position(target_position, speed_factor=speed_factor, precise=precise)

    # =========================================================================
    # Real-time Cartesian Streaming Control (20-50 Hz)
    # =========================================================================

    def stream_move(
        self,
        direction: np.ndarray,
    ) -> None:
        """
        Set streaming target as a relative motion in WORLD FRAME.

        This is a convenience method for streaming control that moves relative
        to the current position while maintaining the current orientation.

        Args:
            direction: [dx, dy, dz] displacement in meters (world frame)

        Note:
            Must be in streaming mode (call start_streaming() first).
            Call this at 20-50 Hz within a streaming context.
        """
        direction = np.asarray(direction)
        if direction.shape != (3,):
            raise ValueError("Direction must be a 3D vector [dx, dy, dz]")
        
        current_position = self.get_position()
        current_orientation = self.get_orientation()
        target_position = current_position + direction
        self.set_cartesian_target(target_position, current_orientation)

    def start_streaming(
        self,
        impedance: Optional[np.ndarray] = None,
        damping_ratio: float = 1.0,
        nullspace_stiffness: float = 0.5,
        filter_coeff: float = 1.0,
        precise: bool = False,
    ) -> None:
        """
        Start streaming control mode.
        
        This prepares the robot for real-time pose commands at 20-50 Hz.
        
        NOTE: For precise mode, prefer using streaming_context() which handles
        the timing-critical controller initialization properly.

        Args:
            impedance: 6x6 Cartesian impedance matrix (only for precise=False)
            damping_ratio: Damping ratio (default 1.0 = critically damped)
            nullspace_stiffness: Nullspace stiffness (default 0.5)
            filter_coeff: Filter coefficient for smoothing (0.0-1.0, higher = less smoothing)
            precise: If True, use JointPosition controller with IK for precise positioning.
                     If False, use CartesianImpedance controller (compliant but ~10mm error).
        """
        if self._is_streaming:
            logger.warning("Already in streaming mode")
            return

        self._streaming_precise = precise
        
        if precise:
            logger.info("Starting precise joint position streaming control")
            self._controller = controllers.JointPosition()
            self.panda.start_controller(self._controller)
            # NOTE: For JointPosition, user must immediately enter a control context
            # and call set_cartesian_target, otherwise robot will abort with NaN error.
            # Prefer using streaming_context(precise=True) instead.
        else:
            logger.info("Starting Cartesian impedance streaming control")
            if impedance is None:
                impedance = np.diag([800.0, 800.0, 800.0, 40.0, 40.0, 40.0])
            
            self._controller = controllers.CartesianImpedance(
                damping_ratio=damping_ratio,
                nullspace_stiffness=nullspace_stiffness,
                filter_coeff=filter_coeff,
            )
            self.panda.start_controller(self._controller)
        
        self._is_streaming = True

    def stop_streaming(self) -> None:
        """Stop streaming control mode."""
        if not self._is_streaming:
            return

        logger.info("Stopping streaming control")
        if self._panda is not None:
            self._panda.stop_controller()
        self._controller = None
        self._is_streaming = False
        self._streaming_precise = False

    def set_joint_target(
        self,
        joint_positions: np.ndarray,
        joint_velocities: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set target joint positions for streaming control.
        
        Call this at 20-50 Hz after start_streaming(precise=True).
        
        This method directly sets joint targets without IK conversion,
        useful when you already have joint-space commands (e.g., from a policy).

        Args:
            joint_positions: Target joint positions (7D array in radians)
            joint_velocities: Target joint velocities (7D array, default: zeros)
        
        Raises:
            RuntimeError: If not in streaming mode or not using JointPosition controller
            ValueError: If joint_positions has wrong shape
        """
        if not self._is_streaming or self._controller is None:
            raise RuntimeError("Not in streaming mode. Call start_streaming(precise=True) first.")
        
        if not self._streaming_precise:
            raise RuntimeError("set_joint_target() requires precise mode. "
                             "Call start_streaming(precise=True) or use set_cartesian_target() instead.")
        
        joint_positions = np.asarray(joint_positions).flatten()
        if joint_positions.shape != (7,):
            raise ValueError(f"joint_positions must have shape (7,), got {joint_positions.shape}")
        
        if joint_velocities is None:
            joint_velocities = np.zeros(7)
        else:
            joint_velocities = np.asarray(joint_velocities).flatten()
            if joint_velocities.shape != (7,):
                raise ValueError(f"joint_velocities must have shape (7,), got {joint_velocities.shape}")
        
        # Stream joint position with specified velocity
        self._controller.set_control(joint_positions, joint_velocities)

    def set_cartesian_target(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        """
        Set target Cartesian pose for streaming control.
        
        Call this at 20-50 Hz after start_streaming().

        Args:
            position: Target position (x, y, z) in world frame
            orientation: Target orientation as quaternion (x, y, z, w) - panda-py convention
        """
        if not self._is_streaming or self._controller is None:
            raise RuntimeError("Not in streaming mode. Call start_streaming() first.")
        
        if self._streaming_precise:
            # Use IK to compute joint positions, then stream to JointPosition controller
            # IMPORTANT: Use get_pose() as base and modify position, rather than
            # reconstructing from quaternion (which has convention issues)
            pose = self.get_pose().copy()
            pose[:3, 3] = np.asarray(position).flatten()
            # For orientation changes, we need to use the robot's actual rotation matrix format
            # For now, we keep the current orientation from the pose
            # TODO: Support orientation changes if needed
            
            q_current = self.get_joint_positions()
            q_target = panda_py.ik(pose, q_current)
            
            if q_target is None:
                logger.warning(f"IK returned None for position {position}, holding current position")
                q_target = q_current
            elif np.any(np.isnan(q_target)):
                logger.warning(f"IK returned NaN: {q_target}, holding current position")
                q_target = q_current
            
            # Stream joint position with zero velocity
            self._controller.set_control(q_target, np.zeros(7))
        else:
            # Cartesian impedance mode
            self._controller.set_control(position, orientation)

    def set_pose_target(self, pose: np.ndarray) -> None:
        """
        Set target pose for streaming control using 4x4 matrix.

        Args:
            pose: Target pose as 4x4 homogeneous transformation matrix
        """
        position = pose[:3, 3]
        orientation = self._rotation_matrix_to_quaternion(pose[:3, :3])
        self.set_cartesian_target(position, orientation)

    @contextmanager
    def streaming_context(
        self,
        frequency: float = 50.0,
        max_runtime: Optional[float] = None,
        impedance: Optional[np.ndarray] = None,
        damping_ratio: float = 1.0,
        nullspace_stiffness: float = 0.5,
        filter_coeff: float = 1.0,
        precise: bool = False,
    ):
        """
        Context manager for streaming Cartesian control.

        Args:
            frequency: Control loop frequency in Hz (20-50 recommended)
            max_runtime: Maximum runtime in seconds (None for infinite)
            impedance: 6x6 Cartesian impedance matrix (only for precise=False)
            damping_ratio: Damping ratio (default 1.0 = critically damped)
            nullspace_stiffness: Nullspace stiffness (default 0.5)
            filter_coeff: Filter coefficient for smoothing
            precise: If True, use JointPosition controller with IK for precise positioning (~1-3mm).
                     If False, use CartesianImpedance controller (compliant but ~10mm error).

        Yields:
            PandaContext for the control loop

        Example:
            # Compliant streaming (default, ~10mm error)
            with robot.streaming_context(frequency=30, max_runtime=10) as ctx:
                while ctx.ok():
                    robot.set_cartesian_target(position, orientation)
            
            # Precise streaming (~1-3mm error)
            with robot.streaming_context(frequency=30, max_runtime=10, precise=True) as ctx:
                while ctx.ok():
                    robot.set_cartesian_target(position, orientation)
        """
        # For JointPosition controller, we need to start the controller and immediately
        # enter the control context to avoid NaN errors. The controller runs at 1kHz
        # internally and needs valid commands from the very first tick.
        self._streaming_precise = precise
        self._is_streaming = True
        
        if precise:
            logger.info("Starting precise joint position streaming control")
            self._controller = controllers.JointPosition()
            self.panda.start_controller(self._controller)
            # Enter context immediately - user must call set_cartesian_target on first iteration
            try:
                with self.panda.create_context(frequency=frequency, max_runtime=max_runtime) as ctx:
                    # Send initial command immediately before yielding to user
                    q_current = self.get_joint_positions()
                    self._controller.set_control(q_current, np.zeros(7))
                    yield ctx
            finally:
                self.stop_streaming()
        else:
            logger.info("Starting Cartesian impedance streaming control")
            if impedance is None:
                impedance = np.diag([800.0, 800.0, 800.0, 40.0, 40.0, 40.0])
            
            self._controller = controllers.CartesianImpedance(
                damping_ratio=damping_ratio,
                nullspace_stiffness=nullspace_stiffness,
                filter_coeff=filter_coeff,
            )
            self.panda.start_controller(self._controller)
            try:
                with self.panda.create_context(frequency=frequency, max_runtime=max_runtime) as ctx:
                    yield ctx
            finally:
                self.stop_streaming()

    # =========================================================================
    # Gripper Control
    # =========================================================================

    def gripper_open(self, width: float = 0.08, speed: float = 0.1) -> None:
        """Open the gripper to specified width."""
        self.gripper.move(width, speed)

    def gripper_close(
        self,
        width: float = 0.0,
        speed: float = 0.1,
        force: float = 10.0,
        epsilon_inner: float = 0.04,
        epsilon_outer: float = 0.04,
    ) -> bool:
        """
        Close the gripper and grasp an object.

        Returns:
            True if grasp was successful
        """
        return self.gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)

    def gripper_home(self) -> None:
        """Home the gripper."""
        self.gripper.homing()

    # =========================================================================
    # Teaching Mode
    # =========================================================================

    def enable_teaching_mode(self) -> None:
        """Enable teaching/gravity compensation mode."""
        logger.info("Enabling teaching mode")
        self.panda.teaching_mode(True)

    def disable_teaching_mode(self) -> None:
        """Disable teaching mode."""
        logger.info("Disabling teaching mode")
        self.panda.teaching_mode(False)

    @contextmanager
    def teaching_context(self):
        """Context manager for teaching mode."""
        self.enable_teaching_mode()
        try:
            yield
        finally:
            self.disable_teaching_mode()

    # =========================================================================
    # Logging
    # =========================================================================

    def enable_logging(self, buffer_size: int = 10000) -> None:
        """Enable state logging with specified buffer size."""
        self.panda.enable_logging(buffer_size)

    def disable_logging(self) -> None:
        """Disable state logging."""
        self.panda.disable_logging()

    def get_log(self) -> dict:
        """Get the logged data."""
        return self.panda.get_log()

    # =========================================================================
    # Utility Functions
    # =========================================================================

    def set_collision_behavior(
        self,
        torque_threshold: float = 100.0,
        force_threshold: float = 100.0,
    ) -> None:
        """
        Set collision detection thresholds. Higher values = less sensitive.

        Args:
            torque_threshold: Joint torque threshold
            force_threshold: Cartesian force threshold
        """
        self.panda.get_robot().set_collision_behavior(
            [torque_threshold] * 7,
            [torque_threshold] * 7,
            [force_threshold] * 6,
            [force_threshold] * 6,
        )

    @staticmethod
    def compute_ik(pose: np.ndarray, q_init: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute inverse kinematics for a given pose.

        Args:
            pose: Target pose as 4x4 matrix
            q_init: Initial joint configuration (optional)

        Returns:
            Joint positions
        """
        if q_init is not None:
            return panda_py.ik(pose, q_init)
        return panda_py.ik(pose)

    @staticmethod
    def compute_fk(q: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint positions.

        Args:
            q: Joint positions (7D)

        Returns:
            4x4 pose matrix
        """
        return panda_py.fk(q)

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        NOTE: panda-py uses (x, y, z, w) quaternion order (scalar-last).
        """
        x, y, z, w = q  # panda-py order: (x, y, z, w)
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])

    @staticmethod
    def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to quaternion.
        
        NOTE: Returns (x, y, z, w) order to match panda-py convention.
        """
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w])  # panda-py order: (x, y, z, w)


# =============================================================================
# Example Usage and Tests
# =============================================================================

def test_move_relative(robot: PandaController):
    """
    Test relative motion (move) in world frame.
    
    This verifies that moving in world X/Y/Z directions works correctly
    regardless of the end-effector orientation.
    """
    print("\n--- Testing Relative Motion (move) in World Frame ---\n")
    
    robot.move_to_start()
    initial_pos = robot.get_position().copy()
    print(f"Initial position: {initial_pos}")
    
    # Test 1: Move in world +X
    print("\nTest 1: Moving 5cm in world +X direction...")
    robot.move([0.05, 0.0, 0.0])
    pos_after_x = robot.get_position()
    delta_x = pos_after_x - initial_pos
    print(f"Position after: {pos_after_x}")
    print(f"Delta: {delta_x}")
    print(f"Expected delta: [0.05, 0.0, 0.0]")
    error_x = np.linalg.norm(delta_x - np.array([0.05, 0.0, 0.0]))
    print(f"Error: {error_x*1000:.2f} mm")
    assert error_x < 0.003, f"X motion error too large: {error_x*1000:.2f} mm (threshold: 3mm)"
    
    # Test 2: Move in world +Y
    print("\nTest 2: Moving 5cm in world +Y direction...")
    pos_before_y = robot.get_position().copy()
    robot.move([0.0, 0.05, 0.0])
    pos_after_y = robot.get_position()
    delta_y = pos_after_y - pos_before_y
    print(f"Position after: {pos_after_y}")
    print(f"Delta: {delta_y}")
    print(f"Expected delta: [0.0, 0.05, 0.0]")
    error_y = np.linalg.norm(delta_y - np.array([0.0, 0.05, 0.0]))
    print(f"Error: {error_y*1000:.2f} mm")
    assert error_y < 0.003, f"Y motion error too large: {error_y*1000:.2f} mm (threshold: 3mm)"
    
    # Test 3: Move in world -Z (down)
    print("\nTest 3: Moving 5cm in world -Z direction (down)...")
    pos_before_z = robot.get_position().copy()
    robot.move([0.0, 0.0, -0.05])
    pos_after_z = robot.get_position()
    delta_z = pos_after_z - pos_before_z
    print(f"Position after: {pos_after_z}")
    print(f"Delta: {delta_z}")
    print(f"Expected delta: [0.0, 0.0, -0.05]")
    error_z = np.linalg.norm(delta_z - np.array([0.0, 0.0, -0.05]))
    print(f"Error: {error_z*1000:.2f} mm")
    assert error_z < 0.003, f"Z motion error too large: {error_z*1000:.2f} mm (threshold: 3mm)"
    
    # Move back to start
    print("\nMoving back to start position...")
    robot.move_to_start()
    
    print("\n✓ All relative motion tests passed!")


def test_move_to_position(robot: PandaController):
    """
    Test absolute position motion (move_to_position) in world frame.
    
    This verifies that moving to absolute coordinates works correctly.
    """
    print("\n--- Testing Absolute Motion (move_to_position) in World Frame ---\n")
    
    robot.move_to_start()
    initial_pos = robot.get_position().copy()
    print(f"Initial position: {initial_pos}")
    
    # Test 1: Move to offset position (using precise=True, IK + joint motion)
    target1 = initial_pos + np.array([0.1, 0.05, -0.05])
    print(f"\nTest 1: Moving to absolute position {target1}...")
    robot.move_to_position(target1)  # precise=True by default
    actual1 = robot.get_position()
    error1 = np.linalg.norm(actual1 - target1)
    print(f"Target: {target1}")
    print(f"Actual: {actual1}")
    print(f"Error: {error1*1000:.2f} mm")
    assert error1 < 0.003, f"Position error too large: {error1*1000:.2f} mm (threshold: 3mm)"
    
    # Test 2: Move to another position
    target2 = initial_pos + np.array([-0.05, 0.1, 0.0])
    print(f"\nTest 2: Moving to absolute position {target2}...")
    robot.move_to_position(target2)
    actual2 = robot.get_position()
    error2 = np.linalg.norm(actual2 - target2)
    print(f"Target: {target2}")
    print(f"Actual: {actual2}")
    print(f"Error: {error2*1000:.2f} mm")
    assert error2 < 0.003, f"Position error too large: {error2*1000:.2f} mm (threshold: 3mm)"
    
    # Test 3: Move back to initial position
    print(f"\nTest 3: Moving back to initial position {initial_pos}...")
    robot.move_to_position(initial_pos)
    actual3 = robot.get_position()
    error3 = np.linalg.norm(actual3 - initial_pos)
    print(f"Target: {initial_pos}")
    print(f"Actual: {actual3}")
    print(f"Error: {error3*1000:.2f} mm")
    assert error3 < 0.003, f"Position error too large: {error3*1000:.2f} mm (threshold: 3mm)"
    
    robot.move_to_start()
    print("\n✓ All absolute motion tests passed!")


def test_frame_consistency(robot: PandaController):
    """
    Test that move and move_to_position are consistent in world frame.
    
    This test verifies that:
    1. move([dx, dy, dz]) is equivalent to move_to_position(current + [dx, dy, dz])
    2. Both methods use world frame coordinates correctly
    """
    print("\n--- Testing Frame Consistency ---\n")
    
    robot.move_to_start()
    
    # Record starting position
    start_pos = robot.get_position().copy()
    print(f"Start position: {start_pos}")
    
    # Method 1: Use move() to go +X, +Y
    print("\nUsing move() for relative motion...")
    robot.move([0.05, 0.0, 0.0])
    robot.move([0.0, 0.05, 0.0])
    pos_after_move = robot.get_position().copy()
    print(f"Position after move(): {pos_after_move}")
    
    # Go back to start
    robot.move_to_position(start_pos)
    
    # Method 2: Use move_to_position() with calculated target
    print("\nUsing move_to_position() for same motion...")
    target = start_pos + np.array([0.05, 0.05, 0.0])
    robot.move_to_position(target)
    pos_after_move_to = robot.get_position().copy()
    print(f"Position after move_to_position(): {pos_after_move_to}")
    
    # Compare results
    diff = np.linalg.norm(pos_after_move - pos_after_move_to)
    print(f"\nDifference between methods: {diff*1000:.2f} mm")
    assert diff < 0.006, f"Methods not consistent: {diff*1000:.2f} mm difference (threshold: 6mm)"
    
    robot.move_to_start()
    print("\n✓ Frame consistency test passed!")


def test_streaming_move(robot: PandaController):
    """
    Test streaming relative motion (stream_move) in world frame.
    """
    print("\n--- Testing Streaming Relative Motion ---\n")
    
    robot.move_to_start()
    initial_pos = robot.get_position().copy()
    print(f"Initial position: {initial_pos}")
    
    # Move in a square pattern using streaming control
    print("\nMoving in a square pattern (5cm sides) using streaming...")
    
    with robot.streaming_context(frequency=30, max_runtime=8) as ctx:
        phase = 0
        target_offset = np.array([0.0, 0.0, 0.0])
        
        while ctx.ok():
            t = ctx.num_ticks / 30.0
            
            # Update target based on time
            if t < 2:
                target_offset = np.array([0.05 * (t / 2), 0.0, 0.0])  # +X
            elif t < 4:
                target_offset = np.array([0.05, 0.05 * ((t - 2) / 2), 0.0])  # +Y
            elif t < 6:
                target_offset = np.array([0.05 * (1 - (t - 4) / 2), 0.05, 0.0])  # -X
            else:
                target_offset = np.array([0.0, 0.05 * (1 - (t - 6) / 2), 0.0])  # -Y
            
            target_pos = initial_pos + target_offset
            robot.set_cartesian_target(target_pos, robot.get_orientation())
    
    final_pos = robot.get_position()
    error = np.linalg.norm(final_pos - initial_pos)
    print(f"Final position: {final_pos}")
    print(f"Error from start: {error*1000:.2f} mm")
    
    robot.move_to_start()
    print("\n✓ Streaming motion test completed!")


def run_all_tests():
    """Run all motion tests."""
    robot = PandaController()
    
    try:
        robot.connect()
        
        print("=" * 60)
        print("PANDA-PY MOTION TESTS")
        print("=" * 60)
        print("\nThese tests verify that move() and move_to_position() work")
        print("correctly in world frame (unlike franky which uses tool frame).")
        print("=" * 60)
        
        test_move_relative(robot)
        test_move_to_position(robot)
        test_frame_consistency(robot)
        test_streaming_move(robot)
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    finally:
        robot.disconnect()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Panda robot wrapper with tests")
    parser.add_argument(
        "--test",
        choices=["all", "move", "position", "consistency", "streaming", "demo"],
        default="demo",
        help="Which test to run (default: demo)",
    )
    args = parser.parse_args()
    
    robot = PandaController()
    
    try:
        robot.connect()
        
        if args.test == "all":
            run_all_tests()
        elif args.test == "move":
            test_move_relative(robot)
        elif args.test == "position":
            test_move_to_position(robot)
        elif args.test == "consistency":
            test_frame_consistency(robot)
        elif args.test == "streaming":
            test_streaming_move(robot)
        elif args.test == "demo":
            # Original demo code
            robot.move_to_start()
            
            print(f"Current position: {robot.get_position()}")
            print(f"Current orientation: {robot.get_orientation()}")
            
            x0 = robot.get_position()
            q0 = robot.get_orientation()
            
            print("Starting streaming control demo (5 seconds)")
            with robot.streaming_context(frequency=30, max_runtime=5) as ctx:
                while ctx.ok():
                    t = ctx.num_ticks / 30.0
                    x = x0.copy()
                    x[0] += 0.02 * np.sin(2 * np.pi * 0.5 * t)
                    x[1] += 0.02 * np.cos(2 * np.pi * 0.5 * t)
                    robot.set_cartesian_target(x, q0)
            
            print("Demo complete")
            robot.move_to_start()
    
    finally:
        robot.disconnect()
