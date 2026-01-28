"""
Xbox Controller for Franka Panda Robot (panda-py version)

Uses an Xbox controller to teleoperate the Franka Panda robot arm.

Controller Mapping:
    Left Stick X/Y  -> Robot X/Y movement (world frame)
    Right Stick Y   -> Robot Z movement (world frame)
    LB (hold)       -> Close gripper (incremental)
    RB (hold)       -> Open gripper (incremental)
    A Button        -> Toggle gripper (full open/close)
    B Button        -> Stop motion
    X Button        -> Go to home position
    Y Button        -> Print robot status
    Start Button    -> Exit
    Back Button     -> Recover from errors (not supported in panda-py)

Requirements:
    pip install pygame panda-py
"""

import time
from typing import Optional, Tuple

import numpy as np

try:
    import pygame
except ImportError:
    print("pygame not installed. Install with: pip install pygame")
    exit(1)

from panda_controller import PandaController


class XboxController:
    """
    Xbox controller interface using pygame.
    
    Handles controller initialization, reading inputs, and providing
    normalized values for joysticks and triggers.
    """
    
    # Xbox controller button indices (may vary by controller/driver)
    BUTTON_A = 0
    BUTTON_B = 1
    BUTTON_X = 2
    BUTTON_Y = 3
    BUTTON_LB = 4
    BUTTON_RB = 5
    BUTTON_BACK = 6
    BUTTON_START = 7
    BUTTON_GUIDE = 8  # Xbox button (may not be available)
    BUTTON_L_STICK = 9
    BUTTON_R_STICK = 10
    
    # Axis indices
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_TRIGGER_LEFT = 2
    AXIS_TRIGGER_RIGHT = 5
    
    def __init__(self, controller_id: int = 0, deadzone: float = 0.15):
        """
        Initialize Xbox controller.
        
        Args:
            controller_id: Index of the controller (0 for first controller).
            deadzone: Minimum joystick value to register (prevents drift).
        """
        self.controller_id = controller_id
        self.deadzone = deadzone
        self.joystick = None
        
        # Initialize pygame
        pygame.init()
        pygame.joystick.init()
        
        # Check for controllers
        num_joysticks = pygame.joystick.get_count()
        if num_joysticks == 0:
            raise RuntimeError("No controllers found. Please connect an Xbox controller.")
        
        print(f"Found {num_joysticks} controller(s)")
        
        # Initialize the specified controller
        self.joystick = pygame.joystick.Joystick(controller_id)
        self.joystick.init()
        
        print(f"Initialized controller: {self.joystick.get_name()}")
        print(f"  Axes: {self.joystick.get_numaxes()}")
        print(f"  Buttons: {self.joystick.get_numbuttons()}")
        print(f"  Hats: {self.joystick.get_numhats()}")
    
    def _apply_deadzone(self, value: float) -> float:
        """Apply deadzone to joystick value."""
        if abs(value) < self.deadzone:
            return 0.0
        # Scale value to maintain full range after deadzone
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)
    
    def update(self) -> None:
        """Process pygame events. Call this each frame."""
        pygame.event.pump()
    
    def get_left_stick(self) -> Tuple[float, float]:
        """
        Get left stick position.
        
        Returns:
            (x, y) where x is left(-1) to right(+1), y is up(-1) to down(+1)
        """
        x = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_X))
        y = self._apply_deadzone(self.joystick.get_axis(self.AXIS_LEFT_Y))
        return (x, y)
    
    def get_right_stick(self) -> Tuple[float, float]:
        """
        Get right stick position.
        
        Returns:
            (x, y) where x is left(-1) to right(+1), y is up(-1) to down(+1)
        """
        x = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_X))
        y = self._apply_deadzone(self.joystick.get_axis(self.AXIS_RIGHT_Y))
        return (x, y)
    
    def get_triggers(self) -> Tuple[float, float]:
        """
        Get trigger values.
        
        Returns:
            (left_trigger, right_trigger) each in range [0, 1]
        
        Note: On some systems, triggers may be in range [-1, 1], this normalizes to [0, 1]
        """
        # Triggers may report as -1 (released) to 1 (pressed) or 0 to 1
        left = self.joystick.get_axis(self.AXIS_TRIGGER_LEFT)
        right = self.joystick.get_axis(self.AXIS_TRIGGER_RIGHT)
        
        # Normalize to [0, 1] if needed
        left = (left + 1) / 2 if left < 0 else left
        right = (right + 1) / 2 if right < 0 else right
        
        return (max(0, left), max(0, right))
    
    def get_button(self, button_id: int) -> bool:
        """Check if a button is pressed."""
        return self.joystick.get_button(button_id)
    
    def get_button_a(self) -> bool:
        return self.get_button(self.BUTTON_A)
    
    def get_button_b(self) -> bool:
        return self.get_button(self.BUTTON_B)
    
    def get_button_x(self) -> bool:
        return self.get_button(self.BUTTON_X)
    
    def get_button_y(self) -> bool:
        return self.get_button(self.BUTTON_Y)
    
    def get_button_start(self) -> bool:
        return self.get_button(self.BUTTON_START)
    
    def get_button_back(self) -> bool:
        return self.get_button(self.BUTTON_BACK)
    
    def get_button_lb(self) -> bool:
        return self.get_button(self.BUTTON_LB)
    
    def get_button_rb(self) -> bool:
        return self.get_button(self.BUTTON_RB)
    
    def close(self) -> None:
        """Clean up pygame resources."""
        pygame.quit()


class XboxRobotController:
    """
    Maps Xbox controller inputs to Franka Panda robot movements using panda-py.
    
    Control Scheme:
        Left Stick X    -> Move in world Y direction (left/right)
        Left Stick Y    -> Move in world X direction (forward/backward)
        Right Stick Y   -> Move in world Z direction (up/down)
        LB (hold)       -> Close gripper (incremental)
        RB (hold)       -> Open gripper (incremental)
        A Button        -> Toggle gripper (full open/close)
        B Button        -> Stop motion
        X Button        -> Go home
        Y Button        -> Print status
        Start           -> Exit
        Back            -> (Reserved)
    """
    
    def __init__(
        self,
        max_velocity: float = 0.05,  # m/s max velocity
        control_rate: float = 20.0,  # Hz
    ):
        """
        Initialize Xbox robot controller.
        
        Args:
            max_velocity: Maximum velocity in m/s.
            control_rate: Control loop frequency in Hz.
        """
        self.max_velocity = max_velocity
        self.control_rate = control_rate
        self.dt = 1.0 / control_rate
        
        # Maximum displacement per control cycle
        self.max_step = max_velocity * self.dt
        
        # Initialize Xbox controller
        print("Initializing Xbox controller...")
        self.xbox = XboxController()
        
        # Initialize robot
        print("Initializing robot...")
        self.robot = PandaController()
        self.robot.connect()
        
        # Connect gripper
        try:
            self.robot.connect_gripper()
            self._gripper_available = True
        except Exception as e:
            print(f"Warning: Could not connect gripper: {e}")
            self._gripper_available = False
        
        # State
        self.running = False
        self.gripper_open = True
        self.current_gripper_width = 0.08  # Start assuming open (8cm)
        
        # Gripper control parameters
        self.gripper_speed = 0.02  # m/s for smooth control
        self.gripper_step = self.gripper_speed * self.dt  # Width change per cycle
        
        # Button debouncing
        self._last_button_time = {}
        self._button_cooldown = 0.3  # seconds
        
        # Current target position/orientation (for streaming)
        self._target_position = None
        self._target_orientation = None
        
        print("\nXbox Robot Controller initialized!")
        self._print_controls()
    
    def _print_controls(self) -> None:
        """Print control scheme."""
        print("\n" + "=" * 50)
        print("CONTROL SCHEME")
        print("=" * 50)
        print("Left Stick      : Move X/Y (forward/back, left/right)")
        print("Right Stick Y   : Move Z (up/down)")
        print("-" * 50)
        print("LB (hold)       : Close gripper (incremental)")
        print("RB (hold)       : Open gripper (incremental)")
        print("A Button        : Toggle gripper (full open/close)")
        print("-" * 50)
        print("B Button        : Stop motion")
        print("X Button        : Go to home position")
        print("Y Button        : Print robot status")
        print("Start Button    : Exit")
        print("=" * 50 + "\n")
    
    def _is_button_ready(self, button_name: str) -> bool:
        """Check if button is ready (debouncing)."""
        current_time = time.time()
        last_time = self._last_button_time.get(button_name, 0)
        if current_time - last_time > self._button_cooldown:
            self._last_button_time[button_name] = current_time
            return True
        return False
    
    def _get_movement_from_sticks(self) -> Tuple[float, float, float]:
        """
        Convert joystick inputs to robot movement.
        
        Returns:
            (dx, dy, dz) displacement in world frame (meters)
        """
        # Get stick values
        left_x, left_y = self.xbox.get_left_stick()
        right_x, right_y = self.xbox.get_right_stick()
        
        # Map sticks to world frame movement
        # Left stick Y (forward/back) -> world X
        # Left stick X (left/right) -> world Y
        # Right stick Y (up/down) -> world Z
        dx = left_y * self.max_step  # Forward is -Y on stick, +X in world
        dy = left_x * self.max_step   # Right is +X on stick, +Y in world
        dz = -right_y * self.max_step  # Up is -Y on stick, +Z in world
        
        return (dx, dy, dz)
    
    def _handle_gripper_bumpers(self) -> None:
        """
        Handle incremental gripper control via LB/RB bumpers.
        
        LB (hold): Close gripper gradually
        RB (hold): Open gripper gradually
        """
        if not self._gripper_available:
            return
            
        lb_pressed = self.xbox.get_button_lb()
        rb_pressed = self.xbox.get_button_rb()
        
        if lb_pressed and not rb_pressed:
            # Close gripper gradually
            self.current_gripper_width = max(0.0, self.current_gripper_width - self.gripper_step)
            try:
                self.robot.gripper.move(self.current_gripper_width, self.gripper_speed)
            except Exception:
                pass
        elif rb_pressed and not lb_pressed:
            # Open gripper gradually
            self.current_gripper_width = min(0.08, self.current_gripper_width + self.gripper_step)
            try:
                self.robot.gripper.move(self.current_gripper_width, self.gripper_speed)
            except Exception:
                pass
    
    def _handle_buttons(self) -> bool:
        """
        Handle button presses.
        
        Returns:
            False if should exit, True otherwise.
        """
        # A Button - Toggle gripper
        if self.xbox.get_button_a() and self._is_button_ready('A'):
            if self._gripper_available:
                print("Toggle gripper")
                if self.gripper_open:
                    self.robot.gripper_close()
                    self.gripper_open = False
                    self.current_gripper_width = 0.0
                else:
                    self.robot.gripper_open()
                    self.gripper_open = True
                    self.current_gripper_width = 0.08
        
        # B Button - Stop (reset target to current position)
        if self.xbox.get_button_b() and self._is_button_ready('B'):
            print("Stop - resetting target to current position")
            self._target_position = np.array(self.robot.get_position())
            self._target_orientation = np.array(self.robot.get_orientation())
        
        # X Button - Go home (need to stop streaming first)
        if self.xbox.get_button_x() and self._is_button_ready('X'):
            print("Going home...")
            # Stop streaming, go home, then restart
            self.robot.stop_streaming()
            self.robot.move_to_start()
            # Update targets to home position (use np.array for true copy)
            self._target_position = np.array(self.robot.get_position())
            self._target_orientation = np.array(self.robot.get_orientation())
            # Restart streaming
            self.robot.start_streaming()
        
        # Y Button - Print status
        if self.xbox.get_button_y() and self._is_button_ready('Y'):
            self._print_status()
        
        # Start Button - Exit
        if self.xbox.get_button_start():
            print("\nStart pressed - Exiting...")
            return False
        
        return True
    
    def _print_status(self) -> None:
        """Print robot status."""
        pos = self.robot.get_position()
        ori = self.robot.get_orientation()
        joints = self.robot.get_joint_positions()
        
        print("\n" + "=" * 50)
        print("ROBOT STATUS")
        print("=" * 50)
        print(f"Position (m):    [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
        print(f"Orientation (q): [{ori[0]:.4f}, {ori[1]:.4f}, {ori[2]:.4f}, {ori[3]:.4f}]")
        print(f"Joint positions: {np.round(np.degrees(joints), 1)} deg")
        print(f"Gripper width:   {self.current_gripper_width * 1000:.1f} mm")
        print("=" * 50 + "\n")
    
    def run(self) -> None:
        """Main control loop."""
        print("\nStarting control loop. Press Start to exit.\n")
        self.running = True
        
        # Move to start and get initial position
        self.robot.move_to_start()
        # Use np.array() to ensure true copy (panda-py may return buffer references)
        self._target_position = np.array(self.robot.get_position())
        self._target_orientation = np.array(self.robot.get_orientation())
        
        last_print_time = time.time()
        print_interval = 1.0  # Print status every second
        
        # Start streaming control with lower filter coefficient for smoother motion
        # filter_coeff: lower = more smoothing (0.0-1.0)
        self.robot.start_streaming()
        
        try:
            while self.running:
                loop_start = time.time()
                
                # Update controller state
                self.xbox.update()
                
                # Handle buttons
                if not self._handle_buttons():
                    break
                
                # Get movement from sticks
                dx, dy, dz = self._get_movement_from_sticks()
                
                # Update target position
                self._target_position[0] += dx
                self._target_position[1] += dy
                self._target_position[2] += dz
                
                # Clamp Z to safe range (don't go too low or too high)
                self._target_position[2] = np.clip(self._target_position[2], 0.05, 0.8)
                
                # Send target to robot
                self.robot.set_cartesian_target(self._target_position, self._target_orientation)
                
                # Handle gripper bumpers (LB close, RB open)
                self._handle_gripper_bumpers()
                
                # Periodic status print
                current_time = time.time()
                if current_time - last_print_time > print_interval:
                    pos = self.robot.get_position()
                    gripper_w = self.current_gripper_width * 1000  # mm
                    print(f"Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] | "
                          f"Target: [{self._target_position[0]:.3f}, {self._target_position[1]:.3f}, {self._target_position[2]:.3f}] | "
                          f"Gripper: {gripper_w:.1f}mm")
                    last_print_time = current_time
                
                # Maintain control rate
                elapsed = time.time() - loop_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt - stopping...")
        
        finally:
            self.running = False
            self.robot.stop_streaming()
            self.xbox.close()
            self.robot.disconnect()
            print("Controller stopped.")
    
    def close(self) -> None:
        """Clean up resources."""
        self.running = False
        self.xbox.close()
        self.robot.disconnect()


# ==================== Test Functions ====================

def test_controller_input():
    """Test Xbox controller input without robot connection."""
    print("\n" + "=" * 50)
    print("XBOX CONTROLLER INPUT TEST")
    print("=" * 50)
    print("Testing controller inputs. Press Ctrl+C to exit.\n")
    
    try:
        controller = XboxController()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    print("Move the sticks and press buttons to see values.\n")
    
    try:
        while True:
            controller.update()
            
            # Get all inputs
            left_x, left_y = controller.get_left_stick()
            right_x, right_y = controller.get_right_stick()
            left_trigger, right_trigger = controller.get_triggers()
            
            # Get button states
            buttons = {
                'A': controller.get_button_a(),
                'B': controller.get_button_b(),
                'X': controller.get_button_x(),
                'Y': controller.get_button_y(),
                'LB': controller.get_button_lb(),
                'RB': controller.get_button_rb(),
                'Start': controller.get_button_start(),
                'Back': controller.get_button_back(),
            }
            
            # Format button string
            pressed_buttons = [name for name, pressed in buttons.items() if pressed]
            button_str = ', '.join(pressed_buttons) if pressed_buttons else 'None'
            
            # Print status
            print(f"\rL:[{left_x:+.2f},{left_y:+.2f}] "
                  f"R:[{right_x:+.2f},{right_y:+.2f}] "
                  f"Triggers:[{left_trigger:.2f},{right_trigger:.2f}] "
                  f"Buttons:[{button_str}]" + " " * 20, end='')
            
            # Exit on Start button
            if controller.get_button_start():
                print("\n\nStart pressed - exiting.")
                break
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    
    finally:
        controller.close()
        print("Controller test completed.")


def test_controller_mapping():
    """Test how controller inputs map to robot movements (simulation only)."""
    print("\n" + "=" * 50)
    print("CONTROLLER MAPPING TEST (No Robot)")
    print("=" * 50)
    print("Shows how stick inputs would map to robot movements.")
    print("Press Ctrl+C or Start to exit.\n")
    
    try:
        controller = XboxController()
    except RuntimeError as e:
        print(f"Error: {e}")
        return
    
    max_velocity = 0.05  # m/s
    control_rate = 20.0  # Hz
    max_step = max_velocity / control_rate
    
    print(f"Max velocity: {max_velocity} m/s")
    print(f"Control rate: {control_rate} Hz")
    print(f"Max step per cycle: {max_step * 1000:.2f} mm\n")
    
    try:
        while True:
            controller.update()
            
            # Get stick values
            left_x, left_y = controller.get_left_stick()
            right_x, right_y = controller.get_right_stick()
            
            # Calculate movement (same as XboxRobotController)
            dx = -left_y * max_step  # Forward
            dy = left_x * max_step   # Right
            dz = -right_y * max_step  # Up
            
            # Print mapping
            print(f"\rSticks: L[{left_x:+.2f},{left_y:+.2f}] R[{right_x:+.2f},{right_y:+.2f}] "
                  f"-> Move (mm): [{dx*1000:+.2f}, {dy*1000:+.2f}, {dz*1000:+.2f}]" + " " * 10, end='')
            
            if controller.get_button_start():
                print("\n\nStart pressed - exiting.")
                break
            
            time.sleep(0.05)
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
    
    finally:
        controller.close()
        print("Mapping test completed.")


def test_robot_control():
    """Test full Xbox robot control."""
    print("\n" + "=" * 50)
    print("XBOX ROBOT CONTROL TEST")
    print("=" * 50)
    
    try:
        controller = XboxRobotController(
            max_velocity=0.1,  # 5 cm/s max velocity
            control_rate=20.0,  # 20 Hz control rate
        )
        
        print("\nStarting control test...")
        controller.run()
        
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


def run_all_controller_tests():
    """Run all controller tests."""
    print("=" * 60)
    print("XBOX CONTROLLER TEST SUITE")
    print("=" * 60)
    print("\nAvailable tests:")
    print("  1. Input test (no robot) - Test controller inputs")
    print("  2. Mapping test (no robot) - See stick-to-movement mapping")
    print("  3. Robot control test - Full robot teleoperation")
    print()
    
    choice = input("Select test (1/2/3) or 'q' to quit: ").strip()
    
    if choice == '1':
        test_controller_input()
    elif choice == '2':
        test_controller_mapping()
    elif choice == '3':
        test_robot_control()
    elif choice.lower() == 'q':
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Xbox Controller for Franka Panda Robot")
    parser.add_argument(
        "--test",
        choices=["all", "input", "mapping", "robot"],
        default="all",
        help="Which test to run (default: all - shows menu)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="Max velocity in m/s (default: 0.05)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Control loop rate in Hz (default: 20)",
    )
    args = parser.parse_args()
    
    if args.test == "all":
        run_all_controller_tests()
    elif args.test == "input":
        test_controller_input()
    elif args.test == "mapping":
        test_controller_mapping()
    elif args.test == "robot":
        test_robot_control()
