"""
Intel RealSense Camera Wrapper

A wrapper for Intel RealSense cameras (D400 series) using pyrealsense2.
Provides easy access to color images, depth images, and aligned depth-color frames.
"""

import time
from typing import Optional, List, Tuple
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise ImportError(
        "pyrealsense2 is not installed. Install it with: pip install pyrealsense2"
    )


class RealSenseCamera:
    """
    A wrapper class for Intel RealSense depth cameras.
    
    Provides simplified interfaces for:
    - Capturing color and depth images
    - Getting aligned depth-color frames
    - Retrieving camera intrinsics
    - Converting pixel coordinates to 3D points
    """
    
    # Default stream configurations
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    DEFAULT_FPS = 30
    
    def __init__(
        self,
        serial_number: Optional[str] = None,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        enable_color: bool = True,
        enable_depth: bool = True,
        align_to_color: bool = True,
    ):
        """
        Initialize the RealSense camera.
        
        Args:
            serial_number: Specific camera serial number to connect to.
                          If None, connects to the first available camera.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Frames per second.
            enable_color: Enable color stream.
            enable_depth: Enable depth stream.
            align_to_color: Align depth frames to color frames.
        """
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_to_color = align_to_color
        
        # Pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # If specific serial number provided
        if serial_number:
            self.config.enable_device(serial_number)
        
        # Configure streams
        if enable_color:
            self.config.enable_stream(
                rs.stream.color, width, height, rs.format.bgr8, fps
            )
        
        if enable_depth:
            self.config.enable_stream(
                rs.stream.depth, width, height, rs.format.z16, fps
            )
        
        # Alignment object
        self.align = None
        if align_to_color and enable_depth and enable_color:
            self.align = rs.align(rs.stream.color)
        
        # Filters for depth processing
        self.decimation_filter = rs.decimation_filter()
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter()
        
        # Camera intrinsics (set after start)
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        
        # State
        self._is_running = False
        self._profile = None
    
    def start(self, warmup_timeout_ms: int = 15000) -> None:
        """Start the camera pipeline.
        
        Args:
            warmup_timeout_ms: Timeout in milliseconds for initial frame capture.
                              Use longer timeout (15000+) for multi-camera setups.
        """
        if self._is_running:
            print("Camera is already running.")
            return
        
        try:
            self._profile = self.pipeline.start(self.config)
            self._is_running = True
            
            # Get depth scale
            if self.enable_depth:
                depth_sensor = self._profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
            
            # Allow camera to warm up and get intrinsics
            time.sleep(1.0)  # Longer warmup for multi-camera setups
            frames = self.pipeline.wait_for_frames(warmup_timeout_ms)
            
            if self.align and self.enable_color and self.enable_depth:
                frames = self.align.process(frames)
            
            # Get intrinsics
            if self.enable_color:
                color_frame = frames.get_color_frame()
                if color_frame:
                    self.color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            
            if self.enable_depth:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            
            device = self._profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            
            print(f"RealSense camera started: {device_name} (S/N: {device_serial})")
            print(f"  Resolution: {self.width}x{self.height} @ {self.fps}fps")
            if self.depth_scale:
                print(f"  Depth scale: {self.depth_scale:.6f} m/unit")
            
        except Exception as e:
            self._is_running = False
            raise RuntimeError(f"Failed to start RealSense camera: {e}")
    
    def stop(self, hardware_reset: bool = True) -> None:
        """Stop the camera pipeline.
        
        Args:
            hardware_reset: If True, perform a hardware reset after stopping.
                           This ensures the USB device is in a clean state for the next run.
        """
        if self._is_running:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"Warning during pipeline stop: {e}")
            finally:
                self._is_running = False
            print(f"RealSense camera stopped (S/N: {self.serial_number or 'auto'}).")
            
            # Hardware reset to ensure clean state for next run
            if hardware_reset:
                try:
                    ctx = rs.context()
                    devices = ctx.query_devices()
                    for dev in devices:
                        serial = dev.get_info(rs.camera_info.serial_number)
                        # Reset only this specific camera, or the first one if no serial was specified
                        if self.serial_number is None or serial == self.serial_number:
                            dev.hardware_reset()
                            time.sleep(0.5)  # Wait for reset to complete
                            break
                except Exception as e:
                    print(f"Warning during hardware reset: {e}")
    
    def is_running(self) -> bool:
        """Check if the camera is running."""
        return self._is_running
    
    def get_frames(
        self,
        apply_filters: bool = False,
        timeout_ms: int = 5000,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture a single frame set from the camera.
        
        Args:
            apply_filters: Apply depth filtering (spatial, temporal, hole filling).
            timeout_ms: Timeout in milliseconds to wait for frames.
        
        Returns:
            Tuple of (color_image, depth_image) as numpy arrays.
            color_image: BGR format, shape (H, W, 3), dtype uint8
            depth_image: Raw depth values, shape (H, W), dtype uint16
            Either can be None if that stream is disabled.
        """
        if not self._is_running:
            raise RuntimeError("Camera is not running. Call start() first.")
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            print("Timeout waiting for frames.")
            return None, None
        
        # Align frames if enabled
        if self.align:
            frames = self.align.process(frames)
        
        color_image = None
        depth_image = None
        
        # Get color frame
        if self.enable_color:
            color_frame = frames.get_color_frame()
            if color_frame:
                color_image = np.asanyarray(color_frame.get_data())
        
        # Get depth frame
        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            
            if depth_frame and apply_filters:
                depth_frame = self.decimation_filter.process(depth_frame)
                depth_frame = self.spatial_filter.process(depth_frame)
                depth_frame = self.temporal_filter.process(depth_frame)
                depth_frame = self.hole_filling_filter.process(depth_frame)
            
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
        
        return color_image, depth_image
    
    def get_color_image(self) -> Optional[np.ndarray]:
        """
        Get only the color image.
        
        Returns:
            Color image as numpy array (BGR format), or None if unavailable.
        """
        color, _ = self.get_frames()
        return color
    
    def get_depth_image(self, apply_filters: bool = False) -> Optional[np.ndarray]:
        """
        Get only the depth image.
        
        Args:
            apply_filters: Apply depth filtering.
        
        Returns:
            Depth image as numpy array (raw uint16), or None if unavailable.
        """
        _, depth = self.get_frames(apply_filters=apply_filters)
        return depth
    
    def get_depth_meters(self, apply_filters: bool = False) -> Optional[np.ndarray]:
        """
        Get depth image in meters.
        
        Args:
            apply_filters: Apply depth filtering.
        
        Returns:
            Depth image as numpy array (float32, meters), or None if unavailable.
        """
        depth = self.get_depth_image(apply_filters=apply_filters)
        if depth is not None and self.depth_scale is not None:
            return depth.astype(np.float32) * self.depth_scale
        return None
    
    def get_depth_at_pixel(self, x: int, y: int) -> float:
        """
        Get the depth value at a specific pixel coordinate.
        
        Args:
            x: X coordinate (column) in pixels.
            y: Y coordinate (row) in pixels.
        
        Returns:
            Depth value in meters.
        """
        depth = self.get_depth_image()
        if depth is None:
            return 0.0
        
        # Clamp coordinates
        y = max(0, min(y, depth.shape[0] - 1))
        x = max(0, min(x, depth.shape[1] - 1))
        
        return depth[y, x] * self.depth_scale
    
    def pixel_to_point_3d(
        self,
        x: int,
        y: int,
        depth_value: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Convert a pixel coordinate to a 3D point in camera frame.
        
        The camera frame follows the RealSense convention:
        - X: right (positive)
        - Y: down (positive)
        - Z: forward/into scene (positive)
        
        Args:
            x: X coordinate (column) in pixels.
            y: Y coordinate (row) in pixels.
            depth_value: Depth value in meters. If None, reads from current frame.
        
        Returns:
            Tuple (X, Y, Z) in meters (camera frame).
        """
        if self.depth_intrinsics is None:
            raise RuntimeError("Depth intrinsics not available. Start the camera first.")
        
        if depth_value is None:
            depth_value = self.get_depth_at_pixel(x, y)
        
        if depth_value <= 0:
            return (0.0, 0.0, 0.0)
        
        # Use RealSense deprojection
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.depth_intrinsics, [x, y], depth_value
        )
        
        return tuple(point_3d)
    
    def point_3d_to_pixel(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[int, int]:
        """
        Project a 3D point (camera frame) to pixel coordinates.
        
        Args:
            x, y, z: 3D point in camera frame (meters).
        
        Returns:
            Tuple (px, py) pixel coordinates.
        """
        if self.color_intrinsics is None:
            raise RuntimeError("Color intrinsics not available. Start the camera first.")
        
        pixel = rs.rs2_project_point_to_pixel(
            self.color_intrinsics, [x, y, z]
        )
        
        return (int(pixel[0]), int(pixel[1]))
    
    def get_intrinsics(self) -> dict:
        """
        Get camera intrinsic parameters.
        
        Returns:
            Dictionary with intrinsic parameters for color and depth cameras.
        """
        result = {}
        
        if self.color_intrinsics:
            result['color'] = {
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height,
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy,
                'model': str(self.color_intrinsics.model),
                'coeffs': list(self.color_intrinsics.coeffs),
            }
        
        if self.depth_intrinsics:
            result['depth'] = {
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy,
                'model': str(self.depth_intrinsics.model),
                'coeffs': list(self.depth_intrinsics.coeffs),
            }
        
        if self.depth_scale:
            result['depth_scale'] = self.depth_scale
        
        return result
    
    def get_intrinsics_matrix(self, stream: str = 'color') -> np.ndarray:
        """
        Get the camera intrinsics as a 3x3 matrix.
        
        Args:
            stream: 'color' or 'depth'.
        
        Returns:
            3x3 intrinsics matrix K:
            [[fx,  0, ppx],
             [ 0, fy, ppy],
             [ 0,  0,   1]]
        """
        if stream == 'color' and self.color_intrinsics:
            intr = self.color_intrinsics
        elif stream == 'depth' and self.depth_intrinsics:
            intr = self.depth_intrinsics
        else:
            raise ValueError(f"Intrinsics not available for stream: {stream}")
        
        return np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def save_image(
        self,
        filepath: str,
        image_type: str = 'color',
    ) -> bool:
        """
        Save an image to file.
        
        Args:
            filepath: Path to save the image.
            image_type: 'color' or 'depth'.
        
        Returns:
            True if successful.
        """
        try:
            import cv2
        except ImportError:
            raise ImportError("OpenCV is required for saving images: pip install opencv-python")
        
        color, depth = self.get_frames()
        
        if image_type == 'color' and color is not None:
            cv2.imwrite(filepath, color)
            print(f"Color image saved to: {filepath}")
            return True
        elif image_type == 'depth' and depth is not None:
            # Normalize depth for visualization
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(filepath, depth_colored)
            print(f"Depth image saved to: {filepath}")
            return True
        
        print(f"Failed to save {image_type} image.")
        return False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
    
    def __del__(self):
        """Destructor to ensure camera is stopped."""
        if self._is_running:
            self.stop()


# ==================== Utility Functions ====================

def list_connected_cameras() -> List[dict]:
    """
    List all connected RealSense cameras.
    
    Returns:
        List of dictionaries with camera information.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    cameras = []
    for dev in devices:
        info = {
            'name': dev.get_info(rs.camera_info.name),
            'serial_number': dev.get_info(rs.camera_info.serial_number),
            'firmware_version': dev.get_info(rs.camera_info.firmware_version),
            'product_id': dev.get_info(rs.camera_info.product_id),
        }
        cameras.append(info)
    
    return cameras


def print_connected_cameras() -> None:
    """Print information about all connected RealSense cameras."""
    cameras = list_connected_cameras()
    
    if not cameras:
        print("No RealSense cameras found.")
        return
    
    print(f"\nFound {len(cameras)} RealSense camera(s):")
    print("-" * 50)
    
    for i, cam in enumerate(cameras):
        print(f"Camera {i + 1}:")
        print(f"  Name: {cam['name']}")
        print(f"  Serial: {cam['serial_number']}")
        print(f"  Firmware: {cam['firmware_version']}")
        print(f"  Product ID: {cam['product_id']}")
    
    print("-" * 50)


def reset_all_cameras(wait_time: float = 2.0) -> None:
    """
    Hardware reset all connected RealSense cameras.
    
    Useful to call before starting multiple cameras to ensure clean state.
    
    Args:
        wait_time: Time to wait after reset for cameras to reinitialize.
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense cameras found to reset.")
        return
    
    print(f"Resetting {len(devices)} camera(s)...")
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"  Resetting S/N: {serial}")
        dev.hardware_reset()
    
    print(f"Waiting {wait_time}s for cameras to reinitialize...")
    time.sleep(wait_time)
    print("Reset complete.")


# ==================== Test Functions ====================

def test_basic_capture():
    """Test basic frame capture."""
    print("\n--- Testing Basic Capture ---\n")
    
    with RealSenseCamera() as camera:
        print("Capturing 10 frames...")
        for i in range(10):
            color, depth = camera.get_frames()
            if color is not None and depth is not None:
                print(f"Frame {i + 1}: Color {color.shape}, Depth {depth.shape}")
            time.sleep(0.1)


def test_3d_point():
    """Test pixel to 3D point conversion."""
    print("\n--- Testing 3D Point Conversion ---\n")
    
    with RealSenseCamera() as camera:
        # Get center pixel
        cx, cy = camera.width // 2, camera.height // 2
        
        # Get depth at center
        depth = camera.get_depth_at_pixel(cx, cy)
        print(f"Center pixel ({cx}, {cy}): depth = {depth:.3f}m")
        
        # Convert to 3D point
        point_3d = camera.pixel_to_point_3d(cx, cy)
        print(f"3D point: X={point_3d[0]:.3f}, Y={point_3d[1]:.3f}, Z={point_3d[2]:.3f}")


def test_intrinsics():
    """Test intrinsics retrieval."""
    print("\n--- Testing Intrinsics ---\n")
    
    with RealSenseCamera() as camera:
        intrinsics = camera.get_intrinsics()
        print("Color intrinsics:")
        if 'color' in intrinsics:
            for key, value in intrinsics['color'].items():
                print(f"  {key}: {value}")
        
        print("\nIntrinsics matrix:")
        K = camera.get_intrinsics_matrix('color')
        print(K)


def test_live_display():
    """Test live display with OpenCV."""
    print("\n--- Testing Live Display ---\n")
    print("Press 'q' to quit (in the OpenCV window), 's' to save frame")
    print("Or press Ctrl+C in terminal to exit")
    
    try:
        import cv2
    except ImportError:
        print("OpenCV is required for live display: pip install opencv-python")
        return
    
    import signal
    
    # Flag for clean shutdown
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\nCtrl+C detected, shutting down...")
        running = False
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    camera = RealSenseCamera()
    try:
        camera.start()
        
        while running:
            try:
                color, depth = camera.get_frames(timeout_ms=1000)
            except RuntimeError:
                # Timeout - check if we should still be running
                continue
            
            if color is None:
                continue
            
            # Create depth colormap
            if depth is not None:
                depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_colored = cv2.applyColorMap(
                    depth_normalized.astype(np.uint8), cv2.COLORMAP_JET
                )
                
                # Stack horizontally
                display = np.hstack([color, depth_colored])
            else:
                display = color
            
            cv2.imshow('RealSense', display)
            
            # Wait longer (30ms) for better key detection
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("'q' pressed, exiting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                camera.save_image(f'color_{timestamp}.jpg', 'color')
                camera.save_image(f'depth_{timestamp}.jpg', 'depth')
            
            # Also check if window was closed
            if cv2.getWindowProperty('RealSense', cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed, exiting...")
                break
    
    finally:
        # Ensure proper cleanup
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Process the destroy event
        camera.stop()
        print("Cleanup complete.")


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description="RealSense Camera Test")
    parser.add_argument(
        "--test",
        choices=["list", "basic", "3d", "intrinsics", "live"],
        default="list",
        help="Which test to run (default: list)",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Camera serial number to use (use --test list to see available cameras)",
    )
    args = parser.parse_args()
    
    if args.test == "list":
        print_connected_cameras()
    elif args.test == "basic":
        test_basic_capture()
    elif args.test == "3d":
        test_3d_point()
    elif args.test == "intrinsics":
        test_intrinsics()
    elif args.test == "live":
        test_live_display()
