import time
from panda.realsense_camera import RealSenseCamera, list_connected_cameras, reset_all_cameras

VISUALIZE = True
NUM_FRAMES = 30
FPS = 30

# Reset all cameras first to ensure clean state
reset_all_cameras()

# Discover available cameras
camera_info = list_connected_cameras()
num_cameras = len(camera_info)
print(f"Found {num_cameras} camera(s)")
for i, cam in enumerate(camera_info):
    print(f"  [{i}] Serial: {cam['serial_number']}, Name: {cam['name']}")

if num_cameras == 0:
    print("No cameras found. Exiting.")
    exit(1)

# Create and start camera instances
cameras = []
for cam in camera_info:
    camera = RealSenseCamera(serial_number=cam['serial_number'])
    camera.start()
    cameras.append(camera)

frame_buffer = []  # List of frames, each frame is a list of (color, depth) tuples per camera

# Capture from all cameras
start_time = time.time()
for i in range(NUM_FRAMES):
    loop_start = time.time()
    
    # Capture from all cameras
    frame = []
    for camera in cameras:
        color, depth = camera.get_frames()
        frame.append((color.copy(), depth.copy()))
    frame_buffer.append(frame)
    
    # Maintain target FPS
    elapsed = time.time() - loop_start
    sleep_time = max(0, (1.0 / FPS) - elapsed)
    if sleep_time > 0:
        time.sleep(sleep_time)

end_time = time.time()
print(f"Time taken for {NUM_FRAMES} frames: {end_time - start_time:.2f} seconds")
print(f"Frame buffer length: {len(frame_buffer)}")

# Print frame shapes for each camera
for i, (color, depth) in enumerate(frame_buffer[0]):
    print(f"Camera {i}: color={color.shape if color is not None else None}, depth={depth.shape if depth is not None else None}")

if VISUALIZE:
    import cv2
    import numpy as np

    # Get the first frame from all cameras
    first_frames = frame_buffer[0]
    
    # Create a grid of images
    color_images = []
    for i, (color, depth) in enumerate(first_frames):
        if color is not None:
            # Add camera label to image
            labeled = color.copy()
            cv2.putText(labeled, f"Camera {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color_images.append(labeled)
    
    if color_images:
        # Stack images horizontally (or in a grid if many cameras)
        if len(color_images) <= 4:
            combined = np.hstack(color_images)
        else:
            # Create 2-row grid for more than 4 cameras
            half = (len(color_images) + 1) // 2
            row1 = color_images[:half]
            row2 = color_images[half:]
            # Pad row2 if needed
            if len(row2) < len(row1):
                row2.append(np.zeros_like(row1[0]))
            combined = np.vstack([np.hstack(row1), np.hstack(row2)])
        
        cv2.imshow("All Cameras - First Frame", combined)
        print("Press any key in the image window to exit visualization...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image data available to display.")

# Stop all cameras
for camera in cameras:
    camera.stop(hardware_reset=False)
