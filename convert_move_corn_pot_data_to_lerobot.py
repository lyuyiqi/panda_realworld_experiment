"""

Usage:
uv run data/convert_move_corn_pot_data_to_lerobot.py --data_dir /root/openpi/data/move_corn_pot_real_50

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run data/convert_move_corn_pot_data_to_lerobot.py --data_dir /root/openpi/data/move_corn_pot_real_50 --push_to_hub

The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import glob
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "move_corn_pot_real_50"  
TASK_INSTRUCTION = "Lift the yellow corn, then place it next to the kitchen pot with metallic round lid"


def resize_image(image, size):
    """将图像resize到指定尺寸"""
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def compute_joint_velocity(joint_positions):
    """
    计算joint velocity: 下一帧的joint_position - 当前帧的joint_position
    第一帧的velocity设为0（因为没有前一帧）
    最后一帧的velocity设为0（因为没有下一帧）
    """
    num_frames = len(joint_positions)
    joint_velocities = np.zeros_like(joint_positions)
    
    # 计算中间帧的velocity: next - current
    if num_frames > 1:
        joint_velocities[:-1] = joint_positions[1:] - joint_positions[:-1]
        # 最后一帧的velocity设为0
        joint_velocities[-1] = 0.0
    
    return joint_velocities


def normalize_gripper(gripper_widths):
    """
    将gripper_widths归一化并反转：1表示闭合，0表示张开
    gripper_widths表示张开宽度，所以宽度越大（张开）应该对应0，宽度越小（闭合）应该对应1
    """
    max_width = np.max(gripper_widths)
    min_width = np.min(gripper_widths)
    
    # 如果所有值都相同，返回0.5
    if max_width == min_width:
        return np.ones_like(gripper_widths) * 0.5
    
    # 归一化到0-1范围
    normalized = (gripper_widths - min_width) / (max_width - min_width)
    
    # 反转：1 - normalized，使得宽度越大（张开）对应0，宽度越小（闭合）对应1
    inverted = 1.0 - normalized
    return inverted


def main(data_dir: str, *, push_to_hub: bool = False, fps: float = 30.0):
    """
    转换数据到LeRobot格式
    
    Args:
        data_dir: 包含npz文件的目录路径
        push_to_hub: 是否推送到Hugging Face Hub
        fps: 数据采集的帧率，默认30fps
    """
    # 清理已存在的数据集
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    data_dir = Path(data_dir)
    
    # 创建LeRobot数据集，定义要存储的特征
    # 图像尺寸从240x320 resize到180x320（与DROID数据格式保持一致）
    # action是joint_velocity (7D) + gripper_position (1D) = 8D
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=fps,
        features={
            # 使用DROID命名约定以兼容pi05_droid模型
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # joint_velocity (7D) + gripper_position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # 获取所有npz文件
    npz_files = sorted(glob.glob(str(data_dir / "*.npz")))
    print(f"找到 {len(npz_files)} 个episode文件")
    
    # 遍历每个episode文件
    for npz_file in tqdm(npz_files, desc="转换episodes"):
        # 加载数据
        data = np.load(npz_file)
        
        # 提取数据
        right_camera = data['right_camera']  # (N, 240, 320, 3)
        left_camera = data['left_camera']    # (N, 240, 320, 3)
        wrist_camera = data['wrist_camera']  # (N, 240, 320, 3)
        joint_positions = data['joint_positions']  # (N, 7)
        gripper_widths = data['gripper_widths']    # (N,)
        
        num_frames = right_camera.shape[0]
        
        # 计算joint velocity
        joint_velocities = compute_joint_velocity(joint_positions)
        
        # 归一化gripper到0-1范围
        gripper_positions = normalize_gripper(gripper_widths)
        
        # 处理每一帧
        for i in range(num_frames):
            # Resize图像从240x320到180x320（与DROID数据格式保持一致）
            # left_camera作为exterior_image_1_left，right_camera作为exterior_image_2_left
            # 注意：图像可能是RGB格式，需要确保格式正确
            left_img = left_camera[i]
            right_img = right_camera[i]
            wrist_img = wrist_camera[i]
            
            # 确保图像是uint8格式
            def ensure_uint8(img):
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        return (img * 255).astype(np.uint8)
                    else:
                        return img.astype(np.uint8)
                return img
            
            left_img = ensure_uint8(left_img)
            right_img = ensure_uint8(right_img)
            wrist_img = ensure_uint8(wrist_img)
            
            # Resize图像到180x320（保持宽度320，只改变高度从240到180）
            left_img_resized = resize_image(left_img, (320, 180))
            right_img_resized = resize_image(right_img, (320, 180))
            wrist_img_resized = resize_image(wrist_img, (320, 180))
            
            # 构建action: joint_velocity (7D) + gripper_position (1D)
            # 注意：gripper在这里保持连续值（0-1范围），不进行二值化
            # 二值化会在执行action时进行（如DROID main.py中的实现：>0.5 -> 1.0, <=0.5 -> 0.0）
            action = np.concatenate([
                joint_velocities[i].astype(np.float32),
                np.array([gripper_positions[i]], dtype=np.float32)
            ])
            
            # 添加到数据集（使用DROID命名约定）
            dataset.add_frame(
                {
                    "exterior_image_1_left": left_img_resized,  # left_camera
                    "exterior_image_2_left": right_img_resized,  # right_camera
                    "wrist_image_left": wrist_img_resized,
                    "joint_position": joint_positions[i].astype(np.float32),
                    "gripper_position": np.array([gripper_positions[i]], dtype=np.float32),
                    "actions": action,
                    "task": TASK_INSTRUCTION,
                }
            )
        
        # 保存episode
        dataset.save_episode()
    
    print(f"转换完成！数据集保存在: {output_path}")
    
    # 可选：推送到Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["move_corn_pot", "panda", "real_robot"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

