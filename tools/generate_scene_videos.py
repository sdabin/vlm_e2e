"""
Generate videos for all nuScenes validation scenes.
Each video shows 6 camera images arranged in a grid over time.
Video filename is the scene_token.
"""

import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nuscenes import NuScenes
from nuscenes.utils import splits


# Camera names in display order (top row: front-left, front, front-right; bottom row: back-left, back, back-right)
CAMERA_NAMES = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
]


def create_camera_grid(nusc, sample_token, target_size=(1920, 1080)):
    """
    Create a 2x3 grid of camera images for a given sample.

    Args:
        nusc: NuScenes instance
        sample_token: Sample token
        target_size: Target output size (width, height)

    Returns:
        Combined image as numpy array
    """
    sample = nusc.get('sample', sample_token)

    images = []
    for cam_name in CAMERA_NAMES:
        cam_data = nusc.get('sample_data', sample['data'][cam_name])
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            img = np.zeros((900, 1600, 3), dtype=np.uint8)
        images.append(img)

    # Calculate cell size for 2x3 grid
    grid_cols, grid_rows = 3, 2
    cell_width = target_size[0] // grid_cols
    cell_height = target_size[1] // grid_rows

    # Resize all images to cell size
    resized_images = []
    for img in images:
        resized = cv2.resize(img, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        resized_images.append(resized)

    # Create grid (2 rows x 3 cols)
    top_row = np.hstack(resized_images[0:3])
    bottom_row = np.hstack(resized_images[3:6])
    grid = np.vstack([top_row, bottom_row])

    return grid


def get_scene_samples(nusc, scene_token):
    """
    Get all sample tokens for a scene in temporal order.

    Args:
        nusc: NuScenes instance
        scene_token: Scene token

    Returns:
        List of sample tokens in temporal order
    """
    scene = nusc.get('scene', scene_token)
    sample_tokens = []

    sample_token = scene['first_sample_token']
    while sample_token:
        sample_tokens.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next'] if sample['next'] else None

    return sample_tokens


def create_scene_video(nusc, scene_token, output_dir, fps=2, target_size=(1920, 1080)):
    """
    Create a video for a single scene.

    Args:
        nusc: NuScenes instance
        scene_token: Scene token
        output_dir: Output directory
        fps: Frames per second
        target_size: Target video size (width, height)

    Returns:
        Path to created video
    """
    sample_tokens = get_scene_samples(nusc, scene_token)

    if not sample_tokens:
        print(f"No samples found for scene {scene_token}")
        return None

    # Create video writer
    output_path = os.path.join(output_dir, f"{scene_token}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    # Process each sample
    for sample_token in sample_tokens:
        frame = create_camera_grid(nusc, sample_token, target_size)
        writer.write(frame)

    writer.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate videos for nuScenes scenes')
    parser.add_argument('--dataroot', type=str, default='/home/user/UniAD/data/nuscenes',
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--output_dir', type=str, default='/home/user/UniAD/scene_videos',
                        help='Output directory for videos')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'all'],
                        help='Dataset split: train, val, or all')
    parser.add_argument('--fps', type=int, default=2,
                        help='Frames per second for output video')
    parser.add_argument('--width', type=int, default=1920,
                        help='Output video width')
    parser.add_argument('--height', type=int, default=1080,
                        help='Output video height')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize nuScenes
    print("Loading nuScenes dataset...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Get scene names based on split
    if args.split == 'train':
        target_scene_names = set(splits.train)
    elif args.split == 'val':
        target_scene_names = set(splits.val)
    else:  # all
        target_scene_names = set(splits.train) | set(splits.val)

    # Get target scene tokens
    target_scenes = []
    for scene in nusc.scene:
        if scene['name'] in target_scene_names:
            target_scenes.append(scene)

    print(f"Found {len(target_scenes)} {args.split} scenes")

    # Create videos for each scene
    target_size = (args.width, args.height)

    for scene in tqdm(target_scenes, desc="Creating videos"):
        scene_token = scene['token']
        scene_name = scene['name']

        print(f"\nProcessing scene: {scene_name} (token: {scene_token})")

        output_path = create_scene_video(
            nusc, scene_token, args.output_dir,
            fps=args.fps, target_size=target_size
        )

        if output_path:
            print(f"  Saved: {output_path}")

    print(f"\nDone! Videos saved to {args.output_dir}")
    print(f"Total videos created: {len(target_scenes)}")


if __name__ == '__main__':
    main()
