import argparse
import numpy as np
import matplotlib.pyplot as plt
import mmcv
from nuscenes import NuScenes
from pyquaternion import Quaternion


def get_gt_trajectory(nusc, sample_token, num_future_steps=6):
    """Get GT ego trajectory from future samples."""
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    current_ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    current_translation = np.array(current_ego_pose['translation'])
    current_rotation = Quaternion(current_ego_pose['rotation'])

    gt_traj = [[0.0, 0.0]]  # Start at origin

    next_sample_token = sample['next']
    for _ in range(num_future_steps):
        if next_sample_token == '':
            break
        next_sample = nusc.get('sample', next_sample_token)
        next_lidar_data = nusc.get('sample_data', next_sample['data']['LIDAR_TOP'])
        next_ego_pose = nusc.get('ego_pose', next_lidar_data['ego_pose_token'])

        future_translation = np.array(next_ego_pose['translation'])
        relative_pos = future_translation - current_translation
        relative_pos_local = current_rotation.inverse.rotate(relative_pos)

        gt_traj.append([relative_pos_local[0], relative_pos_local[1]])
        next_sample_token = next_sample['next']

    return np.array(gt_traj)


def main(args):
    # Load predictions
    outputs = mmcv.load(args.predroot)['bbox_results']

    # Load nuScenes
    nusc = NuScenes(version='v1.0-trainval', dataroot=args.dataroot, verbose=True)

    # Get sample token
    if args.sample_idx is not None:
        token = outputs[args.sample_idx]['token']
    else:
        token = outputs[0]['token']

    # Get planner trajectory
    planning_traj = outputs[args.sample_idx]['planning_traj'][0].cpu().numpy()

    # Get GT trajectory
    gt_traj = get_gt_trajectory(nusc, token, num_future_steps=len(planning_traj))

    # Time steps
    t_plan = np.arange(len(planning_traj))
    t_gt = np.arange(len(gt_traj))

    # Add origin to planning traj for comparison
    planning_traj_full = np.vstack([[0, 0], planning_traj])
    t_plan_full = np.arange(len(planning_traj_full))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # X coordinate
    axes[0].plot(t_gt, gt_traj[:, 0], 'g-o', label='GT', linewidth=2, markersize=8)
    axes[0].plot(t_plan_full, planning_traj_full[:, 0], 'r-o', label='Planner', linewidth=2, markersize=8)
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('X (forward) [m]', fontsize=12)
    axes[0].set_title('X Coordinate over Time', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True)

    # Y coordinate
    axes[1].plot(t_gt, gt_traj[:, 1], 'g-o', label='GT', linewidth=2, markersize=8)
    axes[1].plot(t_plan_full, planning_traj_full[:, 1], 'r-o', label='Planner', linewidth=2, markersize=8)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Y (left) [m]', fontsize=12)
    axes[1].set_title('Y Coordinate over Time', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True)

    plt.suptitle(f'Trajectory Comparison (Sample {args.sample_idx})', fontsize=16)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f'Saved to {args.output}')

    # Print values
    print(f'\nGT trajectory (x, y):')
    for i, (x, y) in enumerate(gt_traj):
        print(f'  t={i}: ({x:.3f}, {y:.3f})')

    print(f'\nPlanner trajectory (x, y):')
    for i, (x, y) in enumerate(planning_traj_full):
        print(f'  t={i}: ({x:.3f}, {y:.3f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', required=True, help='Path to results.pkl')
    parser.add_argument('--dataroot', default='/home/user/UniAD/data/nuscenes', help='Path to nuScenes data')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    parser.add_argument('--output', default='traj_comparison.png', help='Output image path')
    args = parser.parse_args()

    main(args)
