import cv2
import torch
import argparse
import os
import sys
import glob
import numpy as np
import mmcv
import matplotlib
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nuscenes import NuScenes
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils import splits
from pyquaternion import Quaternion
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import obtain_map_info
from projects.mmdet3d_plugin.datasets.eval_utils.map_api import NuScenesMap
from PIL import Image
from tools.analysis_tools.visualize.utils import color_mapping, AgentPredictionData
from tools.analysis_tools.visualize.render.bev_render import BEVRender
from tools.analysis_tools.visualize.render.cam_render import CameraRender


class Visualizer:
    """
    BaseRender class
    """

    def __init__(
            self,
            dataroot='/home/user/UniAD/data/nuscenes',
            version='v1.0-trainval',
            predroot=None,
            with_occ_map=False,
            with_map=False,
            with_planning=False,
            with_pred_box=True,
            with_pred_traj=False,
            show_gt_boxes=False,
            show_lidar=False,
            show_command=False,
            show_hd_map=False,
            show_sdc_car=False,
            show_sdc_traj=False,
            show_legend=False,
            show_vlm_text=False):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.predict_helper = PredictHelper(self.nusc)
        self.with_occ_map = with_occ_map
        self.with_map = with_map
        self.with_planning = with_planning
        self.show_lidar = show_lidar
        self.show_command = show_command
        self.show_hd_map = show_hd_map
        self.show_sdc_car = show_sdc_car
        self.show_sdc_traj = show_sdc_traj
        self.show_legend = show_legend
        self.show_vlm_text = show_vlm_text
        self.with_pred_traj = with_pred_traj
        self.with_pred_box = with_pred_box
        self.veh_id_list = [0, 1, 2, 3, 4, 6, 7]
        self.use_json = '.json' in predroot
        self.token_set = set()
        self.is_curated_format = False  # Flag for curated pkl format

        # Build scene mapping for curated format support
        self._build_scene_mapping()

        self.predictions = self._parse_predictions_multitask_pkl(predroot)
        self.bev_render = BEVRender(show_gt_boxes=show_gt_boxes)
        self.cam_render = CameraRender(show_gt_boxes=show_gt_boxes)

        if self.show_hd_map:
            self.nusc_maps = {
                'boston-seaport': NuScenesMap(dataroot=dataroot, map_name='boston-seaport'),
                'singapore-hollandvillage': NuScenesMap(dataroot=dataroot, map_name='singapore-hollandvillage'),
                'singapore-onenorth': NuScenesMap(dataroot=dataroot, map_name='singapore-onenorth'),
                'singapore-queenstown': NuScenesMap(dataroot=dataroot, map_name='singapore-queenstown'),
            }

    def _build_scene_mapping(self):
        """Build mapping from scene_id (e.g., '089') to nuScenes scene and its samples."""
        # Map scene_id to scene_token
        self.scene_id_to_token = {}
        for scene in self.nusc.scene:
            # Extract number from scene name like "scene-0089"
            scene_num = scene['name'].split('-')[1]  # "0089"
            scene_id = scene_num.lstrip('0') or '0'  # "89" or "0"
            # Also store with leading zeros for flexibility
            self.scene_id_to_token[scene_id] = scene['token']
            self.scene_id_to_token[scene_num] = scene['token']

        # Map scene_token to ordered list of sample_tokens
        self.scene_to_samples = {}
        for scene in self.nusc.scene:
            samples = []
            sample_token = scene['first_sample_token']
            while sample_token:
                samples.append(sample_token)
                sample = self.nusc.get('sample', sample_token)
                sample_token = sample['next']
            self.scene_to_samples[scene['token']] = samples

    def _parse_curated_token(self, token):
        """Parse curated token format: {hash}-{scene_id}_{timestamp}_{frame_idx}
        Returns: (nusc_sample_token, scene_id, frame_idx) or (None, None, None) if invalid
        """
        if '-' not in token:
            return None, None, None

        parts = token.split('-')
        if len(parts) != 2:
            return None, None, None

        suffix_parts = parts[1].split('_')
        if len(suffix_parts) < 3:
            return None, None, None

        scene_id = suffix_parts[0]  # e.g., "089"
        frame_idx = int(suffix_parts[2])  # e.g., 0, 1, 2, ...

        # Map scene_id to nuScenes scene_token
        scene_token = self.scene_id_to_token.get(scene_id)
        if scene_token is None:
            return None, None, None

        # Get the sample_token for this frame in the scene
        samples = self.scene_to_samples.get(scene_token, [])
        if frame_idx >= len(samples):
            return None, None, None

        nusc_sample_token = samples[frame_idx]
        return nusc_sample_token, scene_id, frame_idx

    def _parse_predictions_multitask_pkl(self, predroot):

        outputs = mmcv.load(predroot)
        outputs = outputs['bbox_results']
        prediction_dict = dict()

        # Check if this is curated format by examining first token
        if len(outputs) > 0:
            first_token = outputs[0]['token']
            if '-' in first_token and '_' in first_token.split('-')[1]:
                self.is_curated_format = True
                print(f"Detected curated pkl format")

        for k in range(len(outputs)):
            token = outputs[k]['token']

            # Handle curated token format: {hash}-{scene_id}_{timestamp}_{frame_idx}
            if self.is_curated_format:
                sample_token, scene_id, frame_idx = self._parse_curated_token(token)
                if sample_token is None:
                    continue  # Skip invalid tokens
            else:
                sample_token = token

            self.token_set.add(sample_token)
            if self.show_sdc_traj:
                outputs[k]['boxes_3d'].tensor = torch.cat(
                    [outputs[k]['boxes_3d'].tensor, outputs[k]['sdc_boxes_3d'].tensor], dim=0)
                outputs[k]['scores_3d'] = torch.cat(
                    [outputs[k]['scores_3d'], outputs[k]['sdc_scores_3d']], dim=0)
                outputs[k]['labels_3d'] = torch.cat([outputs[k]['labels_3d'], torch.zeros(
                    (1,), device=outputs[k]['labels_3d'].device)], dim=0)
            # detection
            bboxes = outputs[k]['boxes_3d']
            scores = outputs[k]['scores_3d']
            labels = outputs[k]['labels_3d']

            track_scores = scores.cpu().detach().numpy()
            track_labels = labels.cpu().detach().numpy()
            track_boxes = bboxes.tensor.cpu().detach().numpy()

            track_centers = bboxes.gravity_center.cpu().detach().numpy()
            track_dims = bboxes.dims.cpu().detach().numpy()
            track_yaw = bboxes.yaw.cpu().detach().numpy()

            if 'track_ids' in outputs[k]:
                track_ids = outputs[k]['track_ids'].cpu().detach().numpy()
            else:
                track_ids = None

            # speed
            track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

            # trajectories
            trajs = outputs[k][f'traj'].numpy()
            traj_scores = outputs[k][f'traj_scores'].numpy()

            predicted_agent_list = []

            # occflow
            if self.with_occ_map:
                if 'topk_query_ins_segs' in outputs[k]['occ']:
                    occ_map = outputs[k]['occ']['topk_query_ins_segs'][0].cpu(
                    ).numpy()
                else:
                    occ_map = np.zeros((1, 5, 200, 200))
            else:
                occ_map = None

            occ_idx = 0
            for i in range(track_scores.shape[0]):
                if track_scores[i] < 0.25:
                    continue
                if occ_map is not None and track_labels[i] in self.veh_id_list:
                    occ_map_cur = occ_map[occ_idx, :, ::-1]
                    occ_idx += 1
                else:
                    occ_map_cur = None
                if track_ids is not None:
                    if i < len(track_ids):
                        track_id = track_ids[i]
                    else:
                        track_id = 0
                else:
                    track_id = None
                # if track_labels[i] not in [0, 1, 2, 3, 4, 6, 7]:
                #     continue
                predicted_agent_list.append(
                    AgentPredictionData(
                        track_scores[i],
                        track_labels[i],
                        track_centers[i],
                        track_dims[i],
                        track_yaw[i],
                        track_velocity[i],
                        trajs[i],
                        traj_scores[i],
                        pred_track_id=track_id,
                        pred_occ_map=occ_map_cur,
                        past_pred_traj=None
                    )
                )

            if self.with_map:
                map_thres = 0.7
                score_list = outputs[k]['pts_bbox']['score_list'].cpu().numpy().transpose([
                    1, 2, 0])
                predicted_map_seg = outputs[k]['pts_bbox']['lane_score'].cpu().numpy().transpose([
                    1, 2, 0])  # H, W, C
                predicted_map_seg[..., -1] = score_list[..., -1]
                predicted_map_seg = (predicted_map_seg > map_thres) * 1.0
                predicted_map_seg = predicted_map_seg[::-1, :, :]
            else:
                predicted_map_seg = None

            if self.with_planning:
                # detection
                bboxes = outputs[k]['sdc_boxes_3d']
                scores = outputs[k]['sdc_scores_3d']
                labels = 0

                track_scores = scores.cpu().detach().numpy()
                track_labels = labels
                track_boxes = bboxes.tensor.cpu().detach().numpy()

                track_centers = bboxes.gravity_center.cpu().detach().numpy()
                track_dims = bboxes.dims.cpu().detach().numpy()
                track_yaw = bboxes.yaw.cpu().detach().numpy()
                track_velocity = bboxes.tensor.cpu().detach().numpy()[:, -2:]

                if self.show_command:
                    command = outputs[k]['command'][0].cpu().detach().numpy()
                else:
                    command = None
                planning_agent = AgentPredictionData(
                    track_scores[0],
                    track_labels,
                    track_centers[0],
                    track_dims[0],
                    track_yaw[0],
                    track_velocity[0],
                    outputs[k]['planning_traj'][0].cpu().detach().numpy(),
                    1,
                    pred_track_id=-1,
                    pred_occ_map=None,
                    past_pred_traj=None,
                    is_sdc=True,
                    command=command,
                )
                predicted_agent_list.append(planning_agent)
            else:
                planning_agent = None

            # VLM 텍스트 파싱
            vlm_info = None
            if self.show_vlm_text:
                if 'vlm' in outputs[k]:
                    vlm_info = outputs[k]['vlm']
                elif 'planning' in outputs[k] and isinstance(outputs[k]['planning'], dict):
                    if 'vlm_text' in outputs[k]['planning']:
                        vlm_info = dict(
                            text=outputs[k]['planning']['vlm_text'],
                            prompt=outputs[k]['planning'].get('vlm_prompt', '')
                        )

            prediction_dict[sample_token] = dict(predicted_agent_list=predicted_agent_list,
                                          predicted_map_seg=predicted_map_seg,
                                          predicted_planning=planning_agent,
                                          vlm_info=vlm_info)
        return prediction_dict

    def visualize_bev(self, sample_token, out_filename, t=None):
        self.bev_render.reset_canvas(dx=1, dy=1)
        self.bev_render.set_plot_cfg()

        if self.show_lidar:
            self.bev_render.show_lidar_data(sample_token, self.nusc)
        if self.bev_render.show_gt_boxes:
            self.bev_render.render_anno_data(
                sample_token, self.nusc, self.predict_helper)
        if self.with_pred_box:
            self.bev_render.render_pred_box_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_pred_traj:
            self.bev_render.render_pred_traj(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_map:
            self.bev_render.render_pred_map_data(
                self.predictions[sample_token]['predicted_map_seg'])
        if self.with_occ_map:
            self.bev_render.render_occ_map_data(
                self.predictions[sample_token]['predicted_agent_list'])
        if self.with_planning:
            self.bev_render.render_pred_box_data(
                [self.predictions[sample_token]['predicted_planning']])
            self.bev_render.render_planning_data(
                self.predictions[sample_token]['predicted_planning'], show_command=self.show_command)
        if self.show_hd_map:
            self.bev_render.render_hd_map(
                self.nusc, self.nusc_maps, sample_token)
        if self.show_sdc_car:
            self.bev_render.render_sdc_car()
        if self.show_legend:
            self.bev_render.render_legend()
        self.bev_render.save_fig(out_filename + '.jpg')

    def visualize_cam(self, sample_token, out_filename):
        self.cam_render.reset_canvas(dx=2, dy=3, tight_layout=True)
        self.cam_render.render_image_data(sample_token, self.nusc)
        self.cam_render.render_pred_track_bbox(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc)
        self.cam_render.render_pred_traj(
            self.predictions[sample_token]['predicted_agent_list'], sample_token, self.nusc, render_sdc=self.with_planning)
        self.cam_render.save_fig(out_filename + '_cam.jpg')

    def combine(self, out_filename, sample_token=None):
        # pass
        bev_image = cv2.imread(out_filename + '.jpg')
        cam_image = cv2.imread(out_filename + '_cam.jpg')
        merge_image = cv2.hconcat([cam_image, bev_image])

        # Add trajectory comparison graph
        if self.with_planning and sample_token is not None:
            traj_graph = self._create_traj_comparison_graph(sample_token, merge_image.shape[1])
            if traj_graph is not None:
                merge_image = cv2.vconcat([merge_image, traj_graph])

        # VLM 텍스트가 있으면 이미지 하단에 추가
        if self.show_vlm_text and sample_token is not None:
            vlm_info = self.predictions[sample_token].get('vlm_info', None)
            if vlm_info is not None:
                merge_image = self._add_vlm_text_to_image(merge_image, vlm_info)

        cv2.imwrite(out_filename + '.jpg', merge_image)
        os.remove(out_filename + '_cam.jpg')

    def _get_l2g_transform(self, sample_record):
        """Get lidar to global transform (same as trajectory_api.py)."""
        lidar_data = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        l2e_r = cs_record['rotation']
        l2e_t = np.array(cs_record['translation'])
        e2g_r = pose_record['rotation']
        e2g_t = np.array(pose_record['translation'])

        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t

    def _get_gt_trajectory(self, sample_token, num_future_steps=6):
        """Get GT ego trajectory in lidar frame (same method as UniAD's trajectory_api.py)."""
        sd_rec = self.nusc.get('sample', sample_token)
        l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self._get_l2g_transform(sd_rec)

        gt_traj = [[0.0, 0.0]]  # Start at origin

        for _ in range(num_future_steps):
            next_token = sd_rec['next']
            if next_token == '':
                break
            sd_rec = self.nusc.get('sample', next_token)
            l2e_r_mat_curr, l2e_t_curr, e2g_r_mat_curr, e2g_t_curr = self._get_l2g_transform(sd_rec)

            # Future ego position in current lidar frame
            # Step 1: Get future ego position in global frame
            future_pos_global = e2g_t_curr.copy()

            # Step 2: Transform to initial ego frame
            future_pos_ego = np.linalg.inv(e2g_r_mat_init) @ (future_pos_global - e2g_t_init)

            # Step 3: Transform to initial lidar frame
            future_pos_lidar = np.linalg.inv(l2e_r_mat_init) @ (future_pos_ego - l2e_t_init)

            gt_traj.append([future_pos_lidar[0], future_pos_lidar[1]])

        return np.array(gt_traj)

    def _create_traj_comparison_graph(self, sample_token, target_width):
        """Create trajectory comparison graph as image."""
        # Get planner trajectory
        planning_agent = self.predictions[sample_token].get('predicted_planning')
        if planning_agent is None:
            return None

        planning_traj = planning_agent.pred_traj
        planning_traj_full = np.vstack([[0, 0], planning_traj])

        # Get GT trajectory
        gt_traj = self._get_gt_trajectory(sample_token, num_future_steps=len(planning_traj))

        # Create matplotlib figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 2.8))

        t_gt = np.arange(len(gt_traj))
        t_plan = np.arange(len(planning_traj_full))

        # X coordinate (forward, positive = forward)
        axes[0].plot(t_gt, gt_traj[:, 1], 'g-o', label='GT', linewidth=2, markersize=8)
        axes[0].plot(t_plan, planning_traj_full[:, 1], 'r-o', label='Planner', linewidth=2, markersize=8)
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('X (forward) [m]', fontsize=12)
        axes[0].set_title('X Coordinate', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Y coordinate (lateral, positive = left)
        axes[1].plot(t_gt, gt_traj[:, 0], 'g-o', label='GT', linewidth=2, markersize=8)
        axes[1].plot(t_plan, planning_traj_full[:, 0], 'r-o', label='Planner', linewidth=2, markersize=8)
        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('Y (left) [m]', fontsize=12)
        axes[1].set_title('Y Coordinate', fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        graph_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_image = graph_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
        plt.close(fig)

        # Resize to match target width
        h, w = graph_image.shape[:2]
        new_h = int(h * target_width / w)
        graph_image = cv2.resize(graph_image, (target_width, new_h))

        return graph_image

    def _add_vlm_text_to_image(self, image, vlm_info):
        """VLM 텍스트를 이미지 하단에 추가합니다."""
        prompt = vlm_info.get('prompt', '')
        text = vlm_info.get('text', '')

        if not text:
            return image

        h, w = image.shape[:2]

        # 폰트 설정 (가독성 좋은 작은 크기)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.4
        font_thickness = 2
        padding = 30
        bg_color = (40, 40, 40)  # 어두운 회색 배경
        prompt_color = (100, 200, 255)  # 주황색 (BGR)
        text_color = (255, 255, 255)  # 흰색

        # 실제 텍스트 높이 측정
        (_, text_height), baseline = cv2.getTextSize("Ay", font, font_scale, font_thickness)
        line_height = text_height + baseline + 15  # 줄 간격 여유

        # 텍스트 영역은 이미지 너비의 95% 사용
        text_area_width = int(w * 0.95)
        margin_left = (w - text_area_width) // 2

        # 이미지 너비에 맞는 최대 문자 수 계산
        (char_width, _), _ = cv2.getTextSize("a", font, font_scale, font_thickness)
        max_width_chars = int(text_area_width / char_width)

        # 텍스트 줄바꿈 처리
        def wrap_text(text, max_chars):
            words = text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            return lines

        # 프롬프트와 응답 텍스트 준비
        prompt_lines = wrap_text(f"[Prompt] {prompt}", max_width_chars) if prompt else []
        response_lines = wrap_text(f"[VLM Response] {text}", max_width_chars)

        all_lines = prompt_lines + response_lines
        total_lines = len(all_lines)

        # 텍스트 영역 높이 계산 (상하 패딩 포함)
        text_area_height = total_lines * line_height + padding * 2

        # 새 이미지 생성 (원본 + 텍스트 영역)
        new_image = np.zeros((h + text_area_height, w, 3), dtype=np.uint8)
        new_image[:h, :] = image
        new_image[h:, :] = bg_color

        # 텍스트 그리기 (가운데 정렬)
        y_offset = h + padding + text_height  # 첫 줄 시작 위치
        for i, line in enumerate(all_lines):
            if i < len(prompt_lines):
                color = prompt_color
            else:
                color = text_color
            cv2.putText(new_image, line, (margin_left, y_offset),
                        font, font_scale, color, font_thickness, cv2.LINE_AA)
            y_offset += line_height

        return new_image

    def to_video(self, folder_path, out_path, fps=4, downsample=1):
        imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
        imgs_path = sorted(imgs_path)
        if len(imgs_path) == 0:
            print(f"No images found in {folder_path}, skipping video creation")
            return
        img_array = []
        size = None
        for img_path in imgs_path:
            img = cv2.imread(img_path)
            height, width, channel = img.shape
            img = cv2.resize(img, (width//downsample, height //
                             downsample), interpolation=cv2.INTER_AREA)
            height, width, channel = img.shape
            size = (width, height)
            img_array.append(img)
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

def main(args):
    render_cfg = dict(
        with_occ_map=False,
        with_map=False,
        with_planning=True,
        with_pred_box=True,
        with_pred_traj=True,
        show_gt_boxes=False,
        show_lidar=False,
        show_command=True,
        show_hd_map=False,
        show_sdc_car=True,
        show_legend=True,
        show_sdc_traj=False,
        show_vlm_text=args.show_vlm_text,
    )

    viser = Visualizer(version='v1.0-trainval', predroot=args.predroot, dataroot='/home/user/UniAD/data/nuscenes', **render_cfg)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    val_splits = splits.val

    scene_token_to_name = dict()
    for i in range(len(viser.nusc.scene)):
        scene_token_to_name[viser.nusc.scene[i]['token']] = viser.nusc.scene[i]['name']

    # Track frame index per scene
    scene_frame_counter = dict()
    processed_count = 0

    for i in range(len(viser.nusc.sample)):
        sample_token = viser.nusc.sample[i]['token']
        scene_token = viser.nusc.sample[i]['scene_token']

        # For curated format, skip validation split check
        # For standard format, only process validation samples
        if not viser.is_curated_format:
            if scene_token_to_name[scene_token] not in val_splits:
                continue

        if sample_token not in viser.token_set:
            if not viser.is_curated_format:
                print(i, sample_token, 'not in prediction pkl!')
            continue

        # Get frame index for this scene
        if scene_token not in scene_frame_counter:
            scene_frame_counter[scene_token] = 0
        frame_idx = scene_frame_counter[scene_token]
        scene_frame_counter[scene_token] += 1

        # Output filename: {scene_token}_{frame_idx}
        out_filename = os.path.join(args.out_folder, f"{scene_token}_{frame_idx}")

        viser.visualize_bev(sample_token, out_filename)

        if args.project_to_cam:
            viser.visualize_cam(sample_token, out_filename)
            viser.combine(out_filename, sample_token=sample_token)

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} samples...")

    print(f"Total processed: {processed_count} samples")
    viser.to_video(args.out_folder, args.demo_video, fps=4, downsample=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', default='/mnt/nas20/yihan01.hu/tmp/results.pkl', help='Path to results.pkl')
    parser.add_argument('--out_folder', default=None, help='Output folder path (default: {predroot_dir}/visual_result)')
    parser.add_argument('--demo_video', default=None, help='Demo video name (default: {predroot_name}.avi)')
    parser.add_argument('--project_to_cam', default=True, help='Project to cam (default: True)')
    parser.add_argument('--show_vlm_text', action='store_true', default=True, help='Show VLM prompt and response text at the bottom of the image (default: True)')
    parser.add_argument('--no_vlm_text', action='store_true', help='Hide VLM prompt and response text')
    args = parser.parse_args()

    # out_folder 기본값: predroot와 같은 폴더의 visual_result
    if args.out_folder is None:
        predroot_dir = os.path.dirname(os.path.abspath(args.predroot))
        args.out_folder = os.path.join(predroot_dir, 'visual_result')

    # demo_video 기본값: predroot 파일명 기반
    if args.demo_video is None:
        predroot_name = os.path.splitext(os.path.basename(args.predroot))[0]
        args.demo_video = f'{predroot_name}.avi'

    # --no_vlm_text가 지정되면 show_vlm_text를 False로
    if args.no_vlm_text:
        args.show_vlm_text = False

    main(args)
