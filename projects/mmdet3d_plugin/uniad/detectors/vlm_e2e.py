#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


class VLMInferenceManager:
    """VLM 추론을 별도 GPU에서 관리하는 클래스.

    UniAD가 GPU 메모리를 대부분 사용하므로 VLM은 다른 GPU에서 실행해야 함.
    이 클래스는 VLM 모델을 지정된 GPU에 한 번만 로드하고 유지함.
    """

    def __init__(self, vlm_device=None, model_id="llava-hf/llava-1.5-7b-hf"):
        """
        Args:
            vlm_device: VLM을 실행할 GPU 디바이스.
                        None이면 자동으로 마지막 GPU 선택.
                        'cpu'면 CPU에서 실행.
            model_id: HuggingFace 모델 ID
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.vlm_device = vlm_device
        self._initialized = False

    def _get_vlm_device(self, uniad_device):
        """UniAD 디바이스를 기반으로 VLM 디바이스 결정.

        Args:
            uniad_device: UniAD가 실행 중인 디바이스

        Returns:
            VLM을 실행할 torch.device
        """
        if self.vlm_device is not None:
            if self.vlm_device == 'cpu':
                return torch.device('cpu')
            return torch.device(f'cuda:{self.vlm_device}')

        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            print(f"[VLM Warning] GPU가 {num_gpus}개뿐입니다. VLM을 CPU에서 실행합니다.")
            return torch.device('cpu')

        # UniAD가 사용하지 않는 마지막 GPU 선택
        uniad_gpu_idx = uniad_device.index if uniad_device.index is not None else 0
        vlm_gpu_idx = num_gpus - 1
        if vlm_gpu_idx == uniad_gpu_idx:
            vlm_gpu_idx = num_gpus - 2 if num_gpus > 2 else 0
            if vlm_gpu_idx == uniad_gpu_idx:
                print("[VLM Warning] 사용 가능한 별도 GPU가 없습니다. CPU에서 실행합니다.")
                return torch.device('cpu')

        return torch.device(f'cuda:{vlm_gpu_idx}')

    def initialize(self, uniad_device):
        """VLM 모델을 지정된 디바이스에 로드.

        Args:
            uniad_device: UniAD가 실행 중인 디바이스 (VLM 디바이스 결정에 사용)
        """
        if self._initialized:
            return

        device = self._get_vlm_device(uniad_device)
        print(f"[VLM] Loading {self.model_id} on {device}...")

        # CPU에서는 float32, GPU에서는 float16
        dtype = torch.float32 if device.type == 'cpu' else torch.float16

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(device)
        self.model.eval()  # 추론 모드

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(self.model_id, tokenizer=tokenizer)

        self._device = device
        self._dtype = dtype
        self._initialized = True
        print(f"[VLM] Model loaded successfully on {device}")

    def _preprocess_tensor_image(self, img_tensor, img_norm_cfg):
        """정규화된 이미지 텐서를 VLM 입력용 PIL 이미지로 변환.

        Args:
            img_tensor: 정규화된 이미지 텐서 (C, H, W), UniAD GPU에 있음
            img_norm_cfg: 정규화 설정 (mean, std)

        Returns:
            PIL.Image: VLM processor용 PIL 이미지
        """
        # CPU로 이동 후 역정규화 (UniAD GPU 메모리 해제)
        img = img_tensor.detach().cpu().float()

        # 역정규화: img = img * std + mean
        mean = torch.tensor(img_norm_cfg['mean']).view(3, 1, 1)
        std = torch.tensor(img_norm_cfg['std']).view(3, 1, 1)
        img = img * std + mean

        # (C, H, W) -> (H, W, C), [0, 255] uint8로 변환
        img = img.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).numpy()

        return Image.fromarray(img)

    def _prepare_inputs(self, img_tensor, text, img_norm_cfg=None):
        """이미지 텐서와 텍스트로부터 VLM 입력 준비.

        Args:
            img_tensor: 이미지 텐서 (C, H, W)
            text: 프롬프트 텍스트
            img_norm_cfg: 이미지 정규화 설정

        Returns:
            processor 출력 (inputs dict)
        """
        # 기본 ImageNet 정규화 설정
        if img_norm_cfg is None:
            img_norm_cfg = dict(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375]
            )

        # 텐서를 PIL 이미지로 변환 (CPU에서 처리)
        pil_image = self._preprocess_tensor_image(img_tensor, img_norm_cfg)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # processor가 PIL 이미지를 받아서 VLM 디바이스로 전송
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors='pt'
        ).to(self._device, self._dtype)

        return inputs

    @torch.no_grad()
    def infer(self, img_tensor, text, max_new_tokens=30, img_norm_cfg=None):
        """VLM 추론 수행.

        Args:
            img_tensor: 이미지 텐서 (C, H, W) - UniAD에서 사용 중인 정규화된 텐서
            text: 프롬프트 텍스트
            max_new_tokens: 최대 생성 토큰 수
            img_norm_cfg: 이미지 정규화 설정 (mean, std). None이면 ImageNet 기본값 사용.

        Returns:
            생성된 토큰 텐서
        """
        if not self._initialized:
            raise RuntimeError("VLMInferenceManager가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        inputs = self._prepare_inputs(img_tensor, text, img_norm_cfg)

        input_ids = inputs["input_ids"]
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        generated_tokens = output[:, input_ids.shape[1]:]

        return generated_tokens

    @torch.no_grad()
    def get_embedding(self, img_tensor, text, img_norm_cfg=None):
        """VLM의 마지막 hidden state를 임베딩으로 추출.

        Args:
            img_tensor: 이미지 텐서 (C, H, W)
            text: 프롬프트 텍스트
            img_norm_cfg: 이미지 정규화 설정

        Returns:
            torch.Tensor: VLM 임베딩 (1, hidden_dim) - 마지막 토큰의 hidden state
        """
        if not self._initialized:
            raise RuntimeError("VLMInferenceManager가 초기화되지 않았습니다. initialize()를 먼저 호출하세요.")

        inputs = self._prepare_inputs(img_tensor, text, img_norm_cfg)

        # Forward pass하여 hidden states 추출
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )

        # 마지막 레이어의 마지막 토큰 hidden state 사용
        # hidden_states: tuple of (layer_output,), 각각 (batch, seq_len, hidden_dim)
        last_hidden_state = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
        # 마지막 토큰의 임베딩 사용 (가장 많은 컨텍스트 정보 포함)
        embedding = last_hidden_state[:, -1, :]  # (1, hidden_dim)

        return embedding

    @property
    def hidden_dim(self):
        """VLM의 hidden dimension 반환."""
        if not self._initialized:
            return 4096  # LLaVA 7B 기본값
        return self.model.config.text_config.hidden_size

    def decode(self, tokens):
        """토큰을 텍스트로 디코딩.

        Args:
            tokens: 생성된 토큰 텐서

        Returns:
            디코딩된 텍스트 문자열
        """
        if not self._initialized:
            raise RuntimeError("VLMInferenceManager가 초기화되지 않았습니다.")
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0]


# 전역 VLM 매니저 (싱글톤 패턴으로 모델 중복 로드 방지)
_vlm_manager = None

def get_vlm_manager(vlm_device=None):
    """전역 VLM 매니저 인스턴스 반환."""
    global _vlm_manager
    if _vlm_manager is None:
        _vlm_manager = VLMInferenceManager(vlm_device=vlm_device)
    return _vlm_manager

@DETECTORS.register_module()
class VlmE2E(UniADTrack):
    """
    UniAD with VLM: Unifying Detection, Tracking, Segmentation, Motion Forecasting,
    Occupancy Prediction and Planning for Autonomous Driving with Vision-Language Model.

    VLM은 UniAD와 별도의 GPU에서 실행되어 메모리 충돌을 방지합니다.
    """
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        vlm_device=None,  # VLM GPU 지정 (None=자동, 'cpu'=CPU, 숫자=특정 GPU)
        vlm_prompt="Explain which element visible in the image is the most important for an autonomous vehicle to perform path planning.",
        vlm_max_tokens=30,
        **kwargs,
    ):
        super(VlmE2E, self).__init__(**kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}

        # VLM 설정 저장 (lazy initialization)
        self._vlm_device = vlm_device
        self._vlm_prompt = vlm_prompt
        self._vlm_max_tokens = vlm_max_tokens
        self._vlm_initialized = False

    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def _ensure_vlm_initialized(self, uniad_device):
        """VLM이 초기화되지 않았으면 초기화 (lazy initialization).

        DDP 환경에서 각 프로세스가 처음 forward할 때 VLM을 로드합니다.
        이렇게 하면 모델 생성 시점이 아닌 실제 사용 시점에 GPU를 할당할 수 있습니다.
        """
        if not self._vlm_initialized:
            vlm_manager = get_vlm_manager(vlm_device=self._vlm_device)
            vlm_manager.initialize(uniad_device)
            self._vlm_initialized = True

    def _run_vlm_inference(self, front_img_tensor, img_metas, target_device):
        """VLM 추론을 별도 GPU에서 실행하고 임베딩 반환.

        이미지 텐서를 직접 사용하여 디스크 I/O를 제거합니다.
        텐서는 CPU로 복사 후 역정규화되어 VLM GPU로 전송됩니다.

        Args:
            front_img_tensor: 전방 카메라 이미지 텐서 (C, H, W), UniAD GPU에 있음
            img_metas: 이미지 메타데이터 (img_norm_cfg 포함)
            target_device: 임베딩을 반환할 디바이스 (UniAD GPU)

        Returns:
            dict: VLM 추론 결과 (embedding, tokens, text)
        """
        self._ensure_vlm_initialized(front_img_tensor.device)

        vlm_manager = get_vlm_manager()

        # img_metas에서 정규화 설정 가져오기
        img_norm_cfg = img_metas[0].get('img_norm_cfg', None)

        # 임베딩 추출 (VLM GPU에서)
        vlm_embed = vlm_manager.get_embedding(
            img_tensor=front_img_tensor,
            text=self._vlm_prompt,
            img_norm_cfg=img_norm_cfg
        )

        # 임베딩을 UniAD GPU로 이동 (planning_head에서 사용)
        vlm_embed = vlm_embed.to(target_device).float()

        # 텍스트 생성 (선택적 - 디버깅/로깅용)
        vlm_tokens = vlm_manager.infer(
            img_tensor=front_img_tensor,
            text=self._vlm_prompt,
            max_new_tokens=self._vlm_max_tokens,
            img_norm_cfg=img_norm_cfg
        )
        vlm_text = vlm_manager.decode(vlm_tokens)

        return {
            'embedding': vlm_embed,  # (1, hidden_dim) on UniAD GPU
            'tokens': vlm_tokens,
            'text': vlm_text,
        }

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
            gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
            gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
            gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
            gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
            gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
            gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
            sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
            sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
            command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
            gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        len_queue = img.size(1)
        

        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)

        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue-1] for each in img_metas]

        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(bev_embed,
                                                        gt_bboxes_3d, gt_labels_3d, 
                                                        gt_fut_traj, gt_fut_traj_mask, 
                                                        gt_sdc_fut_traj, gt_sdc_fut_traj_mask, 
                                                        outs_track=outs_track, outs_seg=outs_seg
                                                    )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # Forward Occ Head
        if self.with_occ_head:
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                            bev_embed,
                            outs_motion,
                            gt_inds_list=gt_inds,
                            gt_segmentation=gt_segmentation,
                            gt_instance=gt_instance,
                            gt_img_is_valid=gt_occ_img_is_valid,
                        )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)

        # Forward VLM (별도 GPU에서 실행)
        # img shape: (B, queue_len, num_cams, C, H, W)
        # img[:, -1, 0]: 마지막 타임스텝의 전방 카메라 (batch, C, H, W)
        # img[0, -1, 0]: 첫 번째 배치의 전방 카메라 이미지 (C, H, W)
        front_img_tensor = img[0, -1, 0]  # (C, H, W)
        vlm_output = self._run_vlm_inference(front_img_tensor, img_metas, target_device=bev_embed.device)
        vlm_embed = vlm_output['embedding']  # (1, vlm_hidden_dim) on UniAD GPU

        # Forward Plan Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(
                bev_embed, outs_motion, sdc_planning, sdc_planning_mask,
                command, gt_future_boxes, vlm_embed=vlm_embed
            )
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)

        return losses
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """Test function
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        # Forward VLM (별도 GPU에서 실행)
        # img shape: (num_cams, C, H, W) for test
        front_img_tensor = img[0]  # 전방 카메라 (C, H, W)
        vlm_output = self._run_vlm_inference(front_img_tensor, img_metas, target_device=bev_embed.device)
        vlm_embed = vlm_output['embedding']

        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(
                bev_embed, outs_motion, outs_occ, command, vlm_embed=vlm_embed
            )
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=['pts_bbox', 'args_tuple'])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]['occ'] = pop_elem_in_result(result[0]['occ'],  \
                pop_list=['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid'])
        
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result


def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result