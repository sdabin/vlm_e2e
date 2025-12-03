#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import os
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head

from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration


def get_gpu_memory_info():
    """nvidia-smi로 모든 GPU의 메모리 정보 조회.

    Returns:
        dict: {gpu_idx: free_memory_mb} 형태의 딕셔너리
    """
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        gpu_info = {}
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_idx = int(parts[0].strip())
                free_mem_mb = int(parts[1].strip())
                gpu_info[gpu_idx] = free_mem_mb
        return gpu_info
    except Exception as e:
        print(f"[VLM Warning] nvidia-smi 실행 실패: {e}")
        return {}


def find_best_vlm_gpu(min_memory_gb=14):
    """학습 GPU를 제외하고 메모리가 충분한 GPU를 찾음.

    CUDA_VISIBLE_DEVICES 환경변수를 확인하여 학습에 사용 중인 GPU를 파악하고,
    그 외의 GPU 중 메모리가 가장 많이 남은 것을 선택합니다.

    Args:
        min_memory_gb: VLM에 필요한 최소 메모리 (GB)

    Returns:
        int or None: 사용 가능한 물리적 GPU 인덱스, 없으면 None
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)

    # 학습에 사용 중인 물리적 GPU 인덱스
    if cuda_visible is None or cuda_visible == '':
        # 환경변수 없음 - 학습이 모든 GPU를 사용한다고 가정
        print("[VLM Warning] CUDA_VISIBLE_DEVICES가 설정되지 않았습니다.")
        return None

    training_physical_gpus = [int(x.strip()) for x in cuda_visible.split(',')]
    print(f"[VLM] 학습에 사용 중인 물리적 GPU: {training_physical_gpus}")

    # 모든 GPU의 메모리 정보 조회
    gpu_memory = get_gpu_memory_info()
    if not gpu_memory:
        return None

    # 학습 GPU를 제외하고 메모리가 충분한 GPU 찾기
    available_gpus = []
    for gpu_idx, free_mem_mb in gpu_memory.items():
        if gpu_idx not in training_physical_gpus:
            available_gpus.append((gpu_idx, free_mem_mb))

    if not available_gpus:
        print("[VLM Warning] 학습 GPU 외에 사용 가능한 GPU가 없습니다.")
        return None

    # 메모리가 충분한 GPU 우선 선택
    sufficient_gpus = [(idx, mem) for idx, mem in available_gpus if mem >= min_memory_gb * 1024]

    if sufficient_gpus:
        # 메모리가 가장 많은 GPU 선택
        sufficient_gpus.sort(key=lambda x: x[1], reverse=True)
        return sufficient_gpus[0][0]

    # 메모리 조건을 만족하는 GPU가 없으면 가장 여유 있는 것 선택
    available_gpus.sort(key=lambda x: x[1], reverse=True)
    print(f"[VLM Warning] 메모리 {min_memory_gb}GB 이상인 GPU가 없습니다. "
          f"GPU {available_gpus[0][0]} ({available_gpus[0][1]}MB free) 사용.")
    return available_gpus[0][0]


class VLMInferenceManager:
    """VLM 추론을 별도 GPU에서 관리하는 클래스.

    UniAD가 GPU 메모리를 대부분 사용하므로 VLM은 다른 GPU에서 실행해야 함.

    사용법:
    1. 학습 스크립트에서 VLM GPU도 CUDA_VISIBLE_DEVICES에 포함시켜야 함
       예: CUDA_VISIBLE_DEVICES=0,1,2,3 (학습: 0,1 / VLM: 2 또는 3)
    2. vlm_device='auto'면 CUDA_VISIBLE_DEVICES 내에서 DDP가 사용하지 않는 GPU 선택
    """

    def __init__(self, vlm_device='auto', model_id="llava-hf/llava-1.5-7b-hf"):
        """
        Args:
            vlm_device: VLM을 실행할 GPU 디바이스.
                        'auto'면 자동으로 남는 GPU 선택 (기본값).
                        'cpu'면 CPU에서 실행.
                        숫자면 해당 논리적 GPU 인덱스 사용.
            model_id: HuggingFace 모델 ID
        """
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.vlm_device = vlm_device
        self._initialized = False

    def _find_available_vlm_gpu(self, min_memory_gb=14):
        """VLM에 사용할 GPU 찾기.

        CUDA_VISIBLE_DEVICES 내에서 학습에 사용되지 않는 GPU 중
        각 DDP rank에 대응하는 VLM GPU를 선택합니다.

        DDP rank 0 -> VLM GPU (world_size + 0)
        DDP rank 1 -> VLM GPU (world_size + 1)
        ...

        Args:
            min_memory_gb: VLM에 필요한 최소 메모리 (GB)

        Returns:
            int or None: 논리적 GPU 인덱스 (CUDA_VISIBLE_DEVICES 내 인덱스), 없으면 None
        """
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')

        # DDP rank 확인
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))

        print(f"[VLM][Rank {local_rank}] 현재 CUDA_VISIBLE_DEVICES: {cuda_visible}")
        print(f"[VLM][Rank {local_rank}] LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")

        if not cuda_visible:
            print(f"[VLM][Rank {local_rank}] Warning: CUDA_VISIBLE_DEVICES가 설정되지 않았습니다.")
            return None

        # CUDA_VISIBLE_DEVICES의 물리적 GPU 목록
        visible_physical_gpus = [int(x.strip()) for x in cuda_visible.split(',') if x.strip()]
        print(f"[VLM][Rank {local_rank}] CUDA_VISIBLE_DEVICES 내 물리적 GPU: {visible_physical_gpus}")

        # VLM이 사용할 수 있는 논리적 인덱스: world_size ~ (len(visible_physical_gpus) - 1)
        num_visible = len(visible_physical_gpus)
        num_vlm_gpus = num_visible - world_size

        if num_vlm_gpus <= 0:
            print(f"[VLM][Rank {local_rank}] Warning: 모든 visible GPU가 학습에 사용 중입니다. "
                  f"(visible: {num_visible}, world_size: {world_size})")
            print(f"[VLM][Rank {local_rank}] Warning: VLM GPU를 CUDA_VISIBLE_DEVICES에 추가하세요.")
            return None

        # 각 rank에 대응하는 VLM GPU 할당
        # rank 0 -> world_size, rank 1 -> world_size + 1, ...
        if num_vlm_gpus >= world_size:
            # VLM GPU가 충분한 경우: 각 rank에 1:1 매핑
            vlm_logical_idx = world_size + local_rank
        else:
            # VLM GPU가 부족한 경우: 라운드 로빈으로 분배
            vlm_logical_idx = world_size + (local_rank % num_vlm_gpus)

        physical_idx = visible_physical_gpus[vlm_logical_idx]
        print(f"[VLM][Rank {local_rank}] 할당된 VLM GPU: 논리적 {vlm_logical_idx} (물리적 {physical_idx})")

        # 메모리 확인 (정보 출력용)
        gpu_memory = get_gpu_memory_info()
        if gpu_memory:
            free_mem_mb = gpu_memory.get(physical_idx, 0)
            print(f"[VLM][Rank {local_rank}] GPU {physical_idx} 메모리: {free_mem_mb}MB free")
            if free_mem_mb < min_memory_gb * 1024:
                print(f"[VLM][Rank {local_rank}] Warning: 메모리가 {min_memory_gb}GB 미만입니다.")

        return vlm_logical_idx

    def initialize(self):
        """VLM 모델을 지정된 디바이스에 로드."""
        if self._initialized:
            return

        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        # 디바이스 결정
        if self.vlm_device == 'cpu':
            logical_gpu_idx = None
            dtype = torch.float32
        elif isinstance(self.vlm_device, int):
            # 특정 논리적 GPU 인덱스가 지정된 경우
            logical_gpu_idx = self.vlm_device
            dtype = torch.float16
        elif isinstance(self.vlm_device, str) and self.vlm_device.isdigit():
            logical_gpu_idx = int(self.vlm_device)
            dtype = torch.float16
        else:  # 'auto'
            logical_gpu_idx = self._find_available_vlm_gpu()
            dtype = torch.float16 if logical_gpu_idx is not None else torch.float32

        if logical_gpu_idx is None:
            print(f"[VLM][Rank {local_rank}] Warning: 사용 가능한 GPU가 없습니다. CPU에서 실행합니다.")
            device = torch.device('cpu')
            dtype = torch.float32
        else:
            device = torch.device(f'cuda:{logical_gpu_idx}')
            print(f"[VLM][Rank {local_rank}] 논리적 GPU {logical_gpu_idx}에 VLM 로드 중...")

        print(f"[VLM][Rank {local_rank}] Loading {self.model_id} on {device}...")

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
        ).to(device)

        self.model.eval()

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(self.model_id, tokenizer=tokenizer)

        self._device = device
        self._dtype = dtype
        self._initialized = True
        print(f"[VLM][Rank {local_rank}] Model loaded successfully on {device}")

    def _preprocess_tensor_image(self, img_tensor, img_norm_cfg):
        """정규화된 이미지 텐서를 VLM 입력용 PIL 이미지로 변환.

        Args:
            img_tensor: 정규화된 이미지 텐서, 다양한 형태 지원:
                        - (C, H, W): 단일 이미지
                        - (1, C, H, W): 배치 크기 1인 이미지
                        - (N, C, H, W): 첫 번째 이미지만 사용
            img_norm_cfg: 정규화 설정 (mean, std)

        Returns:
            PIL.Image: VLM processor용 PIL 이미지
        """
        # CPU로 이동 후 역정규화 (UniAD GPU 메모리 해제)
        img = img_tensor.detach().cpu().float()

        # 4D 텐서인 경우 첫 번째 이미지만 사용 (B, C, H, W) -> (C, H, W)
        if img.dim() == 4:
            img = img[0]

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


# 전역 VLM 매니저 (각 DDP rank가 자체 VLM 인스턴스 보유)
# 각 rank는 서로 다른 GPU에 VLM을 로드
_vlm_manager = None
_vlm_lock = None

def get_vlm_manager(vlm_device=None):
    """전역 VLM 매니저 인스턴스 반환.

    DDP 환경에서 각 rank가 자체 VLM 매니저를 가집니다.
    각 프로세스는 서로 다른 GPU에 VLM을 로드합니다.
    """
    global _vlm_manager, _vlm_lock

    import threading
    if _vlm_lock is None:
        _vlm_lock = threading.Lock()

    with _vlm_lock:
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
        # Freeze 옵션들
        freeze_seg_head=False,
        freeze_motion_head=False,
        freeze_occ_head=False,
        freeze_planning_head=False,
        freeze_vlm_proj=False,  # planning_head 내의 vlm_proj만 freeze
        unfreeze_vlm_proj=False,  # planning_head freeze 시 vlm_proj만 학습 가능하게
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

        # dict로 명시적 변환 (pickle 호환성)
        self.task_loss_weight = dict(task_loss_weight)
        assert set(self.task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}

        # VLM 설정 저장 (lazy initialization)
        self._vlm_device = vlm_device
        self._vlm_prompt = vlm_prompt
        self._vlm_max_tokens = vlm_max_tokens
        self._vlm_initialized = False

        # Freeze 상태 저장 (forward에서 불필요한 연산 스킵용)
        self._freeze_seg_head = freeze_seg_head
        self._freeze_motion_head = freeze_motion_head
        self._freeze_occ_head = freeze_occ_head
        self._freeze_planning_head = freeze_planning_head

        # Freeze 설정 적용
        self._apply_freeze_settings(
            freeze_seg_head=freeze_seg_head,
            freeze_motion_head=freeze_motion_head,
            freeze_occ_head=freeze_occ_head,
            freeze_planning_head=freeze_planning_head,
            freeze_vlm_proj=freeze_vlm_proj,
            unfreeze_vlm_proj=unfreeze_vlm_proj,
        )

    def _apply_freeze_settings(
        self,
        freeze_seg_head=False,
        freeze_motion_head=False,
        freeze_occ_head=False,
        freeze_planning_head=False,
        freeze_vlm_proj=False,
        unfreeze_vlm_proj=False,
    ):
        """각 head 모듈의 freeze 설정을 적용합니다."""

        if freeze_seg_head and self.with_seg_head:
            self._freeze_module(self.seg_head, 'seg_head')

        if freeze_motion_head and self.with_motion_head:
            self._freeze_module(self.motion_head, 'motion_head')

        if freeze_occ_head and self.with_occ_head:
            self._freeze_module(self.occ_head, 'occ_head')

        if freeze_planning_head and self.with_planning_head:
            self._freeze_module(self.planning_head, 'planning_head')
            # planning_head를 freeze했지만 vlm_proj만 학습하고 싶은 경우
            if unfreeze_vlm_proj and hasattr(self.planning_head, 'vlm_proj'):
                self._unfreeze_module(self.planning_head.vlm_proj, 'planning_head.vlm_proj')
        elif freeze_vlm_proj and self.with_planning_head:
            # planning_head 전체가 아닌 vlm_proj만 freeze
            if hasattr(self.planning_head, 'vlm_proj'):
                self._freeze_module(self.planning_head.vlm_proj, 'planning_head.vlm_proj')

    def _freeze_module(self, module, module_name):
        """모듈의 모든 파라미터를 freeze합니다."""
        frozen_count = 0
        for param in module.parameters():
            param.requires_grad = False
            frozen_count += 1
        print(f"[VlmE2E] Frozen {module_name}: {frozen_count} parameters")

    def _unfreeze_module(self, module, module_name):
        """모듈의 모든 파라미터를 unfreeze합니다."""
        unfrozen_count = 0
        for param in module.parameters():
            param.requires_grad = True
            unfrozen_count += 1
        print(f"[VlmE2E] Unfrozen {module_name}: {unfrozen_count} parameters")

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

    def _ensure_vlm_initialized(self):
        """VLM이 초기화되지 않았으면 초기화 (lazy initialization).

        DDP 환경에서 각 프로세스가 처음 forward할 때 VLM을 로드합니다.
        학습 GPU를 자동으로 감지하고 남는 GPU에 VLM을 할당합니다.
        """
        if not self._vlm_initialized:
            vlm_manager = get_vlm_manager(vlm_device=self._vlm_device)
            vlm_manager.initialize()
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
        self._ensure_vlm_initialized()

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
            # Skip loss if frozen
            if not self._freeze_seg_head:
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
            # Skip loss if frozen
            if not self._freeze_motion_head:
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
            # Skip loss if frozen
            if not self._freeze_occ_head:
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
            # Skip loss if frozen
            if not self._freeze_planning_head:
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
        vlm_text = vlm_output['text']

        # VLM 출력 저장
        result[0]['vlm'] = dict(
            text=vlm_text,
            prompt=self._vlm_prompt,
        )

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
                vlm_text=vlm_text,  # planning 결과에도 VLM 텍스트 포함
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