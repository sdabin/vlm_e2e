#!/usr/bin/env python
"""
nuScenes Validation 데이터 VLM Curation 도구

이 스크립트는 nuScenes validation 데이터에서 특정 scene만 선택하고,
VLM을 사용하여 각 샘플에 대한 응답을 미리 생성하여 저장합니다.

지원 VLM 모델:
- Qwen2.5-VL: Qwen/Qwen2.5-VL-3B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct
- LLaVA: llava-hf/llava-1.5-7b-hf

사용법:
1. Scene 목록 확인:
   python tools/curate_validation_data_vlm.py --mode list

2. VLM 응답 포함 curated pkl 생성:
   python tools/curate_validation_data_vlm.py --mode vlm_curate \
       --video_dir /path/to/videos \
       --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
       --prompt "Describe the driving scenario in this image." \
       --output_dir data/infos/

3. 시각화 포함:
   python tools/curate_validation_data_vlm.py --mode vlm_curate \
       --video_dir /path/to/videos \
       --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \
       --visualize
"""

import argparse
import pickle
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# VLM imports (lazy loading to avoid import errors when not using VLM)
def get_vlm_imports():
    """Lazy import VLM-related modules"""
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    return torch, AutoProcessor, AutoTokenizer


# ============================================================================
# VLM Model Interfaces
# ============================================================================

def parse_device_map(device: str) -> dict:
    """
    Parse device string to create device_map for multi-GPU support.

    Examples:
        "cuda" -> None (single GPU, default)
        "cuda:0" -> None (single GPU)
        "cuda:0,1" -> {"": [0, 1]} (multi-GPU)
        "auto" -> "auto" (automatic distribution)
    """
    if device == "auto":
        return "auto"

    if "," in device:
        # Multi-GPU: "cuda:0,1" -> [0, 1]
        device_str = device.replace("cuda:", "")
        gpu_ids = [int(x.strip()) for x in device_str.split(",")]
        return gpu_ids

    return None  # Single GPU


class VLMInterface(ABC):
    """VLM 모델 공통 인터페이스"""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device_map = parse_device_map(device)
        self.is_multi_gpu = isinstance(self.device_map, list) or self.device_map == "auto"

    @abstractmethod
    def load_model(self):
        """모델 로드"""
        pass

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 2048) -> Dict[str, Any]:
        """
        이미지와 프롬프트로 응답 생성

        Returns:
            {'text': str, 'token_ids': list}
        """
        pass

    def get_model_device(self):
        """모델이 로드된 디바이스 반환"""
        if self.model is None:
            return None
        if self.is_multi_gpu:
            # Multi-GPU: 첫 번째 파라미터의 디바이스 반환
            return next(self.model.parameters()).device
        return self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device

    def unload_model(self):
        """모델 언로드하여 GPU 메모리 해제"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        torch, _, _ = get_vlm_imports()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LLaVAModel(VLMInterface):
    """LLaVA 모델 인터페이스"""

    def load_model(self):
        torch, AutoProcessor, AutoTokenizer = get_vlm_imports()
        from transformers import LlavaForConditionalGeneration

        dtype = torch.float16 if self.dtype == "float16" else torch.float32

        print(f"Loading LLaVA model: {self.model_id}")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device if self.device == "auto" else None,
        )
        if self.device != "auto":
            self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=False)
        self.processor = AutoProcessor.from_pretrained(self.model_id, tokenizer=self.tokenizer)

        print(f"LLaVA model loaded on {self.device}")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 2048) -> Dict[str, Any]:
        torch, _, _ = get_vlm_imports()

        # LLaVA conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode - get only the generated part
        input_len = inputs['input_ids'].shape[1]
        generated_ids = output_ids[0, input_len:]
        generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

        return {
            'text': generated_text.strip(),
            'token_ids': generated_ids.cpu().tolist(),
        }


class Qwen25VLModel(VLMInterface):
    """Qwen2.5-VL 모델 인터페이스"""

    def load_model(self):
        torch, AutoProcessor, _ = get_vlm_imports()
        from transformers import Qwen2_5_VLForConditionalGeneration

        # Qwen2.5-VL requires bfloat16 for correct output
        if self.dtype == "bfloat16" or self.dtype == "float16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        print(f"Loading Qwen2.5-VL model: {self.model_id}")

        # Multi-GPU support
        if self.is_multi_gpu:
            if isinstance(self.device_map, list):
                # Create device map for specific GPUs
                max_memory = {gpu_id: "24GiB" for gpu_id in self.device_map}
                print(f"  Multi-GPU mode: GPUs {self.device_map}")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                    device_map="auto",
                    max_memory=max_memory,
                )
            else:
                # Auto device map
                print(f"  Auto device map mode")
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                    device_map="auto",
                )
        else:
            # Single GPU
            print(f"  Single GPU mode: {self.device}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                attn_implementation="eager",
            )
            if self.device != "auto":
                self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        print(f"Qwen2.5-VL model loaded (dtype: {dtype})")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 2048,
                 image_path: str = None) -> Dict[str, Any]:
        """
        Generate response from image and prompt.
        """
        torch, _, _ = get_vlm_imports()
        from qwen_vl_utils import process_vision_info

        # Use file path if provided
        if image_path:
            image_content = {"type": "image", "image": image_path}
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image.save(f.name)
                image_content = {"type": "image", "image": f.name}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs
        device = self.get_model_device()
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode
        generated_ids = [
            output_ids[0][inputs['input_ids'].shape[1]:]
        ]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return {
            'text': generated_text.strip(),
            'token_ids': generated_ids[0].cpu().tolist(),
        }


class Qwen3VLModel(VLMInterface):
    """Qwen3-VL 모델 인터페이스"""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        super().__init__(model_id, device, dtype)
        # Check if this is a thinking model
        self.is_thinking_model = "thinking" in model_id.lower()

    def load_model(self):
        torch, AutoProcessor, _ = get_vlm_imports()
        from transformers import Qwen3VLForConditionalGeneration

        # Qwen3-VL requires bfloat16 for correct output
        if self.dtype == "bfloat16" or self.dtype == "float16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        print(f"Loading Qwen3-VL model: {self.model_id}")
        print(f"  Thinking model: {self.is_thinking_model}")

        # Check for MoE model (A3B, A8B suffix indicates MoE)
        is_moe_model = any(x in self.model_id for x in ["-A3B", "-A8B", "-a3b", "-a8b"])
        if is_moe_model:
            print(f"  MoE model detected: special loading required")

        # Multi-GPU support
        if self.is_multi_gpu:
            if isinstance(self.device_map, list):
                # Create device map for specific GPUs
                max_memory = {gpu_id: "24GiB" for gpu_id in self.device_map}
                print(f"  Multi-GPU mode: GPUs {self.device_map}")

                # For MoE models, use balanced device map
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    attn_implementation="eager",  # sdpa or flash_attention_2 may not be available
                    device_map="balanced",  # balanced is better for MoE
                    max_memory=max_memory,
                    trust_remote_code=True,
                )
            else:
                # Auto device map
                print(f"  Auto device map mode")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                    attn_implementation="eager",
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            # Single GPU
            print(f"  Single GPU mode: {self.device}")
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                attn_implementation="eager",
                trust_remote_code=True,
            )
            if self.device != "auto":
                self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        print(f"Qwen3-VL model loaded (dtype: {dtype})")

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 2048,
                 image_path: str = None) -> Dict[str, Any]:
        """
        Generate response from image and prompt.
        For thinking models, enables thinking mode and strips thinking content from output.
        """
        torch, _, _ = get_vlm_imports()
        from qwen_vl_utils import process_vision_info

        # Qwen3-VL works better with file paths
        if image_path:
            image_content = {"type": "image", "image": image_path}
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                image.save(f.name)
                image_content = {"type": "image", "image": f.name}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template - for thinking models, enable thinking
        if self.is_thinking_model:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True
            )
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)

        # Process inputs - use get_model_device() for multi-GPU support
        device = self.get_model_device()
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode - use batch_decode for full output
        generated_ids = [
            output_ids[0][inputs['input_ids'].shape[1]:]
        ]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # For thinking models, strip the thinking content
        if self.is_thinking_model:
            generated_text = strip_thinking_tags(generated_text)

        # Check for garbled text (indicates decoding issues)
        if is_garbled_text(generated_text):
            print(f"  Warning: Detected potentially garbled text output. First 100 chars: {generated_text[:100]}")
            print(f"  This may indicate model/tokenizer mismatch or memory issues.")

        return {
            'text': generated_text.strip(),
            'token_ids': generated_ids[0].cpu().tolist(),
        }


def get_vlm_model(model_id: str, device: str = "cuda", dtype: str = "float16") -> VLMInterface:
    """모델 ID에 따라 적절한 VLM 인터페이스 반환"""
    model_id_lower = model_id.lower()

    if "llava" in model_id_lower:
        return LLaVAModel(model_id, device, dtype)
    elif "qwen3" in model_id_lower:
        return Qwen3VLModel(model_id, device, dtype)
    elif "qwen2" in model_id_lower or "qwen" in model_id_lower:
        return Qwen25VLModel(model_id, device, dtype)
    else:
        raise ValueError(f"Unknown VLM model: {model_id}. Supported: LLaVA, Qwen2.5-VL, Qwen3-VL")


def get_model_short_name(model_id: str) -> str:
    """모델 ID에서 짧은 이름 추출"""
    # Qwen/Qwen2.5-VL-3B-Instruct -> qwen2.5-vl-3b
    # llava-hf/llava-1.5-7b-hf -> llava-1.5-7b
    name = model_id.split("/")[-1].lower()
    name = re.sub(r'-instruct$', '', name)
    name = re.sub(r'-hf$', '', name)
    return name


# ============================================================================
# Visualization
# ============================================================================

def strip_thinking_tags(text: str) -> str:
    """
    Thinking 모델의 <think>...</think> 태그 제거.
    태그 내용을 제거하고 최종 응답만 반환.
    """
    import re

    # <think>...</think> 태그와 내용 제거
    # 패턴: <think> 또는 <|begin_of_thought|> 등 다양한 thinking 태그 지원
    patterns = [
        r'<think>.*?</think>',  # Standard think tags
        r'<\|begin_of_thought\|>.*?<\|end_of_thought\|>',  # Alternative format
        r'<thinking>.*?</thinking>',  # Another format
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.DOTALL | re.IGNORECASE)

    # 남은 닫는 태그 제거 (열린 태그 없이 닫는 태그만 있는 경우)
    result = re.sub(r'</think>', '', result, flags=re.IGNORECASE)
    result = re.sub(r'</thinking>', '', result, flags=re.IGNORECASE)
    result = re.sub(r'<\|end_of_thought\|>', '', result, flags=re.IGNORECASE)

    # 여러 공백/줄바꿈 정리
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip()

    return result


def is_garbled_text(text: str) -> bool:
    """
    텍스트가 깨졌는지 (garbled) 감지.
    MoE 모델의 가중치가 제대로 로드되지 않았을 때 발생하는 완전히 무의미한 출력만 감지.
    정상적인 영어 텍스트는 통과시킴.
    """
    if not text or len(text) < 50:
        return False

    import re

    words = text.split()
    if len(words) < 10:
        return False

    # 휴리스틱 점수 - 매우 보수적으로 설정
    score = 0

    # 1. 극단적으로 긴 단어 (20자 이상) - 거의 확실히 깨진 텍스트
    very_long_words = [w for w in words if len(re.sub(r'[^a-zA-Z]', '', w)) >= 20]
    if len(very_long_words) >= 3:
        score += 3

    # 2. 연속된 자음만 있는 긴 단어 (영어에서는 거의 불가능)
    consonant_only = re.findall(r'\b[bcdfghjklmnpqrstvwxyz]{6,}\b', text.lower())
    if len(consonant_only) >= 3:
        score += 3

    # 3. 동일 문자 4번 이상 반복 (예: "aaaa", "xxxx")
    repeated_chars = re.findall(r'(.)\1{3,}', text.lower())
    if len(repeated_chars) >= 2:
        score += 2

    # 4. 완전 무작위 문자열 패턴 - 대소문자가 매우 빈번하게 바뀜 (5자 이내)
    random_case = re.findall(r'[a-z][A-Z][a-z][A-Z][a-z]', text)
    if len(random_case) >= 5:
        score += 2

    # 임계값: 4점 이상이면 garbled (더 보수적)
    return score >= 4


def sanitize_text_for_display(text: str) -> str:
    """비ASCII 문자를 제거하거나 대체하여 폰트 호환성 문제 방지"""
    # ASCII 및 일반적인 라틴 확장 문자만 유지
    # 중국어, 일본어, 한국어 등 폰트가 지원하지 않는 문자 제거
    sanitized = []
    for char in text:
        # ASCII printable characters (32-126) + newline, tab
        if 32 <= ord(char) <= 126 or char in '\n\t':
            sanitized.append(char)
        # Extended Latin characters (common accented letters)
        elif 192 <= ord(char) <= 255:
            sanitized.append(char)
        # Replace other characters with space (to avoid breaking words)
        else:
            # Skip consecutive non-ASCII chars (avoid multiple spaces)
            if sanitized and sanitized[-1] != ' ':
                sanitized.append(' ')

    # Clean up multiple spaces
    result = ''.join(sanitized)
    result = ' '.join(result.split())
    return result


def create_visualization(image: Image.Image, prompt: str, response: str,
                        output_path: str, sample_token: str):
    """이미지, 프롬프트, 응답을 하나의 이미지로 시각화하여 저장"""

    # Thinking 모델의 <think> 태그 제거 후 sanitize
    response = strip_thinking_tags(response)

    # Garbled text 감지 및 대체
    if is_garbled_text(response):
        response = "[ERROR: Garbled text detected - model output may be corrupted. Please check model/tokenizer configuration.]"

    # 텍스트 sanitize (폰트 호환성 문제 방지)
    prompt = sanitize_text_for_display(prompt)
    response = sanitize_text_for_display(response)

    # 이미지 크기
    img_width, img_height = image.size

    # 폰트 설정 (기본 폰트 사용)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_bold = font

    # 임시 이미지로 Draw 객체 생성 (텍스트 측정용)
    temp_img = Image.new('RGB', (img_width, 100))
    temp_draw = ImageDraw.Draw(temp_img)
    
    margin = 10
    max_width = img_width - 2 * margin
    line_height = 18
    section_spacing = 10

    # 텍스트 줄바꿈 (정확한 너비 계산)
    prompt_lines = wrap_text(prompt, font, max_width, draw=temp_draw)
    response_lines = wrap_text(response, font, max_width, draw=temp_draw)

    # 동적으로 텍스트 영역 높이 계산
    base_height = 10  # 상단 여백
    sample_height = 25  # Sample token
    prompt_header_height = 20  # "Prompt:" 헤더
    prompt_text_height = len(prompt_lines) * line_height
    prompt_spacing = section_spacing
    response_header_height = 20  # "Response:" 헤더
    response_text_height = len(response_lines) * line_height
    bottom_margin = 10  # 하단 여백

    text_height = (base_height + sample_height + prompt_header_height + 
                   prompt_text_height + prompt_spacing + response_header_height + 
                   response_text_height + bottom_margin)
    
    total_height = img_height + text_height

    # 새 이미지 생성 (흰색 배경)
    vis_image = Image.new('RGB', (img_width, total_height), (255, 255, 255))

    # 원본 이미지 붙이기
    vis_image.paste(image, (0, 0))

    # 텍스트 그리기
    draw = ImageDraw.Draw(vis_image)

    y_offset = img_height + base_height

    # Sample token
    draw.text((margin, y_offset), f"Sample: {sample_token}", fill=(100, 100, 100), font=font)
    y_offset += sample_height

    # Prompt
    draw.text((margin, y_offset), "Prompt:", fill=(0, 0, 200), font=font_bold)
    y_offset += prompt_header_height

    # Wrap prompt text (모든 줄 표시)
    for line in prompt_lines:
        draw.text((margin, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += line_height

    y_offset += prompt_spacing

    # Response
    draw.text((margin, y_offset), "Response:", fill=(0, 150, 0), font=font_bold)
    y_offset += response_header_height

    # Wrap response text (모든 줄 표시)
    for line in response_lines:
        draw.text((margin, y_offset), line, fill=(0, 0, 0), font=font)
        y_offset += line_height

    # 저장
    vis_image.save(output_path)


def wrap_text(text: str, font, max_width: int, draw: ImageDraw.Draw = None) -> List[str]:
    """텍스트를 주어진 너비에 맞게 줄바꿈 (정확한 폰트 너비 계산)"""
    if draw is None:
        # 임시 이미지로 Draw 객체 생성 (폰트 측정용)
        temp_img = Image.new('RGB', (100, 100))
        draw = ImageDraw.Draw(temp_img)
    
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        # 현재 줄에 단어 추가
        test_line = ' '.join(current_line + [word])
        
        # PIL의 textbbox를 사용하여 정확한 너비 계산
        try:
            # textbbox는 (left, top, right, bottom) 반환
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            # 구버전 PIL에서는 textlength 사용
            try:
                text_width = draw.textlength(test_line, font=font)
            except AttributeError:
                # 최후의 수단: 대략적 계산
                text_width = len(test_line) * 7
        
        if text_width > max_width:
            if len(current_line) > 0:
                # 현재 줄 저장하고 새 줄 시작
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                # 단어 자체가 너무 길면 강제로 줄바꿈
                # 긴 단어를 여러 줄로 나누기
                if len(word) > 50:  # 매우 긴 단어인 경우
                    # 단어를 최대한 나눔
                    chunk_size = max(1, max_width // 7)  # 대략적인 문자 수
                    for i in range(0, len(word), chunk_size):
                        lines.append(word[i:i+chunk_size])
                    current_line = []
                else:
                    lines.append(word)
                    current_line = []
        else:
            current_line.append(word)

    if current_line:
        lines.append(' '.join(current_line))

    return lines


# ============================================================================
# Original Functions (from curate_validation_data.py)
# ============================================================================

def extract_scene_tokens_from_dir(video_dir):
    """디렉토리 내 영상 파일명에서 scene_token 추출"""
    if not os.path.isdir(video_dir):
        raise ValueError(f"Directory not found: {video_dir}")

    scene_tokens = []
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')

    for filename in os.listdir(video_dir):
        if filename.lower().endswith(video_extensions):
            parts = filename.split('_', 1)
            if len(parts) >= 1:
                scene_token = parts[0]
                if '.' in scene_token:
                    scene_token = os.path.splitext(scene_token)[0]
                scene_tokens.append(scene_token)
                print(f"  Found: {filename} -> {scene_token}")

    if not scene_tokens:
        raise ValueError(f"No video files found in {video_dir}")

    print(f"\nExtracted {len(scene_tokens)} scene tokens from {video_dir}")
    return scene_tokens


def load_pkl(ann_file):
    """pkl 파일 로드"""
    print(f"Loading {ann_file}...")
    with open(ann_file, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data['infos'])} samples")
    return data


def save_pkl(data, output_file):
    """pkl 파일 저장"""
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved to {output_file}")


def timestamp_to_datetime(timestamp):
    """마이크로초 timestamp를 datetime 문자열로 변환"""
    try:
        dt = datetime.fromtimestamp(timestamp / 1e6)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return 'N/A'


def list_scenes(ann_file, verbose=False):
    """모든 scene 정보를 출력"""
    data = load_pkl(ann_file)
    infos = data['infos']

    scenes = defaultdict(list)
    for info in infos:
        scenes[info['scene_token']].append(info)

    print(f"\n{'='*80}")
    print(f"Total: {len(scenes)} scenes, {len(infos)} samples")
    print(f"{'='*80}\n")

    print(f"{'No.':<5} {'Scene Token':<40} {'Samples':<10} {'Objects (avg)':<15}")
    print(f"{'-'*70}")

    scene_list = []
    for idx, (scene_token, samples) in enumerate(sorted(scenes.items(), key=lambda x: x[1][0]['timestamp'])):
        num_samples = len(samples)
        total_objects = 0
        for sample in samples:
            if 'gt_boxes' in sample and sample['gt_boxes'] is not None:
                total_objects += len(sample['gt_boxes'])
        avg_objects = total_objects / num_samples if num_samples > 0 else 0

        print(f"{idx+1:<5} {scene_token:<40} {num_samples:<10} {avg_objects:<15.1f}")
        scene_list.append(scene_token)

        if verbose:
            timestamps = [s['timestamp'] for s in samples]
            start_time = timestamp_to_datetime(min(timestamps))
            end_time = timestamp_to_datetime(max(timestamps))
            print(f"      Time: {start_time} ~ {end_time}")

            class_counts = defaultdict(int)
            for sample in samples:
                if 'gt_names' in sample and sample['gt_names'] is not None:
                    for name in sample['gt_names']:
                        class_counts[name] += 1
            if class_counts:
                top_classes = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
                class_str = ', '.join([f"{c}:{n}" for c, n in top_classes])
                print(f"      Classes: {class_str}")
            print()

    print(f"\n{'='*80}")
    print("Scene tokens (for copy-paste):")
    print(f"{'='*80}")
    for token in scene_list:
        print(token)

    return scene_list


def detail_scenes(ann_file, scene_tokens):
    """특정 scene의 상세 정보 출력"""
    data = load_pkl(ann_file)
    infos = data['infos']

    scenes = defaultdict(list)
    for info in infos:
        scenes[info['scene_token']].append(info)

    for scene_token in scene_tokens:
        if scene_token not in scenes:
            print(f"Warning: Scene '{scene_token}' not found")
            continue

        samples = sorted(scenes[scene_token], key=lambda x: x['timestamp'])

        print(f"\n{'='*80}")
        print(f"Scene: {scene_token}")
        print(f"{'='*80}")
        print(f"Number of samples: {len(samples)}")

        timestamps = [s['timestamp'] for s in samples]
        start_time = timestamp_to_datetime(min(timestamps))
        end_time = timestamp_to_datetime(max(timestamps))
        print(f"Time range: {start_time} ~ {end_time}")

        total_objects = 0
        class_counts = defaultdict(int)
        for sample in samples:
            if 'gt_boxes' in sample and sample['gt_boxes'] is not None:
                total_objects += len(sample['gt_boxes'])
            if 'gt_names' in sample and sample['gt_names'] is not None:
                for name in sample['gt_names']:
                    class_counts[name] += 1

        print(f"Total objects: {total_objects}")
        print(f"Average objects per frame: {total_objects / len(samples):.1f}")

        if class_counts:
            print("\nObject class distribution:")
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                print(f"  {cls}: {count}")

        print(f"\nSamples ({len(samples)}):")
        print(f"{'Frame':<8} {'Token':<40} {'Objects':<10}")
        print(f"{'-'*60}")
        for sample in samples:
            frame_idx = sample.get('frame_idx', 'N/A')
            token = sample['token']
            num_obj = len(sample.get('gt_boxes', [])) if sample.get('gt_boxes') is not None else 0
            print(f"{frame_idx:<8} {token:<40} {num_obj:<10}")


def curate_scenes(ann_file, scene_tokens, output_file):
    """선택한 scene만 포함하는 새 pkl 파일 생성 (기존 기능)"""
    data = load_pkl(ann_file)
    original_count = len(data['infos'])

    scene_token_set = set(scene_tokens)
    filtered_infos = [
        info for info in data['infos']
        if info['scene_token'] in scene_token_set
    ]

    if not filtered_infos:
        print(f"Error: No samples found for the given scene tokens")
        print(f"Given tokens: {scene_tokens}")
        return

    filtered_infos = sorted(filtered_infos, key=lambda x: x['timestamp'])

    print(f"\nCuration Summary:")
    print(f"  Original samples: {original_count}")
    print(f"  Filtered samples: {len(filtered_infos)}")
    print(f"  Reduction: {100 * (1 - len(filtered_infos) / original_count):.1f}%")

    scene_counts = defaultdict(int)
    for info in filtered_infos:
        scene_counts[info['scene_token']] += 1

    print(f"\nIncluded scenes ({len(scene_counts)}):")
    for token, count in sorted(scene_counts.items()):
        print(f"  {token}: {count} samples")

    missing_scenes = scene_token_set - set(scene_counts.keys())
    if missing_scenes:
        print(f"\nWarning: Following scenes not found:")
        for token in missing_scenes:
            print(f"  {token}")

    new_data = {
        'infos': filtered_infos,
        'metadata': data.get('metadata', {'version': 'curated'})
    }

    save_pkl(new_data, output_file)
    print(f"\nCurated pkl file saved: {output_file}")
    print(f"Use this in config: ann_file=\"{output_file}\"")


# ============================================================================
# VLM Curate Mode
# ============================================================================

def vlm_curate_scenes(
    ann_file: str,
    scene_tokens: List[str],
    output_dir: str,
    data_root: str,
    vlm_model_id: str,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    visualize: bool = False,
):
    """VLM 응답을 포함하여 curated pkl 생성"""

    # 데이터 로드
    data = load_pkl(ann_file)
    original_count = len(data['infos'])

    # Scene 필터링
    scene_token_set = set(scene_tokens)
    filtered_infos = [
        info for info in data['infos']
        if info['scene_token'] in scene_token_set
    ]

    if not filtered_infos:
        print(f"Error: No samples found for the given scene tokens")
        return

    # 시간순 정렬
    filtered_infos = sorted(filtered_infos, key=lambda x: x['timestamp'])

    print(f"\nCuration Summary:")
    print(f"  Original samples: {original_count}")
    print(f"  Filtered samples: {len(filtered_infos)}")

    # 출력 파일명 생성
    model_short_name = get_model_short_name(vlm_model_id)
    output_file = os.path.join(
        output_dir,
        f"nuscenes_infos_temporal_val_curated_{model_short_name}.pkl"
    )

    # 시각화 디렉토리
    vis_dir = None
    if visualize:
        vis_dir = os.path.join(output_dir, f"vis_{model_short_name}")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  Visualization dir: {vis_dir}")

    # VLM 모델 로드
    print(f"\nLoading VLM model: {vlm_model_id}")
    vlm = get_vlm_model(vlm_model_id, device=device)
    vlm.load_model()

    # 각 샘플에 대해 VLM 응답 생성
    print(f"\nGenerating VLM responses for {len(filtered_infos)} samples...")

    # Scene별 프레임 인덱스 추적
    scene_frame_counter = defaultdict(int)

    for idx, info in enumerate(tqdm(filtered_infos, desc="Processing samples")):
        try:
            # CAM_FRONT 이미지 로드
            cam_front = info['cams']['CAM_FRONT']
            img_path = os.path.join(data_root, cam_front['data_path'])

            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                info['vlm_response'] = None
                continue

            image = Image.open(img_path).convert('RGB')

            # VLM 응답 생성 (OOM 발생 시 토큰 수 줄여서 재시도)
            import torch
            import gc

            response = None
            retry_tokens = [max_new_tokens, max_new_tokens // 2, max_new_tokens // 4, 256]

            for attempt, tokens in enumerate(retry_tokens):
                try:
                    if hasattr(vlm, 'generate') and 'image_path' in vlm.generate.__code__.co_varnames:
                        response = vlm.generate(image, prompt, max_new_tokens=tokens, image_path=img_path)
                    else:
                        response = vlm.generate(image, prompt, max_new_tokens=tokens)
                    break  # 성공하면 루프 종료
                except (RuntimeError, torch.cuda.OutOfMemoryError) as oom_error:
                    error_msg = str(oom_error).lower()
                    if "cuda out of memory" in error_msg or "out of memory" in error_msg:
                        # CUDA 캐시 정리
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        if attempt == len(retry_tokens) - 1:
                            raise  # 마지막 시도도 실패하면 예외 발생
                        # 다음 시도로 진행 (더 적은 토큰으로)
                    else:
                        raise  # OOM이 아닌 다른 에러는 바로 발생

            if response is None:
                raise RuntimeError("Failed to generate response after all retries")

            # 응답 저장
            info['vlm_response'] = {
                'model': vlm_model_id,
                'prompt': prompt,
                'text': response['text'],
                'token_ids': response['token_ids'],
            }

            # 시각화: {scene_token}_{frame_idx}.jpg 형식으로 저장
            if visualize and vis_dir:
                scene_token = info['scene_token']
                frame_idx = scene_frame_counter[scene_token]
                scene_frame_counter[scene_token] += 1
                vis_path = os.path.join(vis_dir, f"{scene_token}_{frame_idx}.jpg")
                create_visualization(image, prompt, response['text'], vis_path, info['token'])

        except Exception as e:
            print(f"Error processing sample {info.get('token', 'unknown')}: {e}")
            info['vlm_response'] = None

    # 모델 언로드
    vlm.unload_model()

    # 결과 저장
    new_data = {
        'infos': filtered_infos,
        'metadata': {
            'version': 'curated_vlm',
            'vlm_model': vlm_model_id,
            'prompt': prompt,
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    save_pkl(new_data, output_file)

    # 통계 출력
    success_count = sum(1 for info in filtered_infos if info.get('vlm_response') is not None)
    print(f"\nVLM Curation Complete:")
    print(f"  Total samples: {len(filtered_infos)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(filtered_infos) - success_count}")
    print(f"  Output: {output_file}")
    if visualize:
        print(f"  Visualizations: {vis_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='nuScenes Validation 데이터 VLM Curation 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scene 목록 확인
  python tools/curate_validation_data_vlm.py --mode list

  # VLM curate (Qwen2.5-VL)
  python tools/curate_validation_data_vlm.py --mode vlm_curate \\
      --video_dir /home/user/UniAD/data/nuscenes/videos/curated_eval_data \\
      --vlm_model "Qwen/Qwen2.5-VL-3B-Instruct" \\
      --output_dir data/infos/

  # VLM curate with visualization
  python tools/curate_validation_data_vlm.py --mode vlm_curate \\
      --video_dir /home/user/UniAD/data/nuscenes/videos/curated_eval_data \\
      --vlm_model "llava-hf/llava-1.5-7b-hf" \\
      --visualize

  # 기존 curate 모드
  python tools/curate_validation_data_vlm.py --mode curate \\
      --video_dir /path/to/videos \\
      --output data/infos/nuscenes_infos_temporal_val_curated.pkl
        """
    )

    parser.add_argument(
        '--mode',
        choices=['list', 'detail', 'curate', 'vlm_curate'],
        required=True,
        help='실행 모드: list, detail, curate, vlm_curate'
    )
    parser.add_argument(
        '--ann_file',
        default='data/infos/nuscenes_infos_temporal_val.pkl',
        help='원본 annotation pkl 파일 경로'
    )
    parser.add_argument(
        '--data_root',
        default='data/nuscenes/',
        help='nuScenes 데이터 루트 경로'
    )
    parser.add_argument(
        '--scenes',
        nargs='+',
        help='scene_token 목록'
    )
    parser.add_argument(
        '--video_dir',
        help='scene_token을 추출할 영상 파일 디렉토리'
    )
    parser.add_argument(
        '--output',
        default='data/infos/nuscenes_infos_temporal_val_curated.pkl',
        help='출력 pkl 파일 경로 (curate 모드)'
    )
    parser.add_argument(
        '--output_dir',
        default='data/infos/',
        help='출력 디렉토리 (vlm_curate 모드)'
    )
    parser.add_argument(
        '--vlm_model',
        default='Qwen/Qwen2.5-VL-3B-Instruct',
        help='VLM 모델 ID (예: Qwen/Qwen2.5-VL-3B-Instruct, llava-hf/llava-1.5-7b-hf)'
    )
    parser.add_argument(
        '--prompt',
        default='Describe the driving scenario in this image.',
        help='VLM에 전달할 프롬프트'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=2048,
        help='VLM 최대 생성 토큰 수 (기본값: 2048, 긴 응답 생성 가능)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        help='VLM 실행 디바이스 (cuda, cpu, auto)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='VLM 결과 시각화 저장'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 정보 출력'
    )

    args = parser.parse_args()

    if args.mode == 'list':
        list_scenes(args.ann_file, verbose=args.verbose)

    elif args.mode == 'detail':
        if not args.scenes:
            parser.error("--scenes is required for detail mode")
        detail_scenes(args.ann_file, args.scenes)

    elif args.mode == 'curate':
        if args.video_dir:
            scene_tokens = extract_scene_tokens_from_dir(args.video_dir)
        elif args.scenes:
            scene_tokens = args.scenes
        else:
            parser.error("--scenes or --video_dir is required for curate mode")
        curate_scenes(args.ann_file, scene_tokens, args.output)

    elif args.mode == 'vlm_curate':
        if args.video_dir:
            scene_tokens = extract_scene_tokens_from_dir(args.video_dir)
        elif args.scenes:
            scene_tokens = args.scenes
        else:
            parser.error("--scenes or --video_dir is required for vlm_curate mode")

        vlm_curate_scenes(
            ann_file=args.ann_file,
            scene_tokens=scene_tokens,
            output_dir=args.output_dir,
            data_root=args.data_root,
            vlm_model_id=args.vlm_model,
            prompt=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            visualize=args.visualize,
        )


if __name__ == '__main__':
    main()
