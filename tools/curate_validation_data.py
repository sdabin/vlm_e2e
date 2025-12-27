#!/usr/bin/env python
"""
nuScenes Validation 데이터 Curation 도구

이 스크립트는 nuScenes validation 데이터에서 특정 scene만 선택하여
curated validation pkl 파일을 생성합니다.

사용법:
1. Scene 목록 확인:
   python tools/curate_validation_data.py --mode list

2. 특정 scene으로 curated pkl 생성:
   python tools/curate_validation_data.py --mode curate \
       --scenes scene_token1 scene_token2 \
       --output data/infos/nuscenes_infos_temporal_val_curated.pkl

   또는 영상 파일 디렉토리에서 scene_token 추출:
   python tools/curate_validation_data.py --mode curate \
       --video_dir /path/to/videos \
       --output data/infos/nuscenes_infos_temporal_val_curated.pkl
   (파일명 형식: {scene_token}_{description}.mp4)

3. Scene 정보 상세 보기:
   python tools/curate_validation_data.py --mode detail --scenes scene_token1
"""

import argparse
import pickle
from collections import defaultdict
from datetime import datetime
import os


def extract_scene_tokens_from_dir(video_dir):
    """디렉토리 내 영상 파일명에서 scene_token 추출

    파일명 형식: {scene_token}_{description}.{ext}
    예: 3ada261efee347cba2e7557794f1aec8_공사살짝회피.mp4

    Args:
        video_dir: 영상 파일들이 있는 디렉토리 경로

    Returns:
        scene_token 목록
    """
    if not os.path.isdir(video_dir):
        raise ValueError(f"Directory not found: {video_dir}")

    scene_tokens = []
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov')

    for filename in os.listdir(video_dir):
        if filename.lower().endswith(video_extensions):
            # 첫 번째 _ 이전 부분이 scene_token
            parts = filename.split('_', 1)
            if len(parts) >= 1:
                scene_token = parts[0]
                # 확장자만 있는 경우 제외 (언더스코어 없이 scene_token만 있는 경우)
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
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
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
    """모든 scene 정보를 출력

    Args:
        ann_file: annotation pkl 파일 경로
        verbose: 상세 정보 출력 여부
    """
    data = load_pkl(ann_file)
    infos = data['infos']

    # scene_token별로 그룹화
    scenes = defaultdict(list)
    for info in infos:
        scenes[info['scene_token']].append(info)

    print(f"\n{'='*80}")
    print(f"Total: {len(scenes)} scenes, {len(infos)} samples")
    print(f"{'='*80}\n")

    # 정렬된 scene 목록 출력
    print(f"{'No.':<5} {'Scene Token':<40} {'Samples':<10} {'Objects (avg)':<15}")
    print(f"{'-'*70}")

    scene_list = []
    for idx, (scene_token, samples) in enumerate(sorted(scenes.items(), key=lambda x: x[1][0]['timestamp'])):
        # 샘플 수
        num_samples = len(samples)

        # 평균 객체 수 계산
        total_objects = 0
        for sample in samples:
            if 'gt_boxes' in sample and sample['gt_boxes'] is not None:
                total_objects += len(sample['gt_boxes'])
        avg_objects = total_objects / num_samples if num_samples > 0 else 0

        print(f"{idx+1:<5} {scene_token:<40} {num_samples:<10} {avg_objects:<15.1f}")
        scene_list.append(scene_token)

        if verbose:
            # 시간 범위
            timestamps = [s['timestamp'] for s in samples]
            start_time = timestamp_to_datetime(min(timestamps))
            end_time = timestamp_to_datetime(max(timestamps))
            print(f"      Time: {start_time} ~ {end_time}")

            # 객체 클래스 분포
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
    """특정 scene의 상세 정보 출력

    Args:
        ann_file: annotation pkl 파일 경로
        scene_tokens: 조회할 scene_token 목록
    """
    data = load_pkl(ann_file)
    infos = data['infos']

    # scene_token별로 그룹화
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

        # 시간 범위
        timestamps = [s['timestamp'] for s in samples]
        start_time = timestamp_to_datetime(min(timestamps))
        end_time = timestamp_to_datetime(max(timestamps))
        print(f"Time range: {start_time} ~ {end_time}")

        # 객체 통계
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

        # 샘플 목록
        print(f"\nSamples ({len(samples)}):")
        print(f"{'Frame':<8} {'Token':<40} {'Objects':<10}")
        print(f"{'-'*60}")
        for sample in samples:
            frame_idx = sample.get('frame_idx', 'N/A')
            token = sample['token']
            num_obj = len(sample.get('gt_boxes', [])) if sample.get('gt_boxes') is not None else 0
            print(f"{frame_idx:<8} {token:<40} {num_obj:<10}")


def curate_scenes(ann_file, scene_tokens, output_file):
    """선택한 scene만 포함하는 새 pkl 파일 생성

    Args:
        ann_file: 원본 annotation pkl 파일 경로
        scene_tokens: 포함할 scene_token 목록
        output_file: 출력 pkl 파일 경로
    """
    data = load_pkl(ann_file)
    original_count = len(data['infos'])

    # scene_token으로 필터링
    scene_token_set = set(scene_tokens)
    filtered_infos = [
        info for info in data['infos']
        if info['scene_token'] in scene_token_set
    ]

    if not filtered_infos:
        print(f"Error: No samples found for the given scene tokens")
        print(f"Given tokens: {scene_tokens}")
        return

    # 시간순 정렬
    filtered_infos = sorted(filtered_infos, key=lambda x: x['timestamp'])

    # 통계 출력
    print(f"\nCuration Summary:")
    print(f"  Original samples: {original_count}")
    print(f"  Filtered samples: {len(filtered_infos)}")
    print(f"  Reduction: {100 * (1 - len(filtered_infos) / original_count):.1f}%")

    # scene별 샘플 수
    scene_counts = defaultdict(int)
    for info in filtered_infos:
        scene_counts[info['scene_token']] += 1

    print(f"\nIncluded scenes ({len(scene_counts)}):")
    for token, count in sorted(scene_counts.items()):
        print(f"  {token}: {count} samples")

    # 누락된 scene 확인
    missing_scenes = scene_token_set - set(scene_counts.keys())
    if missing_scenes:
        print(f"\nWarning: Following scenes not found:")
        for token in missing_scenes:
            print(f"  {token}")

    # 새 데이터 구조 생성
    new_data = {
        'infos': filtered_infos,
        'metadata': data.get('metadata', {'version': 'curated'})
    }

    # 저장
    save_pkl(new_data, output_file)
    print(f"\nCurated pkl file saved: {output_file}")
    print(f"Use this in config: ann_file=\"{output_file}\"")


def main():
    parser = argparse.ArgumentParser(
        description='nuScenes Validation 데이터 Curation 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scene 목록 확인
  python tools/curate_validation_data.py --mode list

  # 상세 정보 포함
  python tools/curate_validation_data.py --mode list --verbose

  # 특정 scene 상세 정보
  python tools/curate_validation_data.py --mode detail --scenes scene_token1 scene_token2

  # Curated pkl 생성 (scene_token 직접 지정)
  python tools/curate_validation_data.py --mode curate \\
      --scenes scene_token1 scene_token2 \\
      --output data/infos/nuscenes_infos_temporal_val_curated.pkl

  # Curated pkl 생성 (영상 디렉토리에서 scene_token 추출)
  python tools/curate_validation_data.py --mode curate \\
      --video_dir /path/to/curated_videos \\
      --output data/infos/nuscenes_infos_temporal_val_curated.pkl
        """
    )

    parser.add_argument(
        '--mode',
        choices=['list', 'detail', 'curate'],
        required=True,
        help='실행 모드: list(scene 목록), detail(상세 정보), curate(pkl 생성)'
    )
    parser.add_argument(
        '--ann_file',
        default='data/infos/nuscenes_infos_temporal_val.pkl',
        help='원본 annotation pkl 파일 경로'
    )
    parser.add_argument(
        '--scenes',
        nargs='+',
        help='scene_token 목록 (curate/detail 모드에서 사용)'
    )
    parser.add_argument(
        '--video_dir',
        help='scene_token을 추출할 영상 파일 디렉토리 (curate 모드, 파일명 형식: {scene_token}_{description}.mp4)'
    )
    parser.add_argument(
        '--output',
        default='data/infos/nuscenes_infos_temporal_val_curated.pkl',
        help='출력 pkl 파일 경로 (curate 모드에서 사용)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 정보 출력 (list 모드에서 사용)'
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


if __name__ == '__main__':
    main()
