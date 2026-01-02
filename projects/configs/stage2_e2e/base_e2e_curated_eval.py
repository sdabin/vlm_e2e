_base_ = ["./base_e2e.py"]

# Curated validation 데이터 사용
# curated_eval_data 디렉토리의 영상에 해당하는 scene들
# 33 scenes, 1321 samples

info_root = "data/infos/"
ann_file_val = info_root + "nuscenes_infos_temporal_val_curated.pkl"
ann_file_test = info_root + "nuscenes_infos_temporal_val_curated.pkl"

data = dict(
    val=dict(
        ann_file=ann_file_val,
    ),
    test=dict(
        ann_file=ann_file_test,
    ),
)
