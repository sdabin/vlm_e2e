_base_ = ["./vlm_e2e.py"]

# Curated validation 데이터 사용
# 5개 scene, 198 샘플 (원본 6019개의 3.3%)
# - 2ed0fcbfc214478ca3b3ce013e7723ba: 39 samples (avg 86.8 objects, 복잡)
# - 848ac962547c4508b8f3b0fcc8d53270: 40 samples (avg 42.0 objects, 중간)
# - 91c071bcc1ad4fa1b555399e1cfbab79: 40 samples (avg 18.4 objects, 중간)
# - 265f002f02d447ad9074813292eef75e: 40 samples (avg 2.4 objects, 단순)
# - c3ab8ee2c1a54068a72d7eb4cf22e43d: 39 samples (avg 19.1 objects, 중간)

info_root = "data/infos/"
ann_file_val = info_root + "nuscenes_infos_temporal_val_tiny.pkl"
ann_file_test = info_root + "nuscenes_infos_temporal_val_tiny.pkl"

data = dict(
    val=dict(
        ann_file=ann_file_val,
    ),
    test=dict(
        ann_file=ann_file_test,
    ),
)
