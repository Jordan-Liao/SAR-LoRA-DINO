_base_ = [
    "../../_base_/models/retinanet_r50_fpn.py",
    "../../_base_/datasets/sardet100k.py",
    "../../_base_/schedules/1x.py",
    "../../_base_/default_runtime.py",
]

num_classes = 6
model = dict(bbox_head=dict(num_classes=num_classes), test_cfg=dict(score_thr=0.0))

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
default_hooks = dict(logger=dict(interval=20), checkpoint=dict(interval=1))
