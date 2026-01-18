import os
import os.path as osp

_base_ = [
    "../../../configs/_base_/models/retinanet_r50_fpn.py",
    "../../../configs/_base_/datasets/SARDet_100k.py",
    "../../../configs/_base_/schedules/schedule_1x.py",
    "../../../configs/_base_/default_runtime.py",
]

num_classes = 6
model = dict(bbox_head=dict(num_classes=num_classes), test_cfg=dict(score_thr=0.0))

repo_root = osp.abspath(os.getcwd())
subset_root = osp.join(repo_root, "data/sardet_subsets")
train_ann_file = osp.join(subset_root, "train_200.json")
val_ann_file = osp.join(subset_root, "val_50.json")

train_dataloader = dict(dataset=dict(ann_file=train_ann_file))
val_dataloader = dict(dataset=dict(ann_file=val_ann_file))
test_dataloader = dict(
    dataset=dict(ann_file=val_ann_file, data_prefix=dict(img="JPEGImages/val/"))
)

val_evaluator = dict(ann_file=val_ann_file)
test_evaluator = dict(ann_file=val_ann_file)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=1, val_interval=1)
default_hooks = dict(logger=dict(interval=20), checkpoint=dict(interval=1))
