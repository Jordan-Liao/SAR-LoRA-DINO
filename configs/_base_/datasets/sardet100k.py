import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
data_root = os.environ.get("SARDET100K_ROOT", str(_REPO_ROOT / "data" / "sardet100k"))
if not data_root.endswith(("/", "\\")):
    data_root += "/"

backend_args = None

metainfo = dict(
    classes=("ship", "aircraft", "car", "tank", "bridge", "harbor"),
    palette=[
        (220, 20, 60),
        (0, 0, 230),
        (106, 0, 228),
        (0, 182, 0),
        (200, 182, 0),
        (0, 182, 200),
    ],
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(800, 800), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='Annotations/train.json',
        data_prefix=dict(img='JPEGImages/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='Annotations/val.json',
        data_prefix=dict(img='JPEGImages/val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)


test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root=data_root,
        ann_file='Annotations/test.json',
        data_prefix=dict(img='JPEGImages/test/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'Annotations/test.json',
    metric='bbox',
    classwise = True,
    format_only=False,
    backend_args=backend_args)
