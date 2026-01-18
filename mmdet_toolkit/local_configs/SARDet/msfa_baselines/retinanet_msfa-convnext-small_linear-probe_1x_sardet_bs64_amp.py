import os

_base_ = [
    "../../../configs/_base_/models/retinanet_r50_fpn.py",
    "../../../configs/_base_/datasets/SARDet_100k.py",
    "../../../configs/_base_/schedules/schedule_1x.py",
    "../../../configs/_base_/default_runtime.py",
]

num_classes = 6

msfa_ckpt = os.environ.get("MSFA_CKPT", "")
if not msfa_ckpt:
    raise RuntimeError(
        "MSFA_CKPT is required for this config. "
        "Set MSFA_CKPT to the MSFA-pretrained ConvNeXt-S checkpoint path."
    )

model = dict(
    backbone=dict(
        _delete_=True,
        type="MSFA",
        use_sar=True,
        freeze_backbone=True,
        backbone=dict(
            type="ConvNeXt",
            depths=[3, 3, 27, 3],
            dims=[96, 192, 384, 768],
            drop_path_rate=0.4,
            layer_scale_init_value=1e-6,
            out_indices=[0, 1, 2, 3],
            init_cfg=None,
        ),
        init_cfg=dict(type="Pretrained", prefix="backbone", checkpoint=msfa_ckpt),
    ),
    neck=dict(
        in_channels=[96, 192, 384, 768],
        start_level=1,
    ),
    bbox_head=dict(
        num_classes=num_classes,
    ),
)

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        betas=(0.9, 0.999),
        lr=0.0001,
        type="AdamW",
        weight_decay=0.05,
    ),
    type="OptimWrapper",
)

