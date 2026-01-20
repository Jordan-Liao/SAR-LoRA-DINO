_base_ = [
    "../../_base_/models/retinanet_r50_fpn.py",
    "../../_base_/datasets/sardet100k.py",
    "../../_base_/schedules/1x.py",
    "../../_base_/default_runtime.py",
]

num_classes = 6

custom_imports = dict(
    imports=[
        "sar_lora_dino.models.backbones.timm_convnext",
    ],
    allow_failed_imports=False,
)

model = dict(
    backbone=dict(
        _delete_=True,
        type="TimmConvNeXt",
        model_name="convnext_small.dinov3_lvd1689m",
        pretrained=True,
        out_indices=(0, 1, 2, 3),
        freeze_backbone=True,
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
