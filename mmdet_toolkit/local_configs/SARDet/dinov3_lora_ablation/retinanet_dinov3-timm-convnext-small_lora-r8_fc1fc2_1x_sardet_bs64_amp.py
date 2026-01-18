_base_ = [
    "../dinov3_lora/retinanet_dinov3-timm-convnext-small_lora-r16_1x_sardet_bs64_amp.py",
]

model = dict(
    backbone=dict(
        lora_r=8,
        lora_alpha=16.0,
    ),
)

