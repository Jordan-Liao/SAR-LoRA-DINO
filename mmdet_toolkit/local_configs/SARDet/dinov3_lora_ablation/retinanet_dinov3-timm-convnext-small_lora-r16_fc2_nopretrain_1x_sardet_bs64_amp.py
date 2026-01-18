_base_ = [
    "./retinanet_dinov3-timm-convnext-small_lora-r16_fc2_1x_sardet_bs64_amp.py",
]

model = dict(
    backbone=dict(
        pretrained=False,
    ),
)

