# Weights

This repo intentionally does not vendor large checkpoints.

## MSFA ConvNeXt-S pretrained checkpoint (E0009)

E0009 uses `mmdet_toolkit/local_configs/SARDet/msfa_baselines/retinanet_msfa-convnext-small_linear-probe_1x_sardet_bs64_amp.py`,
which requires `MSFA_CKPT` to point to an MSFA-pretrained ConvNeXt-S checkpoint file.

- Recommended local path: `artifacts/weights/msfa_convnext_small_msfa_pretrained.pth`
- Source (authors’ released weights bundle): `https://pan.baidu.com/s/1SuEOl_ImqjoT5Y3pYxZt4w?pwd=c6fo`

Example:

```bash
export MSFA_CKPT="$(pwd)/artifacts/weights/msfa_convnext_small_msfa_pretrained.pth"
```
