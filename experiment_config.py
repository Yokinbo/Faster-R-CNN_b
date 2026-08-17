"""Faster R-CNN 论文对比实验的共享模型配置。

本仓库没有官方 S/M 型号族，只提供 ResNet50 与 VGG16 两种骨干。
训练、验证和测试统一从这里读取结构参数，避免权重与模型配置漂移。
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# MODEL_VARIANT：选择 Faster R-CNN 骨干。当前论文基线使用 resnet50；
# 若改为 vgg16，必须同时准备对应预训练权重并重新训练，不能混用权重。
MODEL_VARIANT = "resnet50"
# MODEL_IMAGE_SIZE：train/valid/test 共用的正方形网络输入尺寸；
# 设为512是为了与其他论文对比模型统一输入尺度，数值必须能被32整除。
MODEL_IMAGE_SIZE = 512
# ANCHOR_SCALES：RPN 在步长16特征图上的基础锚框尺度；[8,16,32]
# 约对应128、256、512像素边长，并会结合宽高比生成不同形状的候选框。
ANCHOR_SCALES = [8, 16, 32]

MODEL_BACKBONES = {
    "resnet50": "resnet50",
    "vgg16": "vgg",
}
PRETRAINED_WEIGHT_PATHS = {
    "resnet50": REPO_ROOT / "weights" / "voc_weights_resnet.pth",
    "vgg16": REPO_ROOT / "weights" / "voc_weights_vgg.pth",
}


def normalized_model_variant() -> str:
    variant = MODEL_VARIANT.lower().strip()
    if variant not in MODEL_BACKBONES:
        raise ValueError(
            f"MODEL_VARIANT 必须是 {tuple(MODEL_BACKBONES)} 之一，当前为 {MODEL_VARIANT!r}"
        )
    return variant


MODEL_BACKBONE = MODEL_BACKBONES[normalized_model_variant()]
MODEL_TAG = f"Faster-RCNN-{normalized_model_variant()}"
PRETRAINED_WEIGHT_PATH = PRETRAINED_WEIGHT_PATHS[normalized_model_variant()]
