"""Validate Faster R-CNN on the YOLO-format validation split.

The report follows the same comparison protocol as the YOLO26 validator:
fixed-confidence P/R/F1, COCO mAP50/mAP75/mAP50:95, and the confidence at the
maximum smoothed F1 operating point.
"""

from __future__ import annotations

import json
import os
import time
import copy
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    from faster_coco_eval import COCO, COCOeval_faster as COCOeval

from experiment_config import ANCHOR_SCALES, MODEL_BACKBONE, MODEL_IMAGE_SIZE, MODEL_TAG
from nets.frcnn import FasterRCNN
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox
from utils.yolo_dataset import (build_coco_annotation_lines,
                                validate_yolo_class_files)


# =============================================================================
# 用户验证参数配置区（全部使用绝对路径）
# =============================================================================
# WEIGHTS_PATH：待验证权重；通常填写训练阶段按验证集 AP50 选出的 best_map50.pth。
WEIGHTS_PATH = Path(r"E:\YOLO\faster-rcnn\output\faster_rcnn_resnet50_hdc_240epochs_512\best_map50.pth")
# DATASET_ROOT：数据集根目录；验证脚本只读取其中 val，不会读取 test。
DATASET_ROOT = Path(r"E:\YOLO\faster-rcnn\datasets\mydatasets")
# OUTPUT_DIR：本次验证的完整输出目录，直接填写 Windows 绝对路径。
OUTPUT_DIR = Path(r"E:\YOLO\faster-rcnn\output\validation\faster_rcnn_resnet50_hdc_240epochs_512")
# EXIST_OK=False：目录已存在时自动创建末尾带2、3……的新目录，避免覆盖旧结果；
# 设为True时允许复用并覆盖同名结果文件。
EXIST_OK = False

# PAPER_MODEL_NAME：只控制“论文指标.txt”中的 Model 显示名称；
# 不选择模型、不加载权重，也不会影响实际精度。
PAPER_MODEL_NAME = "faster-rcnn-resnet50-512"


INPUT_SIZE = [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]
DEVICE = "cuda:0"  # CPU 使用 "cpu"
BACKBONE = MODEL_BACKBONE

# AP 使用低阈值保留完整 PR 曲线；固定阈值指标用于论文主表。
AP_CONFIDENCE = 0.001
FIXED_CONFIDENCE = 0.50
FIXED_MATCH_IOU = 0.50
NMS_IOU = 0.70
MAX_DETECTIONS = 300
PLOTS = True
ENABLE_FLOPS = True
ENABLE_FPS_BENCHMARK = True
FPS_WARMUP_ITERS = 10
FPS_TEST_ITERS = 100


# =============================================================================


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def resolve_output_dir(path: Path) -> Path:
    if EXIST_OK or not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}{index}")
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def validate_config() -> None:
    required = {"模型权重": WEIGHTS_PATH, "数据集目录": DATASET_ROOT}
    missing = [f"{name}：{path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("以下验证输入不存在：\n" + "\n".join(missing))
    if len(INPUT_SIZE) != 2 or any(size <= 0 or size % 32 for size in INPUT_SIZE):
        raise ValueError("INPUT_SIZE 必须包含两个能被32整除的正整数。")
    if MAX_DETECTIONS <= 0:
        raise ValueError("MAX_DETECTIONS 必须大于0。")
    thresholds = (AP_CONFIDENCE, FIXED_CONFIDENCE, FIXED_MATCH_IOU, NMS_IOU)
    if not all(0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("置信度和 IoU 阈值必须位于 [0, 1]。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 CUDA，但当前 PyTorch 未检测到可用 GPU。")


def load_model(num_classes: int, device: torch.device) -> torch.nn.Module:
    model = FasterRCNN(
        num_classes,
        mode="predict",
        anchor_scales=ANCHOR_SCALES,
        backbone=BACKBONE,
    )
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    checkpoint = {
        key.removeprefix("module."): value for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    return model.to(device).eval()


def parse_ground_truth(annotation_line: str) -> Tuple[Path, List[Dict[str, object]]]:
    fields = annotation_line.split()
    image_path = Path(fields[0])
    boxes = []
    for field in fields[1:]:
        left, top, right, bottom, class_id = map(float, field.split(","))
        boxes.append(
            {
                "box": [float(left), float(top), float(right), float(bottom)],
                "class_id": int(class_id),
            }
        )
    return image_path, boxes


def infer_image(
    model: torch.nn.Module,
    decoder: DecodeBox,
    image: Image.Image,
    device: torch.device,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    started = time.perf_counter()
    image = cvtColor(image)
    image_width, image_height = image.size
    resized = resize_image(image, [INPUT_SIZE[1], INPUT_SIZE[0]])
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(resized, dtype=np.float32)), (2, 0, 1)),
        axis=0,
    )
    tensor = torch.from_numpy(image_data).to(device)
    preprocess_ms = (time.perf_counter() - started) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_started = time.perf_counter()
    with torch.no_grad():
        roi_cls_locs, roi_scores, rois, _ = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    post_started = time.perf_counter()
    results = decoder.forward(
        roi_cls_locs,
        roi_scores,
        rois,
        np.array([image_height, image_width]),
        INPUT_SIZE,
        nms_iou=NMS_IOU,
        confidence=AP_CONFIDENCE,
    )[0]

    predictions: List[Dict[str, object]] = []
    if len(results):
        results = np.asarray(results)
        results = results[np.argsort(results[:, 4])[::-1]][:MAX_DETECTIONS]
        for top, left, bottom, right, score, class_id in results:
            left = float(np.clip(left, 0, image_width))
            right = float(np.clip(right, 0, image_width))
            top = float(np.clip(top, 0, image_height))
            bottom = float(np.clip(bottom, 0, image_height))
            if right <= left or bottom <= top:
                continue
            predictions.append(
                {
                    "box": [left, top, right, bottom],
                    "score": float(score),
                    "class_id": int(class_id),
                }
            )
    postprocess_ms = (time.perf_counter() - post_started) * 1000.0
    return predictions, {
        "preprocess": preprocess_ms,
        "inference": inference_ms,
        "postprocess": postprocess_ms,
    }


def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_counts(
    ground_truth: List[Dict[str, object]],
    predictions: List[Dict[str, object]],
    confidence: float,
) -> Tuple[int, int, int]:
    """与 D-FINE Validator 一致：按最高 IoU 做一对一匹配。"""
    selected = [item for item in predictions if item["score"] >= confidence]
    candidates = []
    for pred_index, prediction in enumerate(selected):
        for gt_index, target in enumerate(ground_truth):
            if target["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(prediction["box"], target["box"])
            if overlap >= FIXED_MATCH_IOU:
                candidates.append((overlap, pred_index, gt_index))
    matched_predictions = set()
    matched_targets = set()
    for _, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or gt_index in matched_targets:
            continue
        matched_predictions.add(pred_index)
        matched_targets.add(gt_index)
    tp = len(matched_predictions)
    return tp, len(selected) - tp, len(ground_truth) - tp


def threshold_curves(
    evaluation_records: List[Tuple[List[Dict[str, object]], List[Dict[str, object]]]],
) -> Dict[str, np.ndarray]:
    # 与 D-FINE standalone valid.py 一致：0.00~1.00，步长0.01。
    thresholds = np.linspace(0.0, 1.0, 101)
    precision = np.zeros_like(thresholds)
    recall = np.zeros_like(thresholds)
    f1 = np.zeros_like(thresholds)
    tp_values = np.zeros_like(thresholds, dtype=np.int64)
    fp_values = np.zeros_like(thresholds, dtype=np.int64)
    fn_values = np.zeros_like(thresholds, dtype=np.int64)

    for index, threshold in enumerate(thresholds):
        tp = fp = fn = 0
        for ground_truth, predictions in evaluation_records:
            image_tp, image_fp, image_fn = match_counts(
                ground_truth, predictions, float(threshold)
            )
            tp += image_tp
            fp += image_fp
            fn += image_fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision[index] = p
        recall[index] = r
        f1[index] = 2.0 * p * r / (p + r) if p + r else 0.0
        tp_values[index] = tp
        fp_values[index] = fp
        fn_values[index] = fn
    return {
        "thresholds": thresholds,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp_values,
        "fp": fp_values,
        "fn": fn_values,
    }


def smooth_curve(values: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    window = max(1, round(len(values) * fraction * 2) // 2 + 1)
    padding = np.ones(window // 2)
    padded = np.concatenate((padding * values[0], values, padding * values[-1]))
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def metrics_at_index(curves: Dict[str, np.ndarray], index: int) -> Dict[str, object]:
    return {
        "Confidence": float(curves["thresholds"][index]),
        "matching_iou_threshold": FIXED_MATCH_IOU,
        "Precision": float(curves["precision"][index]),
        "Recall": float(curves["recall"][index]),
        "F1-score": float(curves["f1"][index]),
        "TP": int(curves["tp"][index]),
        "FP": int(curves["fp"][index]),
        "FN": int(curves["fn"][index]),
    }


def coco_evaluate(
    images: List[Dict[str, object]], annotations: List[Dict[str, object]],
    categories: List[Dict[str, object]], detections: List[Dict[str, object]],
) -> Tuple[np.ndarray, object | None]:
    if not detections:
        return np.zeros(12, dtype=np.float64), None
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats.copy(), evaluator


def save_curve_plot(curves: Dict[str, np.ndarray], output_dir: Path, best_index: int) -> None:
    plt.figure(figsize=(9, 6))
    thresholds = curves["thresholds"]
    plt.plot(thresholds, curves["precision"], label="Precision", linewidth=1.7)
    plt.plot(thresholds, curves["recall"], label="Recall", linewidth=1.7)
    plt.plot(thresholds, curves["f1"], label="F1", linewidth=2.0)
    plt.axvline(FIXED_CONFIDENCE, color="gray", linestyle="--", label="Fixed conf")
    plt.axvline(thresholds[best_index], color="red", linestyle=":", label="Best F1 conf")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_precision_recall_f1.png", dpi=220)
    plt.close()

    with (output_dir / "confidence_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["confidence", "precision", "recall", "f1", "TP", "FP", "FN"])
        for index, threshold in enumerate(thresholds):
            writer.writerow(
                [
                    f"{threshold:.2f}",
                    f"{curves['precision'][index]:.8f}",
                    f"{curves['recall'][index]:.8f}",
                    f"{curves['f1'][index]:.8f}",
                    int(curves["tp"][index]),
                    int(curves["fp"][index]),
                    int(curves["fn"][index]),
                ]
            )


def calculate_model_flops(model: torch.nn.Module) -> float | None:
    """沿用本仓库 summary.py 的 THOP×2 口径。"""
    if not ENABLE_FLOPS or thop_profile is None:
        return None
    try:
        profile_model = copy.deepcopy(model).cpu().eval()
        dummy = torch.zeros(1, 3, INPUT_SIZE[0], INPUT_SIZE[1])
        operations, _ = thop_profile(profile_model, inputs=(dummy,), verbose=False)
        del profile_model
        return float(operations * 2.0) / 1e9
    except Exception as error:
        print(f"FLOPs 统计跳过：{error}")
        return None


def benchmark_single_image(
    model: torch.nn.Module,
    decoder: DecodeBox,
    image_path: Path,
    device: torch.device,
) -> Dict[str, float]:
    """batch=1，预加载输入，统计模型前向和检测后处理。"""
    if not ENABLE_FPS_BENCHMARK:
        return {}
    with Image.open(image_path) as source:
        image = cvtColor(source.copy())
    width, height = image.size
    resized = resize_image(image, [INPUT_SIZE[1], INPUT_SIZE[0]])
    array = np.expand_dims(
        np.transpose(preprocess_input(np.asarray(resized, dtype=np.float32)), (2, 0, 1)),
        axis=0,
    )
    tensor = torch.from_numpy(array).to(device)

    def run_once() -> None:
        outputs = model(tensor)
        decoder.forward(
            outputs[0], outputs[1], outputs[2],
            np.array([height, width]), INPUT_SIZE,
            nms_iou=NMS_IOU, confidence=FIXED_CONFIDENCE,
        )

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(FPS_WARMUP_ITERS):
            run_once()
        synchronize()
        started = time.perf_counter()
        for _ in range(FPS_TEST_ITERS):
            run_once()
        synchronize()
    seconds_per_image = (time.perf_counter() - started) / FPS_TEST_ITERS
    return {
        "latency_ms_single_image_forward_post": seconds_per_image * 1000.0,
        "FPS_single_image_forward_post": 1.0 / seconds_per_image,
        "warmup_iterations": FPS_WARMUP_ITERS,
        "test_iterations": FPS_TEST_ITERS,
        "efficiency_confidence": FIXED_CONFIDENCE,
    }


def format_optional(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def save_paper_table(
    output_dir: Path,
    coco_metrics: Dict[str, float],
    fixed_metrics: Dict[str, object],
    model_parameters: int,
    flops_g: float | None,
    benchmark: Dict[str, float],
) -> Path:
    latency = benchmark.get("latency_ms_single_image_forward_post")
    fps = benchmark.get("FPS_single_image_forward_post")
    accuracy_header = (
        f"{'Model':<30}{'Input':>8}{'Params(M)':>12}{'AP50':>10}{'AP75':>10}"
        f"{'mAP50:95':>12}{'P@0.5':>10}{'R@0.5':>10}{'F1@0.5':>10}"
    )
    accuracy_row = (
        f"{PAPER_MODEL_NAME:<30}{INPUT_SIZE[0]:>8}{model_parameters / 1e6:>12.3f}"
        f"{coco_metrics['mAP@0.5']:>10.4f}{coco_metrics['mAP@0.75']:>10.4f}"
        f"{coco_metrics['mAP@0.5:0.95']:>12.4f}{fixed_metrics['Precision']:>10.4f}"
        f"{fixed_metrics['Recall']:>10.4f}{fixed_metrics['F1-score']:>10.4f}"
    )
    efficiency_header = (
        f"{'Model':<30}{'Input':>8}{'Params(M)':>12}{'FLOPs(G)':>12}"
        f"{'Latency(ms/image)':>20}{'FPS':>12}"
    )
    efficiency_row = (
        f"{PAPER_MODEL_NAME:<30}{INPUT_SIZE[0]:>8}{model_parameters / 1e6:>12.3f}"
        f"{format_optional(flops_g):>12}{format_optional(latency):>20}{format_optional(fps):>12}"
    )
    lines = [
        "Faster R-CNN 论文对比实验指标",
        "=" * 112,
        "一、精度对比表",
        accuracy_header,
        "-" * len(accuracy_header),
        accuracy_row,
        "",
        "二、效率对比表",
        efficiency_header,
        "-" * len(efficiency_header),
        efficiency_row,
        "",
        "评价口径：",
        "1. P@0.5、R@0.5、F1@0.5：置信度≥0.50，匹配IoU≥0.50。",
        "2. AP50、AP75、mAP50:95：COCO标准检测评价。",
        f"3. FLOPs：本仓库THOP×2口径，batch=1，输入{INPUT_SIZE[0]}×{INPUT_SIZE[1]}。",
        "4. Latency与FPS：batch=1，模型前向+检测后处理，不含磁盘和预处理。",
        f"5. 测速预热{FPS_WARMUP_ITERS}次，正式测试{FPS_TEST_ITERS}次。",
    ]
    path = output_dir / "论文指标.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    validate_config()
    output_dir = resolve_output_dir(OUTPUT_DIR)
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    device = torch.device(DEVICE)

    classes_path = validate_yolo_class_files(DATASET_ROOT)
    class_names, num_classes = get_classes(str(classes_path))
    val_lines, dataset_summary = build_coco_annotation_lines(
        DATASET_ROOT, "val", class_names
    )
    model = load_model(num_classes, device)
    std = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device).repeat(num_classes + 1)[None]
    decoder = DecodeBox(std, num_classes)

    print("\n========== Faster R-CNN 验证配置 ==========")
    print(f"共享模型：{MODEL_TAG}")
    print(f"模型权重：{WEIGHTS_PATH}")
    print(f"验证数据：{DATASET_ROOT / 'val'}")
    print(f"输入尺寸：{INPUT_SIZE[0]} × {INPUT_SIZE[1]} | 设备：{device}")
    print(f"影像数量：{dataset_summary.images} | 目标数量：{dataset_summary.objects}")
    print(f"结果目录：{output_dir}\n")

    coco_images: List[Dict[str, object]] = []
    coco_annotations: List[Dict[str, object]] = []
    coco_detections: List[Dict[str, object]] = []
    evaluation_records: List[Tuple[List[Dict[str, object]], List[Dict[str, object]]]] = []
    timing = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    annotation_id = 1

    for image_id, annotation_line in enumerate(val_lines, start=1):
        image_path, ground_truth = parse_ground_truth(annotation_line)
        with Image.open(image_path) as image:
            width, height = image.size
            predictions, image_timing = infer_image(model, decoder, image, device)
        for name in timing:
            timing[name] += image_timing[name]

        coco_images.append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        for target in ground_truth:
            left, top, right, bottom = target["box"]
            box_width, box_height = right - left, bottom - top
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(target["class_id"]) + 1,
                    "bbox": [left, top, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
        for prediction in predictions:
            left, top, right, bottom = prediction["box"]
            coco_detections.append(
                {
                    "image_id": image_id,
                    "category_id": int(prediction["class_id"]) + 1,
                    "bbox": [left, top, right - left, bottom - top],
                    "score": float(prediction["score"]),
                }
            )
        evaluation_records.append((ground_truth, predictions))
        if image_id % 20 == 0 or image_id == len(val_lines):
            print(f"验证进度：{image_id}/{len(val_lines)}")

    categories = [
        {"id": index + 1, "name": name, "supercategory": name}
        for index, name in enumerate(class_names)
    ]
    coco_stats, coco_evaluator = coco_evaluate(
        coco_images, coco_annotations, categories, coco_detections
    )
    curves = threshold_curves(evaluation_records)
    fixed_index = int(round(FIXED_CONFIDENCE * 100))
    fixed_metrics = metrics_at_index(curves, fixed_index)
    smoothed_f1 = smooth_curve(curves["f1"], 0.1)
    best_index = int(np.argmax(smoothed_f1))
    optimal_metrics = metrics_at_index(curves, best_index)
    optimal_metrics["smoothed_F1-score"] = float(smoothed_f1[best_index])

    elapsed_seconds = time.perf_counter() - start_time
    image_count = max(1, dataset_summary.images)
    speed = {name: value / image_count for name, value in timing.items()}
    speed["total"] = sum(speed.values())
    coco_metrics = {
        "mAP@0.5": float(coco_stats[1]),
        "mAP@0.75": float(coco_stats[2]),
        "mAP@0.5:0.95": float(coco_stats[0]),
        "AR@100": float(coco_stats[8]),
    }
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    flops_g = calculate_model_flops(model)
    benchmark = benchmark_single_image(
        model, decoder, parse_ground_truth(val_lines[0])[0], device
    )

    report = {
        "weights": str(WEIGHTS_PATH),
        "model_tag": MODEL_TAG,
        "dataset": str(DATASET_ROOT),
        "input_size": INPUT_SIZE,
        "images": dataset_summary.images,
        "targets": dataset_summary.objects,
        "negative_images": dataset_summary.negative_images,
        "fixed_threshold_metrics": fixed_metrics,
        "coco_metrics": coco_metrics,
        "optimal_f1_point_metrics": optimal_metrics,
        "speed_ms_per_image": speed,
        "model_parameters": model_parameters,
        "FLOPs_G": flops_g,
        "benchmark": benchmark,
        "validation_elapsed_seconds": elapsed_seconds,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "coco_predictions.json").write_text(
        json.dumps(coco_detections, ensure_ascii=False), encoding="utf-8"
    )
    if coco_evaluator is not None:
        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    finished_at = datetime.now().astimezone()
    text_lines = [
        "Faster R-CNN ResNet50 HDC 验证报告",
        "=" * 54,
        f"模型权重：{WEIGHTS_PATH}",
        f"验证数据：{DATASET_ROOT / 'val'}",
        f"输入尺寸：{INPUT_SIZE[0]} × {INPUT_SIZE[1]}",
        f"影像/目标/负样本：{dataset_summary.images}/{dataset_summary.objects}/{dataset_summary.negative_images}",
        "",
        "一、论文主表固定阈值指标",
        *[f"{key:24s}: {value:.6f}" if isinstance(value, float) else f"{key:24s}: {value}" for key, value in fixed_metrics.items()],
        "",
        "二、COCO 标准 AP 指标",
        *[f"{key:24s}: {value:.6f}" for key, value in coco_metrics.items()],
        "",
        "三、最大平滑 F1 工作点",
        *[f"{key:24s}: {value:.6f}" if isinstance(value, float) else f"{key:24s}: {value}" for key, value in optimal_metrics.items()],
        "",
        "四、速度与规模",
        *[f"{key:24s}: {value:.4f} ms" for key, value in speed.items()],
        f"模型参数量              : {model_parameters / 1e6:.3f} M",
        f"FLOPs                   : {format_optional(flops_g)} G",
        f"Latency                 : {format_optional(benchmark.get('latency_ms_single_image_forward_post'))} ms/image",
        f"FPS                     : {format_optional(benchmark.get('FPS_single_image_forward_post'))}",
        "",
        "五、验证耗时",
        f"开始时间：{started_at.isoformat(timespec='seconds')}",
        f"结束时间：{finished_at.isoformat(timespec='seconds')}",
        f"总耗时：{format_duration(elapsed_seconds)}",
        f"结果目录：{output_dir}",
    ]
    (output_dir / "精度指标报告.txt").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )
    if PLOTS:
        save_curve_plot(curves, output_dir, best_index)
    paper_path = save_paper_table(
        output_dir, coco_metrics, fixed_metrics, model_parameters, flops_g, benchmark
    )

    print("\n========== 固定阈值精度（论文主表） ==========")
    print(f"置信度 / 匹配IoU：{FIXED_CONFIDENCE:.2f} / {FIXED_MATCH_IOU:.2f}")
    print(f"Precision          ：{fixed_metrics['Precision']:.6f}")
    print(f"Recall             ：{fixed_metrics['Recall']:.6f}")
    print(f"F1-score           ：{fixed_metrics['F1-score']:.6f}")
    print(f"TP / FP / FN       ：{fixed_metrics['TP']} / {fixed_metrics['FP']} / {fixed_metrics['FN']}")
    print("\n========== COCO 标准精度 ==========")
    print(f"mAP@0.5            ：{coco_metrics['mAP@0.5']:.6f}")
    print(f"mAP@0.75           ：{coco_metrics['mAP@0.75']:.6f}")
    print(f"mAP@0.5:0.95       ：{coco_metrics['mAP@0.5:0.95']:.6f}")
    print(f"最大F1推荐置信度   ：{optimal_metrics['Confidence']:.3f}")
    print(f"验证总耗时         ：{format_duration(elapsed_seconds)}")
    print(f"完整结果已保存     ：{output_dir}")
    print(f"论文指标文件       ：{paper_path}")


if __name__ == "__main__":
    main()
