"""Faster R-CNN 独立测试集评估与论文最终核心指标输出。

模型、权重和阈值必须先在训练集/验证集上确定。本脚本只在独立 test 集上
进行一次最终评估，不搜索最佳置信度，也不使用测试结果选择权重。
"""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from PIL import Image

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    from faster_coco_eval import COCO, COCOeval_faster as COCOeval

try:
    from thop import profile as thop_profile
except ImportError:
    thop_profile = None

from experiment_config import ANCHOR_SCALES, MODEL_BACKBONE, MODEL_IMAGE_SIZE, MODEL_TAG
from nets.frcnn import FasterRCNN
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_bbox import DecodeBox


# =============================================================================
# 用户测试参数配置区：均可直接填写 Windows 绝对路径
# =============================================================================

# 只能填写已经根据验证集 AP50 选定的最终权重，禁止根据 test 结果更换。
WEIGHTS_PATH = Path(r"E:\YOLO\faster-rcnn\output\faster_rcnn_resnet50_hdc_240epochs_512\best_map50.pth")
# TEST_IMAGES_DIR：独立测试集影像目录；test 不参与训练、选权重或调参。
TEST_IMAGES_DIR = Path(r"E:\YOLO\faster-rcnn\datasets\mydatasets\test\images")
# TEST_ANNOTATION：独立测试集 COCO 真值标签，必须明确指向 test.json。
TEST_ANNOTATION = Path(r"E:\YOLO\faster-rcnn\datasets\mydatasets\test\annotations\test.json")
# OUTPUT_DIR：本次测试的完整输出目录，直接填写 Windows 绝对路径。
OUTPUT_DIR = Path(r"E:\YOLO\faster-rcnn\output\test\faster_rcnn_resnet50_hdc_240epochs_512")
# PAPER_MODEL_NAME：只控制“测试集最终核心指标.txt”中的 Model 显示名称；
# 不选择模型、不加载权重，也不会影响实际精度。
PAPER_MODEL_NAME = "faster-rcnn-resnet50-512"
# EXIST_OK=False：目录已存在时自动创建末尾带2、3……的新目录，避免覆盖旧结果；
# 设为True时允许复用并覆盖同名结果文件。
EXIST_OK = False

# 必须与训练、验证和权重结构完全一致。本仓库原生仅支持 "resnet50"、"vgg"。
BACKBONE = MODEL_BACKBONE
INPUT_SIZE = [MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE]
DEVICE = "cuda:0"  # CPU 使用 "cpu"

# AP 使用低阈值保留完整 PR 曲线；P/R/F1 固定为 confidence=0.50、IoU=0.50。
AP_CONFIDENCE = 0.001
FIXED_CONFIDENCE = 0.50
MATCH_IOU = 0.50
NMS_IOU = 0.70
MAX_DETECTIONS = 300

# 效率口径：batch=1，输入张量预先放入 GPU，统计模型前向+检测后处理。
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
    required = {
        "模型权重": WEIGHTS_PATH,
        "测试影像目录": TEST_IMAGES_DIR,
        "测试集 COCO JSON": TEST_ANNOTATION,
    }
    missing = [f"{name}：{path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("以下测试输入不存在：\n" + "\n".join(missing))
    if TEST_ANNOTATION.name.lower() != "test.json":
        raise ValueError(f"测试标注必须明确使用 test.json：{TEST_ANNOTATION}")
    if "test" not in {part.lower() for part in TEST_IMAGES_DIR.parts}:
        raise ValueError(f"测试影像路径看起来不是 test 分区：{TEST_IMAGES_DIR}")
    if BACKBONE not in {"resnet50", "vgg"}:
        raise ValueError('BACKBONE 只能是 "resnet50" 或 "vgg"。')
    if len(INPUT_SIZE) != 2 or any(size <= 0 or size % 32 for size in INPUT_SIZE):
        raise ValueError("INPUT_SIZE 必须包含两个能被 32 整除的正整数。")
    if MAX_DETECTIONS <= 0:
        raise ValueError("MAX_DETECTIONS 必须大于 0。")
    if AP_CONFIDENCE > FIXED_CONFIDENCE:
        raise ValueError("AP_CONFIDENCE 必须不高于 FIXED_CONFIDENCE。")
    if FIXED_CONFIDENCE != 0.50 or MATCH_IOU != 0.50:
        raise ValueError("论文统一测试口径要求 confidence=0.50、匹配 IoU=0.50。")
    if not all(0.0 <= value <= 1.0 for value in (AP_CONFIDENCE, NMS_IOU)):
        raise ValueError("置信度和 NMS IoU 必须位于 [0, 1]。")
    if FPS_WARMUP_ITERS < 0 or FPS_TEST_ITERS <= 0:
        raise ValueError("FPS_WARMUP_ITERS 必须≥0，FPS_TEST_ITERS 必须>0。")
    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 CUDA，但当前 PyTorch 未检测到可用 GPU。")


def load_test_dataset() -> Tuple[dict, List[dict], Dict[int, int]]:
    payload = json.loads(TEST_ANNOTATION.read_text(encoding="utf-8"))
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    categories = payload.get("categories", [])
    if not images or not categories:
        raise ValueError("test.json 必须包含非空 images 和 categories。")

    names = [item["file_name"] for item in images]
    image_ids = [int(item["id"]) for item in images]
    if len(names) != len(set(names)) or len(image_ids) != len(set(image_ids)):
        raise ValueError("test.json 存在重复 file_name 或 image id。")

    missing_images = [name for name in names if not (TEST_IMAGES_DIR / name).is_file()]
    if missing_images:
        raise FileNotFoundError(
            f"test.json 中有 {len(missing_images)} 张影像不存在，例如：{missing_images[:5]}"
        )

    category_ids = [int(item["id"]) for item in categories]
    if len(category_ids) != len(set(category_ids)):
        raise ValueError("test.json 存在重复 category id。")
    category_to_model = {
        category_id: model_index for model_index, category_id in enumerate(category_ids)
    }
    valid_image_ids = set(image_ids)
    for annotation in annotations:
        bbox = annotation.get("bbox", [])
        if int(annotation.get("image_id", -1)) not in valid_image_ids:
            raise ValueError(f"标注引用了不存在的 image_id：{annotation}")
        if int(annotation.get("category_id", -1)) not in category_to_model:
            raise ValueError(f"标注引用了不存在的 category_id：{annotation}")
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError(f"test.json 中存在非法 bbox：{annotation}")
    return payload, categories, category_to_model


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


def prepare_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    image = cvtColor(image)
    resized = resize_image(image, [INPUT_SIZE[1], INPUT_SIZE[0]])
    array = np.expand_dims(
        np.transpose(preprocess_input(np.asarray(resized, dtype=np.float32)), (2, 0, 1)),
        axis=0,
    )
    return torch.from_numpy(array).to(device)


def decode_predictions(
    decoder: DecodeBox,
    outputs: Tuple[torch.Tensor, ...],
    image_width: int,
    image_height: int,
    confidence: float,
) -> List[Dict[str, object]]:
    roi_cls_locs, roi_scores, rois, _ = outputs
    decoded = decoder.forward(
        roi_cls_locs,
        roi_scores,
        rois,
        np.array([image_height, image_width]),
        INPUT_SIZE,
        nms_iou=NMS_IOU,
        confidence=confidence,
    )[0]
    if not len(decoded):
        return []
    decoded = np.asarray(decoded)
    decoded = decoded[np.argsort(decoded[:, 4])[::-1]][:MAX_DETECTIONS]
    predictions = []
    for top, left, bottom, right, score, class_id in decoded:
        left = float(np.clip(left, 0, image_width))
        right = float(np.clip(right, 0, image_width))
        top = float(np.clip(top, 0, image_height))
        bottom = float(np.clip(bottom, 0, image_height))
        if right > left and bottom > top:
            predictions.append(
                {
                    "box": [left, top, right, bottom],
                    "score": float(score),
                    "class_id": int(class_id),
                }
            )
    return predictions


def infer_image(
    model: torch.nn.Module,
    decoder: DecodeBox,
    image: Image.Image,
    device: torch.device,
) -> List[Dict[str, object]]:
    tensor = prepare_tensor(image, device)
    with torch.inference_mode():
        outputs = model(tensor)
    return decode_predictions(decoder, outputs, image.width, image.height, AP_CONFIDENCE)


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


def fixed_threshold_counts(
    ground_truth: List[Dict[str, object]],
    predictions: List[Dict[str, object]],
) -> Tuple[int, int, int]:
    selected = [item for item in predictions if item["score"] >= FIXED_CONFIDENCE]
    candidates = []
    for pred_index, prediction in enumerate(selected):
        for gt_index, target in enumerate(ground_truth):
            if target["class_id"] != prediction["class_id"]:
                continue
            overlap = box_iou(prediction["box"], target["box"])
            if overlap >= MATCH_IOU:
                candidates.append((overlap, pred_index, gt_index))
    matched_predictions = set()
    matched_targets = set()
    for _, pred_index, gt_index in sorted(candidates, reverse=True):
        if pred_index in matched_predictions or gt_index in matched_targets:
            continue
        matched_predictions.add(pred_index)
        matched_targets.add(gt_index)
    true_positives = len(matched_predictions)
    false_positives = len(selected) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives


def coco_evaluate(payload: dict, detections: List[dict]) -> Tuple[np.ndarray, object | None]:
    if not detections:
        return np.zeros(12, dtype=np.float64), None
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": payload.get("info", {}),
        "licenses": payload.get("licenses", []),
        "images": payload["images"],
        "annotations": payload["annotations"],
        "categories": payload["categories"],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator.stats.copy(), evaluator


def calculate_model_flops(model: torch.nn.Module) -> float | None:
    """沿用本 Faster R-CNN 仓库 summary.py 的 THOP×2 统计口径。"""
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
    if not ENABLE_FPS_BENCHMARK:
        return {}
    with Image.open(image_path) as source:
        image = cvtColor(source.copy())
    tensor = prepare_tensor(image, device)

    def run_once() -> None:
        outputs = model(tensor)
        decode_predictions(
            decoder, outputs, image.width, image.height, FIXED_CONFIDENCE
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
        "latency_ms_per_image": seconds_per_image * 1000.0,
        "fps": 1.0 / seconds_per_image,
    }


def format_optional(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> None:
    validate_config()
    payload, categories, category_to_model = load_test_dataset()
    output_dir = resolve_output_dir(OUTPUT_DIR)
    device = torch.device(DEVICE)
    model = load_model(len(categories), device)
    std = torch.tensor([0.1, 0.1, 0.2, 0.2], device=device).repeat(len(categories) + 1)[None]
    decoder = DecodeBox(std, len(categories))

    annotations_by_image: Dict[int, List[dict]] = {
        int(image["id"]): [] for image in payload["images"]
    }
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    model_to_category = {value: key for key, value in category_to_model.items()}

    print("\n========== Faster R-CNN 独立测试配置 ==========")
    print(f"共享模型：{MODEL_TAG}")
    print(f"模型权重：{WEIGHTS_PATH}")
    print(f"测试影像：{TEST_IMAGES_DIR}")
    print(f"测试标注：{TEST_ANNOTATION}")
    print(f"模型结构：{BACKBONE} | 输入：{INPUT_SIZE[0]}×{INPUT_SIZE[1]}")
    print(f"图片/标注框：{len(payload['images'])}/{len(payload['annotations'])}")
    print(f"结果目录：{output_dir}\n")

    started_at = datetime.now().astimezone()
    started = time.perf_counter()
    detections: List[dict] = []
    total_tp = total_fp = total_fn = 0

    for sequence, image_info in enumerate(payload["images"], 1):
        image_id = int(image_info["id"])
        image_path = TEST_IMAGES_DIR / image_info["file_name"]
        ground_truth = []
        for annotation in annotations_by_image[image_id]:
            left, top, width, height = map(float, annotation["bbox"])
            ground_truth.append(
                {
                    "box": [left, top, left + width, top + height],
                    "class_id": category_to_model[int(annotation["category_id"])],
                }
            )
        with Image.open(image_path) as source:
            predictions = infer_image(model, decoder, source, device)

        tp, fp, fn = fixed_threshold_counts(ground_truth, predictions)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        for prediction in predictions:
            left, top, right, bottom = prediction["box"]
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": model_to_category[int(prediction["class_id"])],
                    "bbox": [left, top, right - left, bottom - top],
                    "score": float(prediction["score"]),
                }
            )
        if sequence % 20 == 0 or sequence == len(payload["images"]):
            print(f"测试进度：{sequence}/{len(payload['images'])}")

    coco_stats, coco_evaluator = coco_evaluate(payload, detections)
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    flops_g = calculate_model_flops(model)
    speed = benchmark_single_image(
        model,
        decoder,
        TEST_IMAGES_DIR / payload["images"][0]["file_name"],
        device,
    )
    latency_ms = speed.get("latency_ms_per_image")
    fps = speed.get("fps")
    elapsed = time.perf_counter() - started
    finished_at = datetime.now().astimezone()

    metrics = {
        "AP50": float(coco_stats[1]),
        "AP75": float(coco_stats[2]),
        "mAP50:95": float(coco_stats[0]),
        "P@0.5": precision,
        "R@0.5": recall,
        "F1@0.5": f1,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Params(M)": params_m,
        "FLOPs(G)": flops_g,
        "Latency(ms/image)": latency_ms,
        "FPS": fps,
    }
    report = {
        "model_name": PAPER_MODEL_NAME,
        "model_tag": MODEL_TAG,
        "backbone": BACKBONE,
        "weights": str(WEIGHTS_PATH),
        "test_images": str(TEST_IMAGES_DIR),
        "test_annotation": str(TEST_ANNOTATION),
        "input_size": INPUT_SIZE,
        "images": len(payload["images"]),
        "targets": len(payload["annotations"]),
        "negative_images": sum(not annotations_by_image[int(item["id"])] for item in payload["images"]),
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "coco_predictions.json").write_text(
        json.dumps(detections, ensure_ascii=False), encoding="utf-8"
    )
    if coco_evaluator is not None:
        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "test_eval.pth")

    detailed_lines = [
        "Faster R-CNN 独立测试集精度报告",
        "=" * 58,
        f"模型名称：{PAPER_MODEL_NAME}",
        f"模型结构：{BACKBONE}",
        f"模型权重：{WEIGHTS_PATH}",
        f"测试影像：{TEST_IMAGES_DIR}",
        f"测试标注：{TEST_ANNOTATION}",
        f"输入尺寸：{INPUT_SIZE[0]} × {INPUT_SIZE[1]}",
        f"影像/目标/负样本：{report['images']}/{report['targets']}/{report['negative_images']}",
        "",
        "一、固定阈值指标",
        f"Precision(P)            : {precision:.6f}",
        f"Recall(R)               : {recall:.6f}",
        f"F1-score                : {f1:.6f}",
        f"TP / FP / FN            : {total_tp} / {total_fp} / {total_fn}",
        f"置信度 / 匹配IoU        : {FIXED_CONFIDENCE:.2f} / {MATCH_IOU:.2f}",
        "",
        "二、COCO 标准指标",
        f"AP@0.5                  : {coco_stats[1]:.6f}",
        f"AP@0.75                 : {coco_stats[2]:.6f}",
        f"mAP@0.5:0.95            : {coco_stats[0]:.6f}",
        "",
        "三、模型规模与效率",
        f"参数量                   : {params_m:.3f} M",
        f"FLOPs                    : {format_optional(flops_g)} G",
        f"Latency                  : {format_optional(latency_ms)} ms/image",
        f"FPS                      : {format_optional(fps)}",
        "",
        "四、测试记录",
        f"开始时间：{started_at.isoformat(timespec='seconds')}",
        f"结束时间：{finished_at.isoformat(timespec='seconds')}",
        f"总耗时：{format_duration(elapsed)}",
        f"结果目录：{output_dir}",
    ]
    (output_dir / "测试集精度报告.txt").write_text(
        "\n".join(detailed_lines) + "\n", encoding="utf-8"
    )

    core_lines = [
        "Faster R-CNN 独立测试集最终核心对比指标",
        "=" * 94,
        "",
        "一、测试集精度对比表",
        "Model                         Input  Params(M)   AP50    AP75  mAP50:95   P@0.5   R@0.5  F1@0.5",
        "-" * 94,
        f"{PAPER_MODEL_NAME:<29}{INPUT_SIZE[0]:>6}{params_m:>11.3f}{coco_stats[1]:>8.4f}"
        f"{coco_stats[2]:>8.4f}{coco_stats[0]:>10.4f}{precision:>8.4f}{recall:>8.4f}{f1:>8.4f}",
        "",
        "二、效率对比表",
        "Model                         Input  Params(M)  FLOPs(G)  Latency(ms/image)      FPS",
        "-" * 84,
        f"{PAPER_MODEL_NAME:<29}{INPUT_SIZE[0]:>6}{params_m:>11.3f}"
        f"{format_optional(flops_g):>10}{format_optional(latency_ms):>19}{format_optional(fps):>9}",
        "",
        "评价口径：",
        "1. 本表全部精度来自独立 test 集；test 集未参与训练、选权重或调参。",
        "2. P@0.5、R@0.5、F1@0.5：置信度≥0.50，匹配IoU≥0.50。",
        "3. AP50、AP75、mAP50:95：COCO 标准检测评价。",
        "4. FLOPs：沿用本仓库 summary.py 的 THOP×2 口径，batch=1。",
        "5. Latency与FPS：batch=1，模型前向+检测后处理，不含磁盘读取和预处理。",
        f"6. 测速预热{FPS_WARMUP_ITERS}次，正式测试{FPS_TEST_ITERS}次。",
        "",
        "制表提示：将不同模型生成的数值行汇总，即可形成论文最终核心对比表。",
    ]
    (output_dir / "测试集最终核心指标.txt").write_text(
        "\n".join(core_lines) + "\n", encoding="utf-8"
    )

    print("\n========== 独立测试集最终指标 ==========")
    print(f"AP50 / AP75 / mAP50:95：{coco_stats[1]:.4f} / {coco_stats[2]:.4f} / {coco_stats[0]:.4f}")
    print(f"P / R / F1 @0.5        ：{precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"Params / FLOPs         ：{params_m:.3f} M / {format_optional(flops_g)} G")
    print(f"Latency / FPS          ：{format_optional(latency_ms)} ms / {format_optional(fps)}")
    print(f"核心指标文件           ：{output_dir / '测试集最终核心指标.txt'}")


if __name__ == "__main__":
    main()
