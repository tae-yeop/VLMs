import argparse
from typing import Dict, Iterable, List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
from transformers import pipeline


# ───────────────────────────────────────────────────────────
# 기본 라벨 셋. '캐리어' 관련 동의어 포함
DEFAULT_LABELS: List[str] = [
    "person",
    "suitcase",
    "luggage",
    "trolley",
    "rolling suitcase",
    "carry-on",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="입력 동영상 경로")
    parser.add_argument("--output", type=str, default="output_bbox.mp4", help="출력 동영상 경로 (mp4)")
    parser.add_argument(
        "--model",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO 모델 (e.g., tiny/base)",
    )
    parser.add_argument("--device", type=int, default=0, help="GPU index (CPU는 -1)")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="박스(검출) 임계값")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="텍스트 매칭 임계값")
    parser.add_argument("--stride", type=int, default=1, help="프레임 스킵 간격(속도 향상용). 1이면 모든 프레임 처리")
    parser.add_argument("--fps", type=float, default=None, help="출력 FPS(기본은 입력과 동일)")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="탐지 라벨(콤마로 구분). 기본: person,suitcase,luggage,trolley,rolling suitcase,carry-on",
    )
    return parser.parse_args()


def build_color_map(labels: Iterable[str]) -> Dict[str, Tuple[int, int, int]]:
    base_color = (0, 255, 0)
    suitcase_color = (0, 140, 255)
    color_map: Dict[str, Tuple[int, int, int]] = {}
    for label in labels:
        if normalize_label(label) == "suitcase":
            color_map[label] = suitcase_color
        elif label == "person":
            color_map[label] = (0, 200, 0)
        else:
            color_map[label] = base_color
    return color_map


def normalize_label(label: str) -> str:
    label_l = label.lower()
    if label_l in {"luggage", "trolley", "rolling suitcase", "carry-on"}:
        return "suitcase"
    return label_l


def draw_box(img: np.ndarray, box: Dict[str, float], label: str, score: float, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = int(box.get("xmin", 0)), int(box.get("ymin", 0)), int(box.get("xmax", 0)), int(box.get("ymax", 0))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {score:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - baseline - 4)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, text, (x1 + 3, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def setup_video_io(input_path: str, output_path: str, fps_override: Optional[float]) -> Tuple[cv2.VideoCapture, cv2.VideoWriter, int, int, float, int]:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps if args.fps is None else args.fps, (width, height))

    # 클래스별 색상
    color_map = {
        "person": (0, 200, 0),
        "suitcase": (0, 140, 255),
        "luggage": (0, 140, 255),
        "trolley": (0, 140, 255),
        "rolling suitcase": (0, 140, 255),
        "carry-on": (0, 140, 255),
    }

    # 프레임 처리
    pbar = tqdm(total=total, desc="Processing", ncols=80)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 프레임 스킵(속도용 옵션)
        if args.stride > 1 and (frame_idx % args.stride != 1):
            out.write(frame)
            pbar.update(1)
            continue

        # Grounding DINO는 RGB 입력이 안정적
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 추론
        results = det(
            rgb,
            candidate_labels=TARGET_LABELS,
            threshold=args.box_threshold,        # 박스 스코어 임계
            text_threshold=args.text_threshold,  # 텍스트(프롬프트) 매칭 임계
        )
        # 결과는 [{"score": float, "label": str, "box": {"xmin":..,"ymin":..,"xmax":..,"ymax":..}}, ...]
        for r in results:
            label = r.get("label", "").lower()
            score = float(r.get("score", 0.0))
            box   = r.get("box", {})
            # 라벨 정규화: 캐리어류 통일 표기(옵션)
            norm_label = "suitcase" if label in ["luggage", "trolley", "rolling suitcase", "carry-on"] else label

            if norm_label in ["person", "suitcase"]:
                color = color_map.get(label, (0, 255, 0))
                draw_box(frame, box, norm_label, score, color=color)

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="입력 동영상 경로 (e.g., /mnt/data/NIPA_15_06_20190917105959_Lost_00002_blur.mp4)")
    parser.add_argument("--output", type=str, default="output_bbox.mp4", help="출력 동영상 경로 (mp4)")
    parser.add_argument("--model", type=str, default="IDEA-Research/grounding-dino-tiny",
                        help="Grounding DINO 모델 (tiny/base 등)")
    parser.add_argument("--device", type=int, default=0, help="GPU index (CPU는 -1)")
    parser.add_argument("--box_threshold", type=float, default=0.35, help="박스(검출) 임계값")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="텍스트 매칭 임계값")
    parser.add_argument("--stride", type=int, default=1, help="프레임 스킵 간격(속도 향상용). 1이면 모든 프레임 처리")
    parser.add_argument("--fps", type=float, default=None, help="출력 FPS(기본은 입력과 동일)")
    args = parser.parse_args()
    main(args)
