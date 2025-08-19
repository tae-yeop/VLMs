# src/smart_surveillance/utils/roi.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import cv2

Point = Tuple[int, int]

def normalize_polygon(poly: List[Tuple[float, float]], w: int, h: int) -> List[Point]:
    if not poly:
        return []
    if all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in poly):
        return [(int(round(x * w)), int(round(y * h))) for x, y in poly]
    return [(int(x), int(y)) for x, y in poly]

def point_in_polygon(pt: Point, poly: List[Point]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < xin:
                inside = not inside
    return inside

def bbox_center(b: List[float]) -> Point:
    x1, y1, x2, y2 = b
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, ay2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def mask_from_polygon(poly: List[Point], w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
    return mask

def inside_by_mask(center: Point, mask: np.ndarray) -> bool:
    x, y = center
    h, w = mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return False
    return mask[y, x] > 0

def polygon_from_mask(mask_rgba_or_gray: np.ndarray, epsilon_ratio: float = 0.01) -> List[Point]:
    """
    Gradio ImageEditor의 mask(np.uint8/0~255)나 RGBA 이미지에서 ROI 외곽선을 찾아 다각형 좌표로 근사.
    """
    if mask_rgba_or_gray.ndim == 3 and mask_rgba_or_gray.shape[2] == 4:
        # alpha 채널 사용
        mask = mask_rgba_or_gray[..., 3]
    elif mask_rgba_or_gray.ndim == 3:
        # RGB면 임계처리
        gray = cv2.cvtColor(mask_rgba_or_gray, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    else:
        mask = mask_rgba_or_gray

    mask = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    eps = epsilon_ratio * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    return [(int(p[0][0]), int(p[0][1])) for p in approx]
