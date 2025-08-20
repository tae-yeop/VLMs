# smart_surveillance/pipeline/trespass.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, math, json
import cv2
import numpy as np
from PIL import Image

from smart_surveillance.configs import PipelineConfig
from smart_surveillance.ingestion.video_loader import VideoLoader
from smart_surveillance.detection.yoloworld import YOLOWorldDetector
from smart_surveillance.heavy.grd import GroundingDINOWrapper
from smart_surveillance.heavy.sam2_video import SAM2Video
from smart_surveillance.heavy.qwen_vl import QwenVideoQA


def _point_in_polygon(pt: Tuple[int,int], poly: List[Tuple[int,int]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1
            if x < xin:
                inside = not inside
    return inside


def _ensure_roi(cfg: PipelineConfig) -> List[Tuple[int,int]]:
    cam_id = cfg.roi.default_cam_id
    if cam_id not in cfg.roi.rois or not cfg.roi.rois[cam_id]:
        raise ValueError("ROI polygon not set in cfg.roi.rois[default_cam_id]")
    return cfg.roi.rois[cam_id]


def _build_clip(src_video: str, start_ms: int, end_ms: int, out_path: str) -> Tuple[str, int, float]:
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # try mp4v → fallback to MJPG(.avi)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if (not vw.isOpened()) or fps <= 0:
        base, _ = os.path.splitext(out_path)
        out_path = base + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = fps or 30.0
        vw = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not vw.isOpened():
            raise RuntimeError("Failed to open VideoWriter (mp4v and MJPG)")

    start_idx = max(0, int(math.floor((start_ms/1000.0) * fps)))
    end_idx   = int(math.ceil((end_ms/1000.0) * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    idx = start_idx
    while idx <= end_idx:
        ret, frame = cap.read()
        if not ret:
            break
        vw.write(frame)
        idx += 1
    vw.release(); cap.release()
    return out_path, start_idx, fps


def _get_roi_or_none(cfg: PipelineConfig) -> List[Tuple[int,int]] | None:
    cam_id = cfg.roi.default_cam_id
    poly = cfg.roi.rois.get(cam_id)
    return poly if (poly and len(poly) >= 3) else None


def run_trespass(
    video_path: str,
    cfg: PipelineConfig,
    enable_grd: bool = True,
    enable_sam2: bool = True,
    enable_qwen: bool = True,
    queries: List[str] | None = None,
    judge_prompt_override: str | None = None,
) -> Dict[str, Any]:
    """
    Returns an EventVerdict-like dict:
    {
      "verdict": "TRESPASS|NO_TRESPASS|UNCERTAIN",
      "confidence": float,
      "clip_uri": "...",
      "explanation": "...",
      "seed_box": [x1,y1,x2,y2]
    }
    """
    cfg.ensure_dirs()
    # roi_poly = _ensure_roi(cfg)
    roi_poly = _get_roi_or_none(cfg)


    # --- Gate detector (YOLO-World or OWL-ViT fallback) ---
    from PIL import Image
    det = YOLOWorldDetector(backend=cfg.detection.backend,
                            yoloworld_model_path=cfg.detection.yoloworld_model_path,
                            owlvit_model_id=cfg.detection.owlvit_model_id,
                            score_threshold=cfg.detection.score_threshold,
                            max_dets=cfg.detection.max_dets,
                            device="cuda")
    loader = VideoLoader(video_path, cfg.ingestion)

    trigger = None  # (frame_idx, pts_ms, bbox)
    first_frame_img = None

    # Prepare queries: trespass requires 'person'
    q = (queries or cfg.detection.open_vocab_queries or ["person"]).copy()
    q_lower = [s.lower().strip() for s in q]

    trigger_label = None
    for meta, frame in loader.iterate_frames(): # frame = np.array(1080, 1920, 3)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = det.detect(pil, q)
        # 선택된 클래스만 골라 ROI 내부인지 확인 (ROI 없으면 아무 위치)
        for d in dets:
            if d["cls"].lower() not in q_lower: continue
            x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)
            if roi_poly is None:  # 일반 모드
                trigger = (meta["index"], meta["pts_ms"], [x1, y1, x2, y2])
                trigger_label = d.get("cls", "object")
                first_frame_img = pil
                break
            else:
                if _point_in_polygon((cx, cy), roi_poly):
                    trigger = (meta["index"], meta["pts_ms"], [x1, y1, x2, y2])
                    trigger_label = d.get("cls", "object")
                    first_frame_img = pil
                    break
        if trigger: break

    if not trigger:
        reason = (
            f"No target entering ROI (targets: {', '.join(q)})" if roi_poly is not None
            else f"No target detected (targets: {', '.join(q)})"
        )
        return {"verdict": "NO_TRESPASS", "confidence": 0.5, "explanation": reason, "clip_uri": None}

    # --- Build event clip around trigger ---
    pre_ms, post_ms = cfg.heavy.pre_ms, cfg.heavy.post_ms
    start_ms = max(0, trigger[1] - pre_ms)
    end_ms   = trigger[1] + post_ms
    out_clip = os.path.join(cfg.work_dir, "clips", f"evt_{trigger[0]}_{start_ms}_{end_ms}.mp4")
    clip_uri, start_frame_idx, fps = _build_clip(video_path, start_ms, end_ms, out_clip)

    # --- Grounding DINO refine (optional on the first frame of the clip) ---
    seed_box = trigger[2]
    was_refined = False
    grd_label = None
    if enable_grd:
        try:
            grd = GroundingDINOWrapper(cfg.heavy.grd_model_id, device="cuda")
            # Prefer the triggered label if available; otherwise use user's query list
            queries_for_grd = [trigger_label] if (locals().get("trigger_label") and trigger_label) else (q if 'q' in locals() else ["person"])  # type: ignore
            refined = grd.refine(first_frame_img, queries_for_grd, seed_boxes=[trigger[2]])  # bbox 나옴
            # pick highest score
            if refined and len(refined) > 0:
                best = max(refined, key=lambda d: float(d.get("score", 0.0)))
                if "bbox" in best:
                    seed_box = best["bbox"]
                    grd_label = str(best.get("cls", trigger_label or "object"))
                    was_refined = True
        except Exception as e:
            print("[WARN] GroundingDINO refine failed:", e)

    # --- SAM2 tracking/mask over the clip (optional) ---
    tracked: Dict[int, np.ndarray] = {}
    if enable_sam2:
        try:
            sam2 = SAM2Video(cfg.heavy.sam2_model_id, device="cuda")
            tracked = sam2.track(clip_uri, init_frame_idx=0, init_box_xyxy=seed_box)  # 클립 기준 0 프레임
        except Exception as e:
            print("[WARN] SAM2 tracking failed:", e)
            tracked = {}

    # --- Always create overlay video with ROI + bboxes (+ optional mask) ---
    overlay_path = os.path.join(cfg.work_dir, "overlays", f"overlay_{os.path.basename(clip_uri)}")
    try:
        cap = cv2.VideoCapture(clip_uri)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps2 = float(cap.get(cv2.CAP_PROP_FPS) or fps)

        # try avc1 → mp4v → MJPG(.avi)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        vw = cv2.VideoWriter(overlay_path, fourcc, fps2, (w, h))
        if not vw.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vw = cv2.VideoWriter(overlay_path, fourcc, fps2, (w, h))
        if not vw.isOpened():
            base, _ = os.path.splitext(overlay_path)
            overlay_path = base + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            vw = cv2.VideoWriter(overlay_path, fourcc, fps2, (w, h))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Draw ROI polygon (green)
            if roi_poly is not None and len(roi_poly) >= 3:
                pts = np.array(roi_poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw gate bbox (blue)
            gx1, gy1, gx2, gy2 = [int(v) for v in trigger[2]]
            cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
            gate_txt = f"gate:{trigger_label}" if trigger_label else "gate"
            cv2.putText(frame, gate_txt, (gx1, max(0, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Draw refined/seed bbox (red)
            rx1, ry1, rx2, ry2 = [int(v) for v in seed_box]
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
            if was_refined:
                refined_txt = f"grd:{grd_label or trigger_label or 'object'}"
            else:
                refined_txt = f"seed:{trigger_label or 'object'}"
            cv2.putText(frame, refined_txt, (rx1, max(0, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Optional: blend SAM2 mask (semi-transparent red)
            if tracked and (idx in tracked):
                m = (tracked[idx] > 0).astype(np.uint8)
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                overlay[m.astype(bool)] = (255, 0, 0, 100)
                frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                from PIL import Image as PILImage
                pf = PILImage.fromarray(frame_rgba); po = PILImage.fromarray(overlay)
                pf = PILImage.alpha_composite(pf, po)
                frame = cv2.cvtColor(np.array(pf), cv2.COLOR_RGBA2BGR)
                # draw tracked bbox (yellow) following the mask for this frame
                ys, xs = np.where(m > 0)
                if xs.size > 0 and ys.size > 0:
                    x1, y1, w_box, h_box = cv2.boundingRect(np.column_stack((xs, ys)))
                    x2, y2 = x1 + w_box, y1 + h_box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                    cv2.putText(frame, "tracked", (int(x1), max(0, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            vw.write(frame); idx += 1
        vw.release(); cap.release()
    except Exception as e:
        print("[WARN] overlay failed:", e)
        overlay_path = clip_uri

    # --- Qwen judge (optional) ---
    verdict = "UNCERTAIN"; conf = 0.6; answer = ""; prompt_used = None
    if enable_qwen:
        try:
            qwen = QwenVideoQA(cfg.heavy.qwen_model_id)
            prompt = (judge_prompt_override.strip() if isinstance(judge_prompt_override, str) and judge_prompt_override.strip() else
                      (cfg.heavy.trespass_prompt if roi_poly is not None else cfg.heavy.general_prompt))
            prompt_used = prompt
            qwen_video_path = overlay_path if os.path.exists(overlay_path) else clip_uri
            answer = qwen.ask(qwen_video_path, prompt, fps=cfg.heavy.qwen_fps, max_new_tokens=64)
            ans_low = answer.lower()
            if "yes" in ans_low:
                verdict, conf = "TRESPASS", 0.85
            elif "no" in ans_low:
                verdict, conf = "NO_TRESPASS", 0.85
        except Exception as e:
            print("[WARN] Qwen judge failed:", e)
            answer = "Judge failed"

    return {
        "verdict": verdict,
        "confidence": conf,
        "clip_uri": clip_uri,
        "overlay_uri": overlay_path if 'overlay_path' in locals() else clip_uri,
        "seed_box": seed_box,
        "trigger_label": trigger_label,
        "grd_refined": was_refined,
        "grd_label": grd_label,
        "prompt_used": prompt_used,
        "explanation": answer,
        "mode": "roi" if roi_poly is not None else "general"
    }