from __future__ import annotations
from typing import Dict, Any, List, Tuple
import os, math
import cv2
import numpy as np
from PIL import Image

from smart_surveillance.configs import PipelineConfig
from smart_surveillance.ingestion.video_loader import VideoLoader
from smart_surveillance.detection.yoloe import YOLOEOpenVocabDetector
from smart_surveillance.heavy.qwen_vl import QwenVideoQA


def _build_clip(src_video: str, start_ms: int, end_ms: int, out_path: str) -> Tuple[str, int, float]:
    cap = cv2.VideoCapture(src_video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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


def run_anomaly(
    video_path: str,
    cfg: PipelineConfig,
    enable_qwen: bool = True,
    queries: List[str] | None = None,
    judge_prompt_override: str | None = None,
) -> Dict[str, Any]:
    """
    Generic anomaly detection:
    - Uses open-vocab gate to look for suspicious categories (fire/smoke/weapons/fight/etc.)
    - Uses simple temporal rules for loitering, running, crowding with a lightweight motion estimator
    - Optionally asks Qwen2.5-VL to summarize as a final judge

    Returns dict with keys: verdict, confidence, clip_uri, overlay_uri, explanation, events
    """
    cfg.ensure_dirs()

    # Gate detector: unified YOLOE open-vocab with segmentation
    det = YOLOEOpenVocabDetector(
        yoloe_model_path=cfg.detection.yoloe_model_path or "yoloe-11l-seg.pt",
        score_threshold=min(cfg.detection.score_threshold, cfg.anomaly.detection_conf_threshold),
        max_dets=cfg.detection.max_dets,
        device="cuda",
    )
    loader = VideoLoader(video_path, cfg.ingestion)

    # Very light motion baseline: frame-to-frame optical flow magnitude
    prev_gray = None
    running_candidates: List[Tuple[int, int, List[int]]] = []  # (frame_idx, pts_ms, bbox)
    open_vocab_candidates: List[Tuple[int, int, List[int], str, float]] = []  # add class and score
    person_counts: List[Tuple[int, int]] = []  # (pts_ms, num_persons)

    q = (queries or cfg.anomaly.open_vocab_queries)
    for meta, frame in loader.iterate_frames():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = det.detect(pil, q)

        persons = [d for d in dets if d["cls"].lower() == "person"]
        person_counts.append((meta["pts_ms"], len(persons)))

        # Any suspicious class with sufficient confidence → candidate
        for d in dets:
            cls = d["cls"].lower()
            if cls == "person":
                continue
            if float(d.get("score", 0.0)) >= cfg.anomaly.detection_conf_threshold:
                x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                open_vocab_candidates.append((meta["index"], meta["pts_ms"], [x1,y1,x2,y2], cls, float(d.get("score", 0.0))))

        # Simple running heuristic via average flow in bbox regions (if previous frame exists)
        if prev_gray is not None and len(persons) > 0:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=1, winsize=15,
                                                iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            for d in persons:
                x1,y1,x2,y2 = [int(v) for v in d["bbox"]]
                x1 = max(0, min(x1, mag.shape[1]-1)); x2 = max(0, min(x2, mag.shape[1]-1))
                y1 = max(0, min(y1, mag.shape[0]-1)); y2 = max(0, min(y2, mag.shape[0]-1))
                if x2 <= x1 or y2 <= y1:
                    continue
                roi_mag = mag[y1:y2, x1:x2]
                avg_px = float(np.mean(roi_mag)) * meta["fps"]  # px/frame -> px/sec
                if avg_px >= cfg.anomaly.running_min_px_per_sec:
                    running_candidates.append((meta["index"], meta["pts_ms"], [x1,y1,x2,y2]))

        prev_gray = gray

    # Loitering and crowding evaluation
    events: List[Dict[str, Any]] = []

    # Crowd if consecutive windows exceed threshold
    if len(person_counts) > 0:
        window_ms = 3000
        step = max(1, int(3000 / (person_counts[0][0] + 1e-6) if False else 1))
        # Simplified: check max count
        max_count = max(n for _, n in person_counts)
        if max_count >= cfg.anomaly.crowd_person_threshold:
            t_ms = max(t for t, n in person_counts if n == max_count)
            events.append({"type": "crowd", "t_ms": int(t_ms), "score": 0.7, "meta": {"max_count": int(max_count)}})

    # Loitering: if a person persists in roughly same area; simplified using length of stream
    total_duration_ms = person_counts[-1][0] - person_counts[0][0] if len(person_counts) >= 2 else 0
    if total_duration_ms >= cfg.anomaly.loiter_seconds * 1000 and max(n for _, n in person_counts) > 0:
        events.append({"type": "loiter", "t_ms": int(person_counts[0][0]), "score": 0.6})

    # Running
    for fidx, tms, bbox in running_candidates:
        events.append({"type": "running", "t_ms": int(tms), "bbox": bbox, "score": 0.7})

    # Open-vocab suspicious
    for fidx, tms, bbox, cls, sc in open_vocab_candidates:
        events.append({"type": cls, "t_ms": int(tms), "bbox": bbox, "score": float(sc)})

    # If no events → no anomaly
    if len(events) == 0:
        return {
            "verdict": "NO_ANOMALY",
            "confidence": 0.75,
            "clip_uri": None,
            "overlay_uri": None,
            "explanation": "No suspicious detections or behaviors",
            "events": [],
        }

    # Build a clip around the first event
    first_tms = int(events[0]["t_ms"]) if "t_ms" in events[0] else 0
    pre_ms, post_ms = cfg.heavy.pre_ms, cfg.heavy.post_ms
    start_ms = max(0, first_tms - pre_ms)
    end_ms   = first_tms + post_ms
    out_clip = os.path.join(cfg.work_dir, "clips", f"anomaly_{first_tms}_{start_ms}_{end_ms}.mp4")
    clip_uri, _, _ = _build_clip(video_path, start_ms, end_ms, out_clip)

    # Simple overlay with boxes for events
    overlay_path = os.path.join(cfg.work_dir, "overlays", f"overlay_{os.path.basename(clip_uri)}")
    try:
        cap = cv2.VideoCapture(clip_uri)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps2 = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(overlay_path, fourcc, fps2, (w, h))
        if not vw.isOpened():
            base, _ = os.path.splitext(overlay_path)
            overlay_path = base + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            vw = cv2.VideoWriter(overlay_path, fourcc, fps2, (w, h))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Draw all candidate boxes lightly
            for e in events:
                if "bbox" in e:
                    x1,y1,x2,y2 = [int(v) for v in e["bbox"]]
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    label = e.get("type", "object")
                    cv2.putText(frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            vw.write(frame)
        vw.release(); cap.release()
    except Exception as e:
        print("[WARN] anomaly overlay failed:", e)
        overlay_path = clip_uri

    verdict = "ANOMALY"
    conf = 0.8
    explanation = ", ".join(sorted({e["type"] for e in events}))
    prompt_used = None

    if enable_qwen:
        try:
            qwen = QwenVideoQA(cfg.heavy.qwen_model_id)
            prompt = (judge_prompt_override.strip() if isinstance(judge_prompt_override, str) and judge_prompt_override.strip() else cfg.anomaly.general_anomaly_prompt)
            prompt_used = prompt
            answer = qwen.ask(overlay_path if os.path.exists(overlay_path) else clip_uri,
                              prompt,
                              fps=cfg.heavy.qwen_fps,
                              max_new_tokens=64)
            explanation = answer
            ans_low = answer.lower()
            if "no_anomaly" in ans_low or "no anomaly" in ans_low:
                verdict, conf = "NO_ANOMALY", 0.85
            elif "uncertain" in ans_low:
                verdict, conf = "UNCERTAIN", 0.6
        except Exception as e:
            print("[WARN] Qwen judge failed:", e)

    return {
        "verdict": verdict,
        "confidence": float(conf),
        "clip_uri": clip_uri,
        "overlay_uri": overlay_path if 'overlay_path' in locals() else clip_uri,
        "explanation": explanation,
        "events": events,
        "prompt_used": prompt_used,
        "mode": "anomaly",
    }


