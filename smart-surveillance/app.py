# app.py (요약/추가)
import os, sys, json, cv2, numpy as np
# Ensure local `src` package is importable before any package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
from PIL import Image


from smart_surveillance.configs import PipelineConfig
from smart_surveillance.pipeline.trespass import run_trespass
from smart_surveillance.anomaly import run_anomaly
from smart_surveillance.utils.roi import polygon_from_mask

from smart_surveillance.detection.yoloworld import YOLOWorldDetector
from smart_surveillance.heavy.grd import GroundingDINOWrapper
from smart_surveillance.heavy.sam2_video import SAM2Video
from smart_surveillance.heavy.qwen_vl import QwenVideoQA

# 샘플 비디오 경로: 우선순위 = 환경변수 -> ./assets/demo.mp4
DEFAULT_SAMPLE = "/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLMs/smart-surveillance/samples/demo.mp4"
DEFAULT_QUERIES = ", ".join(PipelineConfig().anomaly.open_vocab_queries)

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

def list_sample_files():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    allow = {".mp4", ".avi", ".mov", ".mkv"}
    files = []
    for name in sorted(os.listdir(SAMPLES_DIR)):
        p = os.path.join(SAMPLES_DIR, name)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in allow:
            files.append(p)
    return files

def list_sample_choices():
    return [(os.path.basename(p), p) for p in list_sample_files()]


# import debugpy

# debugpy.listen(('0.0.0.0', 5678))

# print("Waiting for debugger attach")
# debugpy.wait_for_client()
def sample_exists() -> bool:
    return bool(DEFAULT_SAMPLE) and os.path.exists(DEFAULT_SAMPLE)

def extract_first_frame(video_path: str) -> Image.Image:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(f"Cannot open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise gr.Error("Failed to read first frame")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _editor_value(pil_image: Image.Image | None):
    return {
        "background": pil_image,
        "layers": [],
        "composite": pil_image,  # <- 추가
    }


def on_start():
    """
    앱 로드시: 샘플 비디오가 있으면 자동 세팅.
    반환: [video_in, editor, status, roi_state, video_state]  (5개!)
    """
    if sample_exists():
        frame0 = extract_first_frame(DEFAULT_SAMPLE)
        return [
            DEFAULT_SAMPLE,
            _editor_value(frame0),
            f"Sample video loaded: `{DEFAULT_SAMPLE}`\n- Draw a mask to set ROI.",
            [],
            DEFAULT_SAMPLE,
        ]
    else:
        return [None, _editor_value(None), "Upload a video (or set DEMO_VIDEO / assets/demo.mp4).", [], None]

def handle_select_video(video_file, video_state):
    """
    ① 프레임 불러오기 / 소스 확정
    반환: [editor, status, roi_state, video_state]
    """
    path = None
    # gr.Video가 filepath 문자열을 그대로 넘기는 모드도 있음
    if isinstance(video_file, str) and os.path.exists(video_file):
        path = video_file
    elif video_state and os.path.exists(video_state):
        path = video_state
    elif sample_exists():
        path = DEFAULT_SAMPLE

    if not path:
        raise gr.Error("No video found. Upload a video or set DEMO_VIDEO / assets/demo.mp4.")

    frame0 = extract_first_frame(path)
    status = f"Video selected: `{path}`\n- Draw a mask to set ROI."
    return _editor_value(frame0), status, [], path

def handle_select_video_full(video_file, video_state):
    """
    ① 프레임 불러오기 / 소스 확정 (Video 컴포넌트 값도 함께 업데이트)
    반환: [video_in_value, editor, status, roi_state, video_state]
    """
    path = None
    if isinstance(video_file, str) and os.path.exists(video_file):
        path = video_file
    elif video_state and os.path.exists(video_state):
        path = video_state
    elif sample_exists():
        path = DEFAULT_SAMPLE

    if not path:
        raise gr.Error("No video found. Upload a video or set DEMO_VIDEO / assets/demo.mp4.")

    frame0 = extract_first_frame(path)
    status = f"Video selected: `{path}`\n- Draw a mask to set ROI."
    return path, _editor_value(frame0), status, [], path

def _mask_from_image_like(im):
    if im is None:
        return None
    if isinstance(im, Image.Image):
        if im.mode == "RGBA":
            return np.array(im.split()[-1])
        if im.mode in ("L", "1"):
            return np.array(im)
        return np.array(im.convert("L"))
    if isinstance(im, np.ndarray):
        if im.ndim == 2:
            return im
        if im.ndim == 3 and im.shape[2] == 4:
            return im[..., 3]
        if im.ndim == 3:
            return np.mean(im, axis=2)
    return None

def use_drawn_roi(image_editor_data):
    """
    현재 마스크를 ROI로 사용 → 다각형 추출
    반환: [status_md, roi_state]
    """
    if not image_editor_data:
        return "No mask provided.", []

    mask_np = None

    # 1) (구버전) 최상위 mask가 바로 있는 경우
    if image_editor_data.get("mask") is not None:
        mask_np = np.array(image_editor_data["mask"])
    else:
        # 2) Gradio 4.x: layers에 mask 또는 image가 들어올 수 있음
        layers = image_editor_data.get("layers") or []
        for lyr in layers:
            mnp = None
            if isinstance(lyr, dict):
                m = lyr.get("mask")
                if m is not None:
                    mnp = np.array(m)
                else:
                    mnp = _mask_from_image_like(lyr.get("image"))
            elif isinstance(lyr, Image.Image) or isinstance(lyr, np.ndarray):
                mnp = _mask_from_image_like(lyr)

            if mnp is not None:
                mnp = (mnp > 0).astype(np.uint8) * 255
                mask_np = mnp if mask_np is None else np.maximum(mask_np, mnp)

    if mask_np is None:
        return "No mask provided.", []

    poly = polygon_from_mask(mask_np, epsilon_ratio=0.01)
    if len(poly) < 3:
        return "Mask too small or invalid. Please draw a closed region.", []
    return f"ROI set: {len(poly)}-gon", poly

def clear_roi(video_state):
    """
    ROI 초기화: 현재 영상의 첫 프레임을 다시 에디터에 표시
    반환: [editor, status_md, roi_state]
    """
    if video_state and os.path.exists(video_state):
        frame0 = extract_first_frame(video_state)
        return _editor_value(frame0), "ROI cleared. Draw a new mask.", []
    return _editor_value(None), "ROI cleared. Load a video first.", []

def run_pipeline_gr(video_state, backend, image_editor_data, roi_poly_state, enable_grd, enable_sam2, enable_qwen, mode, queries_text, judge_prompt_preset, judge_prompt_override):
    """
    ③ 실행: 선택된 비디오 + (옵션) ROI로 파이프라인 수행
    반환: [overlay_video, markdown_summary, raw_json]
    """
    if not (video_state and os.path.exists(video_state)):
        raise gr.Error("No video selected. Load or upload a video first.")

    cfg = PipelineConfig()
    cfg.detection.backend = backend
    cfg.ensure_dirs()

    # Parse user queries (comma/line separated)
    user_queries = []
    if isinstance(queries_text, str) and len(queries_text.strip()) > 0:
        raw = queries_text.replace("\n", ",")
        user_queries = [s.strip() for s in raw.split(",") if s.strip()]

    # ROI: state 우선, 없으면 editor의 mask 사용
    poly = roi_poly_state or []
    if (not poly) and image_editor_data and image_editor_data.get("mask") is not None:
        mask = np.array(image_editor_data["mask"])
        poly = polygon_from_mask(mask, epsilon_ratio=0.01)

    if poly and len(poly) >= 3:
        cfg.roi.rois[cfg.roi.default_cam_id] = [tuple(p) for p in poly]

    # 실행
    if mode == "anomaly":
        res = run_anomaly(
            video_state,
            cfg,
            enable_qwen=bool(enable_qwen),
            queries=user_queries or None,
            judge_prompt_override=(judge_prompt_override or judge_prompt_preset or None),
        )
    else:
        res = run_trespass(
            video_state,
            cfg,
            enable_grd=bool(enable_grd),
            enable_sam2=bool(enable_sam2),
            enable_qwen=bool(enable_qwen),
            queries=user_queries or None,
            judge_prompt_override=(judge_prompt_override or judge_prompt_preset or None),
        )
    
    overlay = res.get("overlay_uri") or res.get("clip_uri")
    verdict = res.get("verdict")
    conf = float(res.get("confidence", 0.0))
    explanation = res.get("explanation", "")
    mode = res.get("mode", "general" if not poly else "roi")

    md = f"### Verdict: **{verdict}** (conf {conf:.2f})\n- Mode: `{mode}`\n- Explanation: {explanation}"
    return overlay, md, json.dumps(res, indent=2, ensure_ascii=False), (res.get("overlay_uri") or res.get("clip_uri") or None)


def _prompt_presets():
    return {
        "Trespass (YES/NO/UNCERTAIN)": PipelineConfig().heavy.trespass_prompt,
        "General anomaly summary": PipelineConfig().heavy.general_prompt,
        "Safety hazard checklist": (
            "Identify safety hazards (fire/smoke, slippery floor, crowding, PPE missing). "
            "Start with [HAZARD, NO_HAZARD, UNCERTAIN] and list 1-3 hazards."
        ),
        "Action timeline": (
            "Provide a brief timeline of notable actions in the clip in 3 bullet points."
        ),
        "Object inventory": (
            "List notable objects detected in the clip (top-10)."
        ),
    }

def chat_send(chat_state, chat_input, clip_uri, chat_system):
    if not clip_uri or not os.path.exists(clip_uri):
        return chat_state, "No clip to chat about. Run the pipeline first."
    if not isinstance(chat_input, str) or not chat_input.strip():
        return chat_state, None
    try:
        qwen = QwenVideoQA(PipelineConfig().heavy.qwen_model_id)
        system_inst = chat_system.strip() if isinstance(chat_system, str) else "You are a helpful surveillance analyst. Answer strictly based on the provided clip."
        # Use structured chat to avoid literal 'User:'/'Assistant:' tokens in output
        answer = qwen.chat(
            clip_uri,
            chat_input.strip(),
            fps=PipelineConfig().heavy.qwen_fps,
            history=(chat_state or []),
            system=system_inst,
            max_new_tokens=128,
        )
        new_state = (chat_state or []) + [(chat_input.strip(), answer)]
        return new_state, ""
    except Exception as e:
        err = f"Chat failed: {e}"
        new_state = (chat_state or []) + [(chat_input.strip(), err)]
        return new_state, ""


def warmup_models():
    try:
        # 두 백엔드 가볍게 로드(가중치 캐시 목적)
        YOLOWorldDetector(backend="yoloworld", yoloworld_model_path="yolov8s-worldv2.pt", device="cuda")
    except Exception as e:
        print("[WARMUP] YOLO-World warmup failed:", e)
    try:
        YOLOWorldDetector(backend="owlvit", device="cuda")
    except Exception as e:
        print("[WARMUP] OWL-ViT warmup failed:", e)
    try:
        GroundingDINOWrapper("IDEA-Research/grounding-dino-base", device="cuda")
    except Exception as e:
        print("[WARMUP] GroundingDINO warmup failed:", e)
    try:
        SAM2Video("facebook/sam2-hiera-large", device="cuda")
    except Exception as e:
        print("[WARMUP] SAM2 warmup failed:", e)
    try:
        QwenVideoQA("Qwen/Qwen2.5-VL-7B-Instruct")
    except Exception as e:
        print("[WARMUP] Qwen-VL warmup failed:", e)
    return None

with gr.Blocks(title="Smart Surveillance (Trespass & Anomaly)") as demo:
    gr.Markdown("# 스마트 관제 데모 (월담/일반 이상탐지)  \n- 업로드한 **MP4**의 첫 프레임 위에 **마스크를 그려 ROI**를 지정하세요.  \n- 'Anomaly' 모드에서는 ROI 없이 일반적인 이상 탐지를 수행합니다.  \n- 오픈보캡 프롬프트는 영어를 권장합니다(e.g., suitcase, luggage, smoke, fire, crowd).")
    # 상태
    roi_state = gr.State([])       # [(x,y), ...]
    video_state = gr.State(None)   # 현재 선택된 비디오 경로

    # ① 소스 선택
    gr.Markdown("## ① 소스 선택")
    video_in = gr.Video(
        sources=["upload", "webcam"],
        label="입력 동영상",
        interactive=True,
        value=DEFAULT_SAMPLE if sample_exists() else None,
    )
    with gr.Row():
        load_btn = gr.Button("프레임 불러오기 / 소스 확정")
        samples_dd = gr.Dropdown(choices=list_sample_choices(), label="샘플 선택", value=(DEFAULT_SAMPLE if sample_exists() else None))

    # 상태/안내
    status = gr.Markdown("Loading...")

    # ② ROI 그리기
    gr.Markdown("## ② ROI 그리기 (선택)")
    editor = gr.ImageEditor(
        label="ROI Editor (마스크로 영역을 색칠하세요)",
        interactive=True,
        brush=gr.Brush(default_size=40)
    )
    with gr.Row():
        use_mask_btn = gr.Button("현재 마스크를 ROI로 사용")
        clear_btn = gr.Button("ROI 초기화")

    # ③ 실행
    gr.Markdown("## ③ 실행")
    mode = gr.Radio(choices=["trespass", "anomaly"], value="trespass", label="모드")
    backend = gr.Radio(choices=["owlvit", "yoloworld"], value="owlvit", label="게이트 백엔드")
    queries_text = gr.Textbox(label="오픈보캡 프롬프트(쉼표/줄바꿈 구분)", value=DEFAULT_QUERIES, placeholder="person, fire, smoke, gun, knife, fight")
    # Heavy components toggles
    with gr.Row():
        enable_grd = gr.Checkbox(value=True, label="GroundingDINO")
        enable_sam2 = gr.Checkbox(value=True, label="SAM2")
        enable_qwen = gr.Checkbox(value=True, label="Qwen-VL Judge")
    with gr.Row():
        judge_prompt_preset = gr.Dropdown(choices=list(_prompt_presets().keys()), value="Trespass (YES/NO/UNCERTAIN)", label="Qwen Judge 프리셋")
        judge_prompt_override = gr.Textbox(label="Qwen Judge 프롬프트 직접 입력(프리셋보다 우선)")
    run_btn = gr.Button("실행")

    # 출력
    gr.Markdown("## 출력")
    out_video = gr.Video(label="결과 오버레이 (있으면)")
    out_md = gr.Markdown(label="요약")
    out_json = gr.Textbox(label="Raw JSON", lines=18)
    out_clip_state = gr.State(None)

    # ✅ 초기 로드: editor를 한 번만 outputs로 지정 (5개 반환!)
    demo.load(
        fn=on_start,
        inputs=[],
        outputs=[video_in, editor, status, roi_state, video_state],
        show_progress=False
    )
    demo.load(fn=warmup_models, inputs=[], outputs=[], show_progress=False)

    # ✅ 프레임 불러오기 / 소스 확정
    load_btn.click(
        fn=handle_select_video,
        inputs=[video_in, video_state],
        outputs=[editor, status, roi_state, video_state]
    )

    # ROI 관련
    use_mask_btn.click(fn=use_drawn_roi, inputs=[editor], outputs=[status, roi_state])
    clear_btn.click(fn=clear_roi, inputs=[video_state], outputs=[editor, status, roi_state])

    # 샘플 선택 → 즉시 미리보기/상태 업데이트
    samples_dd.change(
        fn=handle_select_video_full,
        inputs=[samples_dd, video_state],
        outputs=[video_in, editor, status, roi_state, video_state]
    )

    # 실행
    run_btn.click(
        fn=run_pipeline_gr,
        inputs=[video_state, backend, editor, roi_state, enable_grd, enable_sam2, enable_qwen, mode, queries_text, judge_prompt_preset, judge_prompt_override],
        outputs=[out_video, out_md, out_json, out_clip_state]
    )

    # ④ 클립 대화 (Qwen‑VL)
    gr.Markdown("## ④ 클립 대화 (Qwen‑VL)")
    chat_system = gr.Textbox(label="시스템 지시문", value="You are a helpful surveillance analyst. Answer strictly based on the provided clip.")
    chatbox = gr.Chatbot(label="Chat")
    chat_input = gr.Textbox(label="질문")
    chat_send_btn = gr.Button("전송")
    chat_send_btn.click(
        fn=chat_send,
        inputs=[chatbox, chat_input, out_clip_state, chat_system],
        outputs=[chatbox, status]
    )

if __name__ == "__main__":
    print("Starting Gradio app...")
    print(f"Will launch on: 0.0.0.0:7861")
    # demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
    demo.queue().launch(share=True, debug=True, server_name="0.0.0.0", server_port=7867)


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--video", required=True)
#     ap.add_argument("--roi", required=False, help='Optional ROI polygon: "[[100,50],[400,50],[400,300],[100,300]]" or json file. If not provided, uses full frame detection.')
#     ap.add_argument("--backend", default="owlvit", choices=["owlvit","yoloworld"])
#     args = ap.parse_args()

#     cfg = PipelineConfig()
#     cfg.detection.backend = args.backend
#     # ROI 세팅
#     if args.roi:
#         if args.roi.endswith(".json") and os.path.exists(args.roi):
#             poly = json.load(open(args.roi))
#         else:
#             poly = json.loads(args.roi)
#         cfg.roi.rois[cfg.roi.default_cam_id] = [tuple(p) for p in poly]

#     res = run_trespass(args.video, cfg)
#     print(json.dumps(res, indent=2, ensure_ascii=False))
