from transformers import Sam2VideoModel, Sam2VideoProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam2VideoModel.from_pretrained("facebook/sam2-hiera-large").to(device, dtype=torch.bfloat16)
processor = Sam2VideoProcessor.from_pretrained("facebook/sam2-hiera-large")

# Load video frames (example assumes you have a list of PIL Images)
# video_frames = [Image.open(f"frame_{i:05d}.jpg") for i in range(num_frames)]

# For this example, we'll use the video loading utility
from transformers.video_utils import load_video
video_url = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/bedroom.mp4"
video_frames, _ = load_video(video_url)

# Initialize video inference session
inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    torch_dtype=torch.bfloat16,
)

# Add click on first frame to select object
ann_frame_idx = 0
ann_obj_id = 1
points = [[[[210, 350]]]]
labels = [[[1]]]

processor.add_inputs_to_inference_session(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
    obj_ids=ann_obj_id,
    input_points=points,
    input_labels=labels,
)

# Segment the object on the first frame
outputs = model(
    inference_session=inference_session,
    frame_idx=ann_frame_idx,
)
video_res_masks = processor.post_process_masks(
    [outputs.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
)[0]
print(f"Segmentation shape: {video_res_masks.shape}")

# Propagate through the entire video
video_segments = {}
for sam2_video_output in model.propagate_in_video_iterator(inference_session):
    video_res_masks = processor.post_process_masks(
        [sam2_video_output.pred_masks], original_sizes=[[inference_session.video_height, inference_session.video_width]], binarize=False
    )[0]
    video_segments[sam2_video_output.frame_idx] = video_res_masks

print(f"Tracked object through {len(video_segments)} frames")


import numpy as np
import cv2

# ---- 타입/파라미터 안전화 ----
H_raw = getattr(inference_session, "video_height", None)
W_raw = getattr(inference_session, "video_width", None)
H = int(H_raw) if H_raw is not None else int(np.array(video_frames[0]).shape[0])
W = int(W_raw) if W_raw is not None else int(np.array(video_frames[0]).shape[1])

# fps 확보 (없으면 30.0)
fps = None
try:
    from transformers.video_utils import load_video
    # 위에서 이미 load_video를 호출했다면, video_info를 그때 받아두세요.
    # 여기서는 방어적으로 fps가 None이면 30으로.
except:
    pass
fps = float(fps) if fps is not None else 30.0
if not (fps > 0):
    fps = 30.0

# ---- VideoWriter 안전 오픈 ----
def open_writer(path, width, height, fps):
    # 코덱 후보 순회
    fourcc_list = ["mp4v", "avc1", "H264", "XVID", "MJPG"]  # 환경에 따라 가용 코덱 다름
    for cc in fourcc_list:
        fourcc = cv2.VideoWriter_fourcc(*cc)
        w = cv2.VideoWriter(path, fourcc, float(fps), (int(width), int(height)))
        if w.isOpened():
            print(f"[VideoWriter] Opened with FOURCC={cc}, fps={fps}, size={(width, height)}")
            return w
    raise RuntimeError("Failed to open VideoWriter with any FOURCC. "
                       "Check OpenCV build (FFmpeg) and install codecs.")

output_path = "sam2_overlay.mp4"
writer = open_writer(output_path, W, H, fps)

# ---- 팔레트/오버레이 설정 ----
palette = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 255), (128, 255, 128), (255, 128, 128),
]
base_alpha = 0.6  # 0~1

# 첫 프레임 결과도 dict에 포함(누락 방지)
video_segments[ann_frame_idx] = video_res_masks  # [num_objs, H, W], float probs

# ---- 프레임 루프: 오버레이 + 리사이즈 일치 ----
for idx, pil_im in enumerate(video_frames):
    rgb = np.array(pil_im)                     # (h0, w0, 3), uint8, RGB
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)

    # 프레임 크기가 writer와 다르면 리사이즈
    h0, w0 = frame_bgr.shape[:2]
    if (w0, h0) != (W, H):
        frame_bgr = cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

    masks_t = video_segments.get(idx, None)    # torch.Tensor or None
    if masks_t is not None:
        masks = masks_t.detach().cpu().float().numpy()  # [num_objs, H, W]
        # (주의) 마스크 크기가 writer 사이즈와 다르면 리사이즈
        mH, mW = masks.shape[-2], masks.shape[-1]
        if (mW, mH) != (W, H):
            # 개체 축 보존한 채, 각 마스크를 프레임 크기로 맞춤
            resized = []
            for k in range(masks.shape[0]):
                resized.append(cv2.resize(masks[k], (W, H), interpolation=cv2.INTER_LINEAR))
            masks = np.stack(resized, axis=0)

        for obj_i in range(masks.shape[0]):
            prob = masks[obj_i]  # (H, W), float32 in [0,1]
            if prob.max() <= 0:
                continue
            color = np.array(palette[obj_i % len(palette)], dtype=np.float32)  # BGR
            alpha_map = (prob * base_alpha)[..., None]  # (H, W, 1)

            frame_bgr = frame_bgr * (1.0 - alpha_map) + color * alpha_map

            # (옵션) 경계선
            bin_mask = (prob > 0.5).astype(np.uint8) * 255
            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_bgr, contours, -1, color.tolist(), thickness=2)

    writer.write(np.clip(frame_bgr, 0, 255).astype(np.uint8))

writer.release()
print(f"Saved overlay video to: {output_path}")
