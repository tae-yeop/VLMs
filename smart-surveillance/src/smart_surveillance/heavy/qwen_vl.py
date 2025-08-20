# smart_surveillance/heavy/qwen_vl.py
from __future__ import annotations
from typing import Optional, List
import os
import tempfile
import cv2
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

class QwenVideoQA:
    def __init__(self, model_id: str, device_map: str = "auto"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map=device_map
        ).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def ask(self, video_path: str, prompt: str, fps: float = 1.0, max_new_tokens: int = 128) -> str:
        """Ask Qwen with a video. Falls back to frame-list mode if direct video reading fails."""
        def _build_messages_with_video(uri: str):
            return [{
                "role": "user",
                "content": [
                    {"type": "video", "video": uri, "fps": float(fps)},
                    {"type": "text", "text": prompt}
                ]
            }]

        def _extract_frames(video_file: str, target_fps: float, max_frames: int = 32) -> List[str]:
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return []
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(src_fps / max(0.1, target_fps))))
            tmpdir = tempfile.mkdtemp(prefix="qwen_frames_")
            saved = []
            idx = 0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    outp = os.path.join(tmpdir, f"f_{idx:05d}.jpg")
                    cv2.imwrite(outp, frame)
                    saved.append(f"file://{outp}")
                    idx += 1
                    if len(saved) >= max_frames:
                        break
                frame_idx += 1
            cap.release()
            # Qwen 데모는 frame 리스트 길이가 짝수이길 기대하는 경우가 있어 맞춰준다
            if len(saved) % 2 == 1 and len(saved) > 1:
                saved = saved[:-1]
            return saved

        # 1) 우선: 비디오 URI 직접 전달
        messages = _build_messages_with_video(f"file://{video_path}")
        try:
            text_input = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, vk = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.proc(text=[text_input], images=images, videos=videos, return_tensors="pt",
                               padding=True, **vk).to(self.model.device if hasattr(self.model, "device") else "cuda")
        except Exception:
            # 2) 폴백: 프레임 리스트로 전달
            frames = _extract_frames(video_path, target_fps=fps, max_frames=32)
            if not frames:
                raise
            messages = [{"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": prompt}]}]
            text_input = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, vk = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.proc(text=[text_input], images=images, videos=videos, return_tensors="pt",
                               padding=True, **vk).to(self.model.device if hasattr(self.model, "device") else "cuda")

        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_text = self.proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        return _strip_role_preamble(out_text)

    @torch.no_grad()
    def chat(
        self,
        video_path: str,
        question: str,
        *,
        fps: float = 1.0,
        history: list[tuple[str, str]] | None = None,
        system: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        """
        Structured multi-turn chat over a video.
        - system: optional system prompt
        - history: list of (user, assistant) text turns
        The final user turn includes the video and the question text.
        """

        def _extract_frames(video_file: str, target_fps: float, max_frames: int = 32) -> list[str]:
            import tempfile
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                return []
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(src_fps / max(0.1, target_fps))))
            tmpdir = tempfile.mkdtemp(prefix="qwen_frames_")
            saved: list[str] = []
            idx = 0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    outp = os.path.join(tmpdir, f"f_{idx:05d}.jpg")
                    cv2.imwrite(outp, frame)
                    saved.append(f"file://{outp}")
                    idx += 1
                    if len(saved) >= max_frames:
                        break
                frame_idx += 1
            cap.release()
            if len(saved) % 2 == 1 and len(saved) > 1:
                saved = saved[:-1]
            return saved

        # Try direct video first
        def _build_messages_with_video(uri: str) -> list[dict]:
            msgs: list[dict] = []
            if isinstance(system, str) and system.strip():
                msgs.append({"role": "system", "content": [{"type": "text", "text": system.strip()}]})
            # history as plain text turns
            if history:
                for u, a in history:
                    if isinstance(u, str) and u.strip():
                        msgs.append({"role": "user", "content": [{"type": "text", "text": u.strip()}]})
                    if isinstance(a, str) and a.strip():
                        msgs.append({"role": "assistant", "content": [{"type": "text", "text": a.strip()}]})
            # final user turn with video + question
            msgs.append({
                "role": "user",
                "content": [
                    {"type": "video", "video": uri, "fps": float(fps)},
                    {"type": "text", "text": question.strip()},
                ]
            })
            return msgs

        try:
            messages = _build_messages_with_video(f"file://{video_path}")
            text_input = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, vk = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.proc(text=[text_input], images=images, videos=videos, return_tensors="pt",
                               padding=True, **vk).to(self.model.device if hasattr(self.model, "device") else "cuda")
        except Exception:
            frames = _extract_frames(video_path, target_fps=fps, max_frames=32)
            if not frames:
                raise
            messages = []
            if isinstance(system, str) and system.strip():
                messages.append({"role": "system", "content": [{"type": "text", "text": system.strip()}]})
            if history:
                for u, a in history:
                    if isinstance(u, str) and u.strip():
                        messages.append({"role": "user", "content": [{"type": "text", "text": u.strip()}]})
                    if isinstance(a, str) and a.strip():
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": a.strip()}]})
            messages.append({"role": "user", "content": [{"type": "video", "video": frames}, {"type": "text", "text": question.strip()}]})
            text_input = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, vk = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.proc(text=[text_input], images=images, videos=videos, return_tensors="pt",
                               padding=True, **vk).to(self.model.device if hasattr(self.model, "device") else "cuda")

        out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out_text = self.proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        return _strip_role_preamble(out_text)


def _strip_role_preamble(text: str) -> str:
    """Remove any 'system\n', 'user\n', 'assistant\n' transcript preamble that the
    chat template may include in the decoded text. Keep only the assistant's final answer.
    """
    try:
        s = text.strip()
        low = s.lower()
        # Prefer the last assistant marker
        markers = ["\nassistant\n", "assistant:\n", "\nassistant:", "assistant\n"]
        cut = -1
        for m in markers:
            i = low.rfind(m)
            if i != -1:
                cut = max(cut, i + len(m))
        if cut != -1 and cut < len(s):
            return s[cut:].strip()
        # Fallback: drop leading role lines if present
        lines = s.splitlines()
        filtered = []
        skip_roles = {"system", "user", "assistant"}
        i = 0
        while i < len(lines) and lines[i].strip().lower() in skip_roles:
            i += 1
            # also skip the following line if it's the system content line
            # (best-effort; safe even if not present)
        filtered = lines[i:]
        return "\n".join(filtered).strip()
    except Exception:
        return text.strip()
