# smart_surveillance/configs.py
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class IngestionConfig:
    every_nth_frame: int = 5
    max_frames: Optional[int] = None
    resize_width: Optional[int] = None
    resize_height: Optional[int] = None

@dataclass
class ROIConfig:
    # cam_id -> list of polygon points
    rois: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    default_cam_id: str = "default_cam"

@dataclass
class DetectionConfig:
    backend: str = "owlvit"  # "yoloworld" or "owlvit"
    # YOLO-World: (선택) 로컬 엔진 경로/모델명 등 (환경별로 채워 쓰기)
    yoloworld_model_path: Optional[str] = None
    # OWL-ViT(폴백) 모델 ID
    owlvit_model_id: str = "google/owlv2-base-patch16"
    # 게이트에서 사용할 오픈보캡 프롬프트
    open_vocab_queries: List[str] = field(default_factory=lambda: ["person"])
    score_threshold: float = 0.25
    max_dets: int = 50

@dataclass
class HeavyConfig:
    grd_model_id: str = "IDEA-Research/grounding-dino-base"
    sam2_model_id: str = "facebook/sam2-hiera-large"
    qwen_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # 이벤트 클립 범위(ms)
    pre_ms: int = 2000
    post_ms: int = 3000
    # Qwen 프롬프트
    trespass_prompt: str = (
        "Did the person trespass over the fence? "
        "Answer strictly with one of [YES, NO, UNCERTAIN] and then one short reason."
    )
    # ✅ ROI가 없을 때 사용할 일반 모드 프롬프트
    general_prompt: str = (
        "Briefly summarize any suspicious or unsafe behavior in this clip. "
        "If a fence trespass occurred, start with one of [YES, NO, UNCERTAIN] about trespass, "
        "then add one short reason."
    )
    qwen_fps: float = 1.0  # 저비용 요약을 위해 낮은 FPS

@dataclass
class AnomalyConfig:
    # Open-vocab queries for generic anomaly gate (in addition to person)
    open_vocab_queries: List[str] = field(
        default_factory=lambda: [
            "person",
            "fire",
            "smoke",
            "gun",
            "knife",
            "fight",
            "blood",
            "explosion",
            "helmet",
            "no helmet",
            "running person",
            "car",
            "motorcycle",
            "bicycle",
        ]
    )
    # Behavioral rule thresholds
    loiter_seconds: float = 15.0
    running_min_px_per_sec: float = 250.0
    crowd_person_threshold: int = 8
    # Confidence threshold to treat open-vocab detection as anomaly
    detection_conf_threshold: float = 0.35
    # Qwen prompt for generic anomaly summarization
    general_anomaly_prompt: str = (
        "Identify whether this clip contains suspicious, unsafe, or abnormal events "
        "such as fire/smoke, violence, weapons, crowding, running, or loitering. "
        "Start with one of [ANOMALY, NO_ANOMALY, UNCERTAIN] and then one brief reason."
    )

@dataclass
class PipelineConfig:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    heavy: HeavyConfig = field(default_factory=HeavyConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    work_dir: str = "./runs"

    def ensure_dirs(self):
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "clips"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "overlays"), exist_ok=True)
        os.makedirs(os.path.join(self.work_dir, "crops"), exist_ok=True)
