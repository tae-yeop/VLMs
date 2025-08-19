from typing import Dict, List, Tuple, Optional, Sequence
import json
import copy

import torch
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText, Qwen2VLImageProcessor, Trainer, BitsAndBytesConfig, HfArgumentParser
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from datasets import load_dataset, Image
try:
    from .arguments import CustomArguments  # when executed as part of the package
except Exception:
    try:
        from qwen25vl.crowd_counting.arguments import CustomArguments  # absolute import fallback
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from arguments import CustomArguments  # local directory fallback

# Support running both as a package module and as a standalone script
try:
    from .rope2d import get_rope_index_25  # when executed as part of the package
except Exception:
    try:
        from qwen25vl.crowd_counting.rope2d import get_rope_index_25  # absolute import fallback
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from rope2d import get_rope_index_25  # local directory fallback

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


def collate_fn(batch):
    messages = [m['messages'] for m in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)


def flatten_mm_content_to_text_and_paths(
    message: Dict,
    image_token: str = "<image>",
    video_token: str = "<video>"
) -> Tuple[str, List[str], List[str]]:
    """
    Converts a message like:
      {'role': 'user',
       'content': [{'type':'text','text':'...'}, {'type':'image','image':'/path/to.jpg'}, ...]}
    into:
      ("...<image>...", ["/path/to.jpg", ...], ["/path/to.mp4", ...])
    preserving order.
    """
    parts: List[str] = []
    image_paths: List[str] = []
    video_paths: List[str] = []

    for c in message.get("content", []):
        ctype = c.get("type")
        if ctype == "text":
            parts.append(c.get("text", ""))
        elif ctype == "image":
            image_paths.append(c.get("image"))
            parts.append(image_token)
        elif ctype == "video":
            video_paths.append(c.get("video"))
            parts.append(video_token)
        else:
            # ignore unknown types
            continue

    return ("".join(parts), image_paths, video_paths)

def to_qwen_style_pair(
    user_msg: Dict,
    assistant_msg: Dict
) -> Dict:
    """
    Returns a structure convenient for your dataset:
      {
        "messages": [
          {"role": "user", "content": "<text with <image> markers>"},
          {"role": "assistant", "content": "<assistant text (kept as-is)>"}
        ],
        "images": [...],
        "videos": [...]
      }
    Also validates assistant JSON if present.
    """
    user_text, images, videos = flatten_mm_content_to_text_and_paths(user_msg)

    # assistant content is usually [{'type': 'text', 'text': '...json...'}]
    assistant_text: Optional[str] = None
    for c in assistant_msg.get("content", []):
        if c.get("type") == "text":
            assistant_text = c.get("text")
            break

    # Optional: validate/normalize assistant JSON
    if assistant_text:
        try:
            _parsed = json.loads(assistant_text)
            # normalize spacing by dumping back (optional)
            assistant_text = json.dumps(_parsed, separators=(",", ":"))
        except json.JSONDecodeError:
            # keep raw if it's not valid JSON
            pass

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text or ""}
        ],
        "images": images,
        "videos": videos
    }

class CrowdCountingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, image_processor, data_args):
        super().__init__()

        self.tokenizer = tokenizer
        self.processor = image_processor
        self.data_args = data_args
        self.hf_dataset = hf_dataset
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.get_rope_index = get_rope_index_25


    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self,idx):
        sources = self.hf_dataset[idx]

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        image, grid_thw = self.process_image_unified(sources["image"])
        image = [image]

        # Compute total number of visual tokens required for this image
        # Sum over all grids/scales instead of using only the first entry
        if isinstance(grid_thw, torch.Tensor):
            per_grid_iter = list(grid_thw)
        elif isinstance(grid_thw, Sequence):
            per_grid_iter = list(grid_thw)
        else:
            per_grid_iter = [grid_thw]

        merge_size = int(self.processor.merge_size)
        total_image_tokens = 0
        for thw in per_grid_iter:
            if torch.is_tensor(thw):
                t = int(thw[0].item()) if thw.numel() >= 3 else 1
                h = int(thw[1].item()) if thw.numel() >= 2 else int(thw[0].item())
                w = int(thw[2].item()) if thw.numel() >= 3 else 1
            else:
                # Expect sequence length >= 3: (T, H, W)
                t = int(thw[0]) if len(thw) >= 3 else 1
                h = int(thw[1]) if len(thw) >= 2 else int(thw[0])
                w = int(thw[2]) if len(thw) >= 3 else 1
            tokens_scale = t * (h // merge_size) * (w // merge_size)
            total_image_tokens += tokens_scale

        # Pass a single summed token count per <image> marker
        grid_thw_merged = [int(total_image_tokens)]

        chat_sources = copy.deepcopy([sources['messages']])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )

        # Prepare image/video grid tensors without ambiguous tensor truthiness checks
        if isinstance(grid_thw, torch.Tensor):
            image_grid_thw_tensor = grid_thw
        elif isinstance(grid_thw, Sequence) and len(grid_thw) > 0:
            image_grid_thw_tensor = torch.stack(grid_thw, dim=0)
        else:
            image_grid_thw_tensor = None

        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw_tensor = video_grid_thw
        elif isinstance(video_grid_thw, Sequence) and len(video_grid_thw) > 0:
            video_grid_thw_tensor = torch.stack(video_grid_thw, dim=0)
        else:
            video_grid_thw_tensor = None

        position_ids, _ = self.get_rope_index(
            self.processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=image_grid_thw_tensor,
            video_grid_thw=video_grid_thw_tensor,
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        # attention_mask will be rebuilt in the collator from input_ids; avoid incorrect shapes here
        data_dict["pixel_values"] = torch.cat(image, dim=0)
        # Use the prepared grid tensors directly to avoid shape/type ambiguity
        data_dict["image_grid_thw"] = image_grid_thw_tensor

        return data_dict



    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.processor)
        image = image_file.convert("RGB")
        # Enforce pixel caps to control number of visual tokens
        try:
            processor.max_pixels = getattr(self.data_args, "max_pixels", processor.max_pixels)
            processor.min_pixels = getattr(self.data_args, "min_pixels", processor.min_pixels)
            if hasattr(processor, "size") and isinstance(processor.size, dict):
                processor.size["longest_edge"] = processor.max_pixels
                processor.size["shortest_edge"] = processor.min_pixels
        except Exception:
            pass

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts
    

from dataclasses import dataclass, field
@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


if __name__ == '__main__':
    parser = HfArgumentParser(CustomArguments)
    (cfg,) = parser.parse_args_into_dataclasses()


    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg.attn_implementation,
            )

    tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            model_max_length=cfg.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    image_processor = processor.image_processor


    train_dataset = load_dataset(
        "ty-kim/total_crowd_count2", 
        split="train",
        download_mode="force_redownload")


    train_dataset = train_dataset.cast_column("image_path", Image(decode=True))
    train_dataset = train_dataset.rename_column("image_path", "image")

    # Keep images as PIL for the Qwen processor; only convert numeric fields
    def _tf(ex):
        # ex["image"] stays as PIL.Image
        ex["points"] = torch.tensor(ex["points"], dtype=torch.float32)
        return ex

    train_dataset.set_transform(_tf)
    # Keep 'image' column in outputs while formatting only tensor fields
    train_dataset.set_format(type="torch", columns=["points", "counts"], output_all_columns=True)

    train_dataset = CrowdCountingDataset(
            train_dataset,
            tokenizer,
            image_processor,
            cfg
        )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            collate_fn=data_collator,
        )

    for batch in train_loader:
        print(batch)
        break