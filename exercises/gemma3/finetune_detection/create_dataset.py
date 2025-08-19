from datasets import load_dataset
import argparse
from config import Configuration
from huggingface_hub import login, create_repo

def coco_to_xyxy(coco_bbox):
    x, y, width, height = coco_bbox
    x1, y1 = x, y
    x2, y2 = x + width, y + height
    return [x1, y1, x2, y2]

def convert_to_detection_string(bboxs, image_width, image_height):
    def format_location(value, max_value):
        return f"<loc{int(round(value * 1024 / max_value)):04}>"
    
    detection_strings = []
    for bbox in bboxs:
        x1, y1, x2, y2 = coco_to_xyxy(bbox)
        name = "plate"
        locs = [
            format_location(y1, image_height),
            format_location(x1, image_width),
            format_location(y2, image_height),
            format_location(x2, image_width),
        ]

        detection_string = "".join(locs) + f" {name}"
        detection_strings.append(detection_string)

    return " ; ".join(detection_strings)

def format_objects(example):
    height = example["height"]
    width = example["width"]
    bboxs = example["objects"]["bbox"]
    formatted_objects = convert_to_detection_string(bboxs, width, height)
    return {"label_for_paligemma": formatted_objects}


if __name__ == "__main__":
    # Support for generic script for dataset
    cfg = Configuration()
    parser = argparse.ArgumentParser(description='Process dataset for PaLiGemma')
    parser.add_argument('--dataset', type=str, default=cfg.dataset_id, help='Hugging Face dataset ID')
    parser.add_argument("--config", default=None)
    parser.add_argument('--output_repo', type=str, required=True, help='Output repository ID for Hugging Face Hub')
    parser.add_argument("--hf_token", help="HF access token (optional, ""fallback to HF_TOKEN env var)")
    args = parser.parse_args()

    if args.hf_token:
        login(token=args.hf_token)

    create_repo(args.output_repo, repo_type="dataset", exist_ok=True)
    # load the dataset
    print(f"[INFO] Loading {args.dataset} from hub...")
    dataset = load_dataset(args.dataset, args.config) if args.config else load_dataset(args.dataset)

    for split in dataset.keys():
        print(f"[INFO] Processing split: {split}")
        dataset[split] = dataset[split].map(format_objects)

    # push to hub
    dataset.push_to_hub(args.output_repo) # max_shard_size="500MB"