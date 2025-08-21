import os
import sys
import torch
import yaml
import math
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def load_config(config_path="config.yaml"):
    """YAML 설정 파일을 불러옵니다."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_and_processor(model_id, device):
    """Hugging Face에서 모델과 프로세서를 불러옵니다."""
    print(f"Loading model '{model_id}'...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print(f"Model loaded on device: {device}")
    return model, processor

def draw_boxes(image, boxes, labels, color_map, width=5):
    """PIL을 사용하여 이미지에 바운딩 박스와 라벨을 그립니다."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label_with_score in zip(boxes, labels):
        # 라벨에서 순수 클래스 이름만 추출 (e.g., "person 0.87" -> "person")
        clean_label = label_with_score.split(' ')[0]
        color = color_map.get(clean_label, "red") # color_map에 없으면 기본값 'red'

        draw.rectangle(box.tolist() if isinstance(box, torch.Tensor) else box, outline=color, width=width)
        text_bbox = draw.textbbox((box[0], box[1]), label_with_score, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((box[0], box[1]), label_with_score, fill="white", font=font)
    return image

def calculate_distance(box1, box2):
    """두 바운딩 박스의 중심점 사이의 유클리드 거리를 계산합니다."""
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

def merge_boxes(box1, box2):
    """두 바운딩 박스를 포함하는 가장 작은 바운딩 박스를 반환합니다."""
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    return torch.tensor([x1, y1, x2, y2])

def calculate_iou(box1, box2):
    """두 바운딩 박스의 Intersection over Union (IoU)을 계산합니다."""
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = float(box1_area + box2_area - intersection_area)
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def find_and_merge_nearby_boxes(boxes, labels, scores, merge_options):
    """(부분 문자열 일치) 올바른 로직으로 근처 박스를 찾아 병합합니다."""
    if not merge_options.get('enabled', False):
        return boxes.tolist(), labels, scores.tolist(), False

    class_a = merge_options['class_a']
    class_b = merge_options['class_b']
    merged_label_str = merge_options['merged_label']
    threshold = merge_options['proximity_threshold']

    detections_a = [(b, s) for b, l, s in zip(boxes, labels, scores) if class_a.startswith(l)]
    detections_b = [(b, s) for b, l, s in zip(boxes, labels, scores) if class_b.startswith(l)]

    merged_detections = []
    used_a_indices = set()
    
    for i, (box_b, score_b) in enumerate(detections_b):
        best_match_a_idx = -1
        min_dist = float('inf')

        for j, (box_a, _) in enumerate(detections_a):
            if j in used_a_indices:
                continue
            
            dist = calculate_distance(box_b, box_a)
            iou = calculate_iou(box_a, box_b)
            
            if dist < threshold or iou > 0:
                if dist < min_dist:
                    min_dist = dist
                    best_match_a_idx = j
        
        if best_match_a_idx != -1:
            box_a, score_a = detections_a[best_match_a_idx]
            merged_box = merge_boxes(box_a, box_b)
            merged_score = (score_a + score_b) / 2
            merged_detections.append((merged_box, merged_label_str, merged_score))
            used_a_indices.add(best_match_a_idx)

    if merged_detections:
        final_boxes, final_labels, final_scores = zip(*merged_detections)
        return list(final_boxes), list(final_labels), [s.item() for s in final_scores], True
    else:
        return boxes.tolist(), labels, scores.tolist(), False

def process_image(image_path, target_path, model, processor, box_threshold, merge_options, color_map, device):
    """단일 이미지를 처리하고, 객체를 탐지하며, 결과를 저장합니다."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}. Skipping.")
        return

    all_classes = merge_options.get('classes', [merge_options.get('class_a'), merge_options.get('class_b')])
    text_prompt = " . ".join(list(set(c for c in all_classes if c)))
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]]
    )

    result = results[0]
    boxes = result["boxes"]
    scores = result["scores"]
    unfiltered_labels = result["labels"]

    mask = scores > box_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = [label for label, keep in zip(unfiltered_labels, mask) if keep]

    final_boxes, final_labels, final_scores, merged_happened = find_and_merge_nearby_boxes(boxes, labels, scores, merge_options)
    
    if merged_happened:
        print(f"  -> Merged boxes for '{merge_options['merged_label']}' in {os.path.basename(image_path)}")
        dir_path, original_filename = os.path.split(target_path)
        new_filename = f"merged_{original_filename}"
        target_path = os.path.join(dir_path, new_filename)

    if len(final_boxes) > 0:
        labels_with_scores = [
            f"{label} {score:.2f}"
            for score, label in zip(final_scores, final_labels)
        ]
        annotated_image = draw_boxes(image.copy(), final_boxes, labels_with_scores, color_map)
    else:
        annotated_image = image.copy()

    annotated_image.save(target_path)
    print(f"Processed {os.path.basename(image_path)} -> {target_path}")

def main():
    """메인 실행 함수."""
    config = load_config()
    source_dirs = config['paths']['source_image_dirs']
    target_base_dir = config['paths']['target_base_dir']
    model_id = config['model']['model_id']
    detection_opts = config['detection']
    box_threshold = detection_opts['box_threshold']
    merge_opts = detection_opts.get('merge_options', {'enabled': False})
    
    # --- 색상 설정 및 검증 ---
    classes = detection_opts.get('classes', [])
    colors = detection_opts.get('colors', [])
    if len(classes) != len(colors):
        print("Error: The number of 'classes' and 'colors' in config.yaml must be the same.")
        sys.exit(1) # 오류 발생 시 스크립트 종료
        
    color_map = {cls: color for cls, color in zip(classes, colors)}
    if merge_opts.get('enabled'):
        color_map[merge_opts['merged_label']] = merge_opts.get('merged_color', 'red')
        
    # 병합 로직을 위한 클래스 설정
    if merge_opts.get('enabled'):
        merge_opts['classes'] = list(set([merge_opts.get('class_a'), merge_opts.get('class_b')]))
    else:
        merge_opts['classes'] = classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model_and_processor(model_id, device)

    for source_dir in source_dirs:
        if not os.path.isdir(source_dir):
            print(f"Source directory: {source_dir} does not exist.")
            continue

        folder_name = os.path.basename(os.path.abspath(source_dir))
        target_dir = os.path.join(target_base_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"--- Processing images in: {source_dir} (results will be in {target_dir}) ---")
        image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

        if not image_files:
            print(f"Warning: No images found in '{source_dir}'.")
            continue

        for image_name in image_files:
            source_path = os.path.join(source_dir, image_name)
            target_path = os.path.join(target_dir, image_name)
            process_image(source_path, target_path, model, processor, box_threshold, merge_opts, color_map, device)
        
        print(f"--- Finished processing: {source_dir} ---")

    print("\n--- All image processing complete ---")

if __name__ == "__main__":
    main()