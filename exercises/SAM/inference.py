import torch, numpy as np, cv2
from PIL import Image
from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers import SamProcessor, SamModel

img_path = "market.png"  # 마트 이미지
image = Image.open(img_path).convert("RGB")
W, H = image.size

# 1) 오픈어휘 박스 탐지 (Grounding DINO)
# 더 구체적이고 효과적인 텍스트 프롬프트
text_prompt = "person . wheeled shopping trolley ."
proc_det = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")   # 빠름
det_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()

inputs = proc_det(images=image, text=text_prompt, return_tensors="pt")
with torch.no_grad():
    det_out = det_model(**inputs)

# post-process → (x1,y1,x2,y2) in pixels
targets = torch.tensor([[H, W]])
results = proc_det.post_process_grounded_object_detection(
    det_out, inputs["input_ids"],
    box_threshold=0.15, text_threshold=0.15,  # threshold 높여서 깔끔하게
    target_sizes=targets
)[0]

boxes = results["boxes"]               # (N,4), float32
labels = results["labels"]             # list[str]
scores = results["scores"]             # (N,)

# 원본 탐지 결과 출력 (디버깅용)
print(f"🔍 원본 탐지 결과:")
for i, (label, score) in enumerate(zip(labels, scores)):
    print(f"  {i}: '{label}' (신뢰도: {score:.3f})")

# 사람/카트만 남기기 (grounding-dino-base의 복합 라벨에 맞춰 포함 관계로 확인)
target_keywords = ["person", "wheeled", "shopping", "trolley"]
keep = [i for i,l in enumerate(labels) if any(keyword in l.lower() for keyword in target_keywords) and scores[i] > 0.30]  # base 모델은 신뢰도 적당히
boxes = boxes[keep]; labels = [labels[i] for i in keep]; scores = scores[keep]

print(f"📋 필터링 후 결과:")
for i, (label, score) in enumerate(zip(labels, scores)):
    print(f"  {i}: '{label}' (신뢰도: {score:.3f})")

# 중복 제거를 위한 간단한 NMS 적용
def simple_nms(boxes, scores, labels, iou_threshold=0.5):
    """간단한 NMS 구현"""
    if len(boxes) == 0:
        return [], [], []
    
    # 신뢰도 순으로 정렬
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current.item())
        
        if len(indices) == 1:
            break
            
        # 현재 박스와 나머지 박스들의 IoU 계산
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        # IoU 계산 (간단 버전)
        iou_scores = []
        for other_box in other_boxes:
            x1 = max(current_box[0], other_box[0])
            y1 = max(current_box[1], other_box[1])
            x2 = min(current_box[2], other_box[2])
            y2 = min(current_box[3], other_box[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                area2 = (other_box[2] - other_box[0]) * (other_box[3] - other_box[1])
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0
            else:
                iou = 0
            iou_scores.append(iou)
        
        # IoU가 threshold보다 낮은 박스들만 남김
        iou_scores = torch.tensor(iou_scores)
        indices = indices[1:][iou_scores < iou_threshold]
    
    return [boxes[i] for i in keep], [scores[i] for i in keep], [labels[i] for i in keep]

# NMS 적용
if len(boxes) > 0:
    boxes_nms, scores_nms, labels_nms = simple_nms(boxes, scores, labels, iou_threshold=0.3)
    boxes = torch.stack(boxes_nms) if boxes_nms else torch.empty(0, 4)
    scores = torch.stack(scores_nms) if scores_nms else torch.empty(0)
    labels = labels_nms
    
    print(f"🎯 NMS 적용 후 최종 결과:")
    for i, (label, score) in enumerate(zip(labels, scores)):
        print(f"  {i}: '{label}' (신뢰도: {score:.3f})")

# 검출된 박스가 없으면 종료
if len(boxes) == 0:
    print("검출된 객체가 없습니다.")
    exit()

# 2) 박스 → SAM 마스크
sam_proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")   # huge도 가능하나 VRAM↑
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").eval()

# SAM은 [B, num_boxes, 4] 모양의 box(픽셀 단위, XYXY)를 기대
# boxes 텐서를 리스트 형태로 변환
boxes_list = boxes.tolist()
inputs_sam = sam_proc(image, input_boxes=[boxes_list], return_tensors="pt")
with torch.no_grad():
    sam_out = sam_model(**inputs_sam)

masks = sam_proc.post_process_masks(
    sam_out.pred_masks,        # [B, num_boxes, 1, H/4, W/4]
    inputs_sam["original_sizes"],
    inputs_sam["reshaped_input_sizes"]
)[0]                            # → [num_boxes, H, W] bool/float

# 3) 시각화
img = cv2.imread(img_path)[:, :, ::-1]
overlay = img.copy()
print(f"이미지 크기: {img.shape}")
print(f"마스크 개수: {len(masks)}")

for i, m in enumerate(masks):
    print(f"마스크 {i} 원본 크기: {m.shape}")
    
    # PyTorch 텐서를 numpy로 변환
    if hasattr(m, 'cpu'):
        mask_tensor = m.cpu().numpy()
    else:
        mask_tensor = np.array(m)
    
    # 3차원 마스크인 경우 첫 번째 채널만 사용하거나 평균 취함
    if len(mask_tensor.shape) == 3:
        if mask_tensor.shape[0] == 3:  # [3, H, W] 형태
            mask = mask_tensor[0]  # 첫 번째 채널 사용
        else:  # [H, W, 3] 형태
            mask = mask_tensor.mean(axis=2)  # 평균 취함
    else:
        mask = mask_tensor.squeeze()
    
    # boolean 마스크로 변환
    mask = mask > 0.5
    print(f"처리된 마스크 {i} 크기: {mask.shape}")
    
    # 마스크와 이미지 크기가 다르면 리사이즈
    if mask.shape != overlay.shape[:2]:
        print(f"리사이즈: {mask.shape} -> {overlay.shape[:2]}")
        mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # 마켓 이미지 객체별로 다른 색상 지정
    label = labels[i].lower()
    if "person" in label:
        color = np.array([0, 255, 0], dtype=np.uint8)  # 녹색 (RGB)
    elif any(cart_word in label for cart_word in ["wheeled", "shopping", "cart"]):
        color = np.array([0, 0, 255], dtype=np.uint8)  # 파란색 (RGB)
    else:
        color = np.array([255, 0, 0], dtype=np.uint8)  # 기본 빨간색 (RGB)
    
    print(f"라벨 '{labels[i]}' -> 색상: {color}")
    
    # 마스크가 True인 픽셀에만 색상 적용
    for c in range(3):  # RGB 각 채널별로 처리
        overlay[mask, c] = (0.5 * overlay[mask, c] + 0.5 * color[c]).astype(np.uint8)
    
    # 라벨 그리기 (색상도 객체별로 맞춤)
    x1,y1,x2,y2 = boxes[i].tolist()
    # OpenCV는 BGR 순서를 사용하므로 RGB -> BGR로 변환
    color_bgr = (int(color[2]), int(color[1]), int(color[0]))
    cv2.rectangle(overlay, (int(x1),int(y1)), (int(x2),int(y2)), color_bgr, 2)
    cv2.putText(overlay, labels[i], (int(x1), max(0,int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

out = (0.6*img + 0.4*overlay).astype(np.uint8)
cv2.imwrite("seg_groundingdino_sam.png", out[:, :, ::-1])
