from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")  # 또는 yolov8n.pt (COCO person=0)
cap = cv2.VideoCapture("/purestorage/AILAB/AI_1/tyk/3_CUProjects/07.지능형_관제_서비스_CCTV_영상_데이터/3.개방데이터/1.데이터/Validation/01.원천데이터/E04_064.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("street_person_bbox.mp4", cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))

while True:
    ok, frame = cap.read()
    if not ok: break
    results = model(frame, classes=[0])  # person만
    r = results[0]
    if r.boxes is not None and len(r.boxes):
        for (x1,y1,x2,y2), conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"person {conf:.2f}", (x1, max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    out.write(frame)

cap.release(); out.release()