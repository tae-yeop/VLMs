# trainer.save_model : 모델 가중치 (state_dict) 저장
# trainer.save_state : 옵티마이저, 스케줄러, RNG 등 훈련상태까지포함 전체체크포인트 저장

# save_model 주의점
# save_model은 파라미터를 GPU 메모리 그대로 state_dict에 복사해서 보음 -> GPU VRAM 부족
# DeepSpeed, ZeRO
# ZeRo-3 파라미터 GPU 마다 샤딩 -> save_model() 샤딩 해제를 못해 깨진 체크포인트가 나옴
# rank마다 중복 저장
# 분산 훈련 중복 저장
# 모든 rank가 동시에 save_mdeol을 호출 -> 동일한 파일을 여러 번 덮어쓰거나 병목현상

import json
import os
from PIL import Image, ImageDraw
from glob import glob
import yaml

train_image_path = '/purestorage/AILAB/AI_1/dataset/Objects365/images/train'
val_image_path = '/purestorage/AILAB/AI_1/dataset/Objects365/images/val'
train_label_path = '/purestorage/AILAB/AI_1/dataset/Objects365/labels/train'
val_label_path = '/purestorage/AILAB/AI_1/dataset/Objects365/labels/val'

with open('/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLM/obj365.yaml', 'r') as f:
    config = yaml.safe_load(f)


def make_json():
    label_trains = os.listdir('/purestorage/AILAB/AI_1/dataset/Objects365/labels/train')
    # print(label_trains)

    label_files = []
    file_names = []
    class_set = set()
    cnt = 0
    messages = []
    for f in label_trains:
        if not f.endswith(".txt"):
            continue
    
        # label_files.append(f)
        file_name = f.split('.')[0]
        # file_names.append(file_name)

        img_path = os.path.join(train_image_path, f"{file_name}.jpg")
        image = Image.open(img_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size

        with open(os.path.join(train_label_path, f), "r") as label_file:
            lines = label_file.readlines()

        response_list = []
        class_set = set()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, box_width, box_height = map(float, parts)
            class_name = config['names'][int(class_id)]

            # 정규화 좌표 → 픽셀 좌표로 변환
            x_center *= width
            y_center *= height
            box_width *= width
            box_height *= height

            x0 = int(max(x_center - box_width / 2, 0))
            y0 = int(max(y_center - box_height / 2, 0))
            x1 = int(min(x_center + box_width / 2, width))
            y1 = int(min(y_center + box_height / 2, height))


            # 박스 그리기 (굵기 강조)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)

            # class_id도 텍스트로 표시
            draw.text((x0 + 2, y0 + 2), str(class_name), fill="yellow")

            class_set.add(class_name)

            response_list.append({
                "bbox_2d": [x0, y0, x1, y1],
                "label": class_name
            })
            # response += f'{{"bbox_2d": [{x0}, {y0}, {x1}, {y1}], "label": "{class_name}"}}'

        prompt = "Outline the position of each"
        for obj in class_set:
            prompt += f" {obj},"

        prompt = prompt.rstrip(',') + " and output all the coordinates in JSON format."

        message = {
            "image": img_path,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": json.dumps(response_list, ensure_ascii=False)}
            ]
        }

        # message.update({"image": img_path})
        # message.update({"conversations": [{"from": "human", "value": prompt},  {"from":"gpt", "value":response}]})

        print(f'{cnt}-----------')
        print(message)

        messages.append(message)

        image.save(f'output_{cnt}.png')
        print(class_set)
        print('-----------')
        cnt += 1
        if cnt > 10:
            break
        
    with open("messages_output.json", "w", encoding="utf-8") as outfile:
        json.dump(messages, outfile, indent=2, ensure_ascii=False)

    # print(len(label_files))

    # print(label_files[:10])
    # print(file_names[:10])

    

    



if __name__ == '__main__':
    make_json()