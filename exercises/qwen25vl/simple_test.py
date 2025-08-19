import os
import hashlib
import requests
import tqdm
from PIL import Image
import numpy as np

import torch
import cv2
from decord import VideoReader, cpu

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def inference_recognition(sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):

    image_path = "/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/qwen25vl/Qwen2.5-VL/cookbooks/assets/universal_recognition/unireco_bird_example.jpg"
    prompt = "What kind of bird is this? Please give its name in Chinese and English."

    image = Image.open(image_path)
    image.thumbnail([640,640], Image.Resampling.LANCZOS)



    image = Image.open(image_path)
    image_local_path = "file://" + image_path
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"image": image_local_path},
            ]
        },
    ]

    # 멀티턴 대화형식 (instructional message) 포맷 리턴
    # placeholder |image_pad|가 추가됨
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)

    # text: <|im_start|>system
    # You are a helpful assistant.<|im_end|>
    # <|im_start|>user
    # What kind of bird is this? Please give its name in Chinese and English.<|vision_start|><|image_pad|><|vision_end|><|im_end|>
    # <|im_start|>assistant


    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    print("inputs:", inputs, inputs.keys())

    print("inputs.input_ids:", inputs.input_ids.shape) # torch.Size([1, 1367])
    # 전체 패치 수 : H x W = 70 x 76 = 5320, vision tokens : 1176
    print("inputs.pixel_values:", inputs.pixel_values.shape) # torch.Size([5320, 1176]

    # grid 갯수, H 방향 쪼갠 패치수 = 70, W 방향 쪼갠 패치수 = 76
    print("inputs.image_grid_thw:", inputs.image_grid_thw) # tensor([[ 1, 70, 76]

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if return_input:
        return output_text[0], inputs
    else:
        return output_text[0]
    

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found

    return json_output
    
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor
import ast
additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
def plot_bounding_boxes(im, bounding_boxes, input_width, input_height):
    img = im
    width, height = img.size
    print(img.size)

    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

    font = ImageFont.truetype("NotoSansCJK-Regular.ttc", size=14)

    try:
        json_output = ast.literal_eval(bounding_boxes)
    except Exception as e:
        end_idx = bounding_boxes.rfind('"}') + len('"}')
        truncated_text = bounding_boxes[:end_idx] + "]"
        json_output = ast.literal_eval(truncated_text)

    # Iterate over the bounding boxes
    for i, bounding_box in enumerate(json_output):
        # Select a color from the list
        color = colors[i % len(colors)]

        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["bbox_2d"][1]/input_height * height)
        abs_x1 = int(bounding_box["bbox_2d"][0]/input_width * width)
        abs_y2 = int(bounding_box["bbox_2d"][3]/input_height * height)
        abs_x2 = int(bounding_box["bbox_2d"][2]/input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        # Draw the bounding box
        draw.rectangle(
            ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
        )

        # Draw the text
        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    # Display the image
    img.save("output_image.png")


def inference_bbox():
    system_prompt="You are a helpful assistant"
    max_new_tokens=1024
    # image_path = '/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/qwen25vl/Qwen2.5-VL/cookbooks/assets/spatial_understanding/cakes.png'
    # image = Image.open(image_path)
    # prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
    # messages = [
    #     {
    #     "role": "system",
    #     "content": system_prompt
    #     },
    #     {
    #     "role": "user",
    #     "content": [
    #         {
    #         "type": "text",
    #         "text": prompt
    #         },
    #         {
    #         "image": image_path
    #         }
    #     ]
    #     }
    # ]


    query_image_path = '/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/qwen25vl/query.jpg'
    key_image_path = '/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/qwen25vl/key2.png'
    query_image = Image.open(query_image_path)
    key_image = Image.open(key_image_path)
    prompt = "Outline the position of each small cake and output all the coordinates in JSON format."
    messages = [
        {
        "role": "system",
        "content": system_prompt
        },
        {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please find the person in the first image within the second image. Output the bounding box coordinates in JSON format. First find the bounding box of all the people in the second image, then find the person in the first image within the second image. The output should be a list of bounding boxes in the format: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'person'}]."
            },
            {
                "type": "image",
                "image": "file://" + query_image_path
            },
            {
                "type": "image",
                "image": "file://" + key_image_path
            }
        ]
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image_inputs, video_inputs = process_vision_info(messages)

    print("input:\n",text)
    inputs = processor(
        text=[text],
        images=image_inputs, 
        videos=video_inputs, 
        padding=True, 
        return_tensors="pt").to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print("output:\n",output_text[0])

    input_height = inputs['image_grid_thw'][0][1]*14
    input_width = inputs['image_grid_thw'][0][2]*14

    return output_text[0], input_height, input_width

if __name__ == "__main__":
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct",trust_remote_code=True)

    ## Use a local HuggingFace model to inference.
    # response = inference_recognition()
    response, input_height, input_width = inference_bbox()
    print('response', response)

    # image_path = '/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/qwen25vl/Qwen2.5-VL/cookbooks/assets/spatial_understanding/cakes.png'
    # image = Image.open(image_path)
    # print(image.size)
    # image.thumbnail([640,640], Image.Resampling.LANCZOS)
    # plot_bounding_boxes(image,response,input_width,input_height)


    image_path = '/purestorage/AILAB/AI_1/tyk/3_CUProjects/MM/VLM/qwen25vl/key2.png'
    image = Image.open(image_path)
    print(image.size)
    image.thumbnail([640,640], Image.Resampling.LANCZOS)
    plot_bounding_boxes(image,response,input_width,input_height)
