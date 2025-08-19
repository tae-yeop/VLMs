from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

if __name__ == '__main__':
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    ).to("cuda")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": "path_to_video.mp4"},
                {"type": "text", "text": "Describe this video in detail"}
            ]
        },
    ]

    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "What are the differences between these two images?"},
    #         {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
    #         {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"},            
    #         ]
    #     },
    # ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print(generated_texts[0])

