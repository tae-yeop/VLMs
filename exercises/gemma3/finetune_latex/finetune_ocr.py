from PIL import Image
from datasets import load_dataset
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
import requests
from PIL import Image

instruction = 'Convert the equation images to LaTeX equations.'


def convert_to_conversation(sample):
    """
    학습을 위한 message dict 형태로. message 하나에 list 하나.
    """
    conversation = [
        # User Prompt : 유저의 질문
        { 'role': 'user',
          'content' : [
            {'type' : 'text',  'text'  : instruction},
            {'type' : 'image', 'image' : sample['image']} ]
        },

        # Assistant Response : 모델이 하는 리턴
        { 'role' : 'assistant',
          'content' : [
            {'type' : 'text',  'text'  : sample['text']} ]
        },
    ]
    return { 'messages' : conversation }


def process_vision_info(messages: list[dict]) -> list[Image.Image]:
    """
    데이터셋 이미지를 gemma3에서 처리할 수 있게 RGB 포맷으로 변경
    """
    image_inputs = []
    for msg in messages: # list of dict를 순회
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]
 
        for element in content: # 한 dict (msg)안에서 다시 content를 살펴서 image가 있으면 이를 RGB로 변환해서 append시킴
            if isinstance(element, dict) and ("image" in element or element.get("type") == "image"):
                image = element["image"]
                image_inputs.append(image.convert("RGB"))
    return image_inputs


if __name__ == "__main__":
    dataset_train = load_dataset('unsloth/LaTeX_OCR', split='train[:3000]')

    # inspect_data()

    print(type(dataset_train))

    # 오히려 이게 더 빠름
    # 
    train_dataset = [convert_to_conversation(sample) for sample in tqdm(dataset_train, total=len(dataset_train))]

    print(type(train_dataset)) # list
    print(train_dataset[0])


    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)
        

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]

        # Mask tokens for not being used in the loss computation
        # padding/image tokens은 loss 계산에 참여하지 않게 하기 위해서
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100
 
        batch["labels"] = labels
        return batch

    model_id = "google/gemma-3-4b-pt"  # or `google/gemma-3-12b-pt`, `google/gemma-3-27-pt`

    model_kwargs = dict(
        attn_implementation="flash_attention_2",  # more efficient attention mechanism for Ampere GPU
        torch_dtype=torch.bfloat16,  
        device_map="auto",  # Let torch decide how to load the model
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    print(model_kwargs["quantization_config"])

    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    print(processor)

    # W = W_0 + \frac{lora_alpha}{r} BA
    peft_config = LoraConfig(
        lora_alpha=16, # 계수 조절값
        lora_dropout=0,
        r=8, # rank of the low-rank adapters
        bias="none", # np bias
        target_modules=[
        'down_proj',
        'o_proj',
        'k_proj',
        'q_proj',
        'gate_proj',
        'up_proj',
        'v_proj'],
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
        )
    
    args=SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=1, # For full training runs over the dataset.
        learning_rate=2e-4,
        bf16=True,
        logging_steps=200,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=2,
        optim='adamw_8bit',
        weight_decay=0.01,
        lr_scheduler_type='linear',
        seed=3407,
        output_dir='outputs',
        report_to='none',    
        remove_unused_columns=False,
        dataset_text_field='',
        dataset_kwargs={'skip_prepare_dataset': True},
        max_seq_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=converted_dataset_train,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )

    trainer.train()

    trainer.save_model()

    del model
    del trainer
    torch.cuda.empty_cache()


    model = AutoModelForImageTextToText.from_pretrained(
        args.output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(args.output_dir)

    image = Image.open("/tpath/to/the/image/file").convert("RGB")
    instruction = 'Convert the equation images to LaTeX equations.'

    def generate_equation(model, processor):
        messages = [
            {
                'role': 'user',
                'content' : [
                    {'type' : 'text', 'text' : instruction},
                    {'type' : 'image', 'image' : image}
                ]
            },
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process the image and text
        image_inputs = process_vision_info(messages)

        # Tokenize the text and process the images
        inputs = processor(
            text = [text],
            images = image_inputs,
            padding=True,
            return_tensors="put"
        )

        inputs = inputs.to(model.device)

        stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            top_p=1.0,
            do_sample=True,
            temperature=0.8,
            eos_token_id=stop_token_ids,
            disable_compile=True
        )

        # Trim the generation and decode the output to text
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids for zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_toknes=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    equation = generate_equation(model, processor)
    print(equation)