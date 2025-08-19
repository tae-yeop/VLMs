## Gemma 특징

- Multilingual support: Out-of-the-box support for 35 languages and pre-trained functionality for over 140 languages.

- Long context window: Comes with 128k-token context window.

- Multimodal capabilities: Supports advanced text, image, and short video reasoning.

- Structured output: Built-in support for function calling.

- Quantized models: Official quantized versions are available, reducing model size and computational requirements.

- Hardware integration: Works with CPUs, NVIDIA GPUs, Google Cloud TPUs, and AMD GPUs.
Integration with popular tools: Works with tools like Hugging Face Transformers, PyTorch, Keras, JAX, Google AI Edge, vLLM, and more.

- ShieldGemma 2: It provides image safety checks for dangerous content, explicit material, and violence, ensuring responsible AI development.

- exhibit better numerical stability and performance when using bfloat16 (bf16) rather than float16 (fp16) because they were trained using TPUs with native BF16 support

## 실행 정보

**1B에는 비전 인코더가 없는 순수 언어 모델**

| Model | CPU Cores | GPU VRAM (Text-to-Text) | GPU VRAM (Image-to-Text) | Recommended Hardware |
|---|---|---|---|---|
| 1B | 16 | 2.3 GB | N/A | GTX 1650 4 GB |
| 4B | 16-32 | 9.2 GB | 10.4 GB | RTX 3060 12 GB |
| 12B | 32 | 27.6 GB | 31.2 GB | RTX 5090 32 GB |
| 27B | 32-64 | 6.1 GB | 70.2 GB | RTX 4090 24 GB (x3) |



## Architectural Components of Gemma 3

### 기본

GQA + RoPE

attention은 local과 global을 혼합해서 사용함. 5개 레이어 중에서 매번 4번째 레이어는 local sliding window를 사용하고 마지막 5번째는 Full global attention 사용함. 이를 통해 연산량과 KV 캐시의 메모리량을 줄임.

### SigLIP 비전 인코더

![Image](https://learnopencv.com/wp-content/uploads/2025/04/SigLIP-Vision-Encoder.png)

학습 중에 고정. 고해상도 (896px)일 때 성능이 좋음. 다운 샘플링해도 디테일을 유지할 수 있어서. siglip의 입력 사이즈가 896px임. do_pan_and_scan





## 성능

regurgitation (memorization) 확률이 0.001 (gemma2는 0.01)

![Image](https://learnopencv.com/wp-content/uploads/2025/03/Gemma-3-Comparison-on-Memorization-Rate.png)


## 허깅페이스

- image_processor_type : "Gemma3ImageProcessor"
- tokenizer: "GemmaTokenizerFast" 
- processor_class": "Gemma3Processor" = "Gemma3ImageProcessor" + "GemmaTokenizerFast"
- architectures:
    - "Gemma3ForConditionalGeneration" : For 4B, 12B, and 27B vision language models.
    - "Gemma3ForCausalLM" : 1B text only model and to load the vision language models like they were language models (omitting the vision tower)
- pt 버전 : pre-trained 버전, 추가 파인튜닝을 하지 안은 상태
- it 버전 : pt에서 post training까지 수행. 채팅 템플릿 필수로 사용


- tokenizer_config.json에  chat_template 포함 → system/user/assistant 토큰 내장

do_pan_and_scan 기법 사용



### 자료

- https://learnopencv.com/gemma-3/
- https://learnopencv.com/fine-tuning-gemma-3/
- http://medium.com/@manyi.yim/google-gemma-3-processor-52690caa3196
- https://github.dev/ariG23498/gemma3-object-detection/blob/main/utils.py