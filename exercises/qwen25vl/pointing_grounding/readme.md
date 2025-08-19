포인팅과 더불어

http://172.100.100.1:12317/tree?token=83b9ab5037f69b5dd1ecf3f70b06b5f9c53124de70675eee


Qwen2.5-VL 포인팅 그라운딩 설명

>
To ensure superior point-based object grounding capabilities, we have constructed a comprehensive
pointing dataset comprising both publicly available and synthetic data. Specifically, the data source
includes public pointing and counting data from PixMo (Deitke et al., 2024), publicly accessible object
grounding data (from both object detection and instance segmentation tasks), and data synthesized by an
automated pipeline for generating precise pointing data towards certain image details.

<img width="1516" height="817" alt="Image" src="https://github.com/user-attachments/assets/b01b0d61-4eb9-42fa-a404-5541f0bb550d" />

과정
일단 이미 pixmo-point로 학습해두었는데 다른 데이터셋에서 오토라벨링으로 데이터셋을 미리 만들어둘까?

1) 포인팅 능력을 일단 테스트 데이터셋을 구축해서 확인해보기
1-1) 예상되는 도메인 위주로 크롤링이던지 몇개를 구해서 VFM을 이용해서 카운팅 데이터셋 만들기 (예를 들어 GroundingDINO 같은 경우 bbox를 찾고 bbox 중점을 이용, SAM이면 마스크의 픽셀의 중앙값)
1-2) 오픈된 Qwen2.5-VL 모델에서 바로 테스트해보기

2) 와일드 이미지에 대해 많이 떨어질 것으로 예상됨
2-1) 학습용 데이터셋을 구축하고 학습시키기
2-2) 1에서 만든 테스트 데이터에 적용하여 얼만큼 향상되는지 확인

3) 혼잡도 상황에 대한 설명도 추후에 내놓도록 해야함
3-1) 정확한 설명을 하는 답변에 선호하도록 하는 DPO 학습



### 1-1. 크롤링

https://unsplash.com/developers

스톡이미지 프로바이더에서 얻기
unsplash에서 얻으려면 API 키 필요
Access Key
OrMJs9mvywQP8VWE5EVXhtfGXqqLlmBNk1IZD_kuoC0

Secret key
6RA74UYKzk3xlA5FLIxbyVZuJTDKMHK4CLWkrkqVrU8


플릭커 - 프로버전 구독자에게 API Key가 제공됨
https://www.flickr.com/services/apps/create


크롤링하고 간단한 디덱션 모델을 돌려서 새나 사람들이 있는지 확인

미리 만들어진걸 쓰면 되긴 함.

### 1-2. 필터링

CLIP, YOLO를 사용해서 필터링

