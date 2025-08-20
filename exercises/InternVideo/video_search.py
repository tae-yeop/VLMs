import faiss, numpy as np, torch
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoProcessor, AutoModel
import cv2
import locale, sys
print(locale.getpreferredencoding())


from huggingface_hub import login, create_repo
hf_token = ""
login(token=hf_token)


MODEL_ID = "microsoft/xclip-base-patch32"  # "OpenGVLab/InternVideo2-CLIP-1B-224p-f8"
NUM_FRAMES = 8
proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True
         ).eval().to("cuda")

def get_frames(path, n_fr=NUM_FRAMES):
    vr = VideoReader(path, ctx=cpu(0))
    idx = np.linspace(0, len(vr)-1, n_fr, dtype=int)
    return [Image.fromarray(vr[i].asnumpy()) for i in idx]


# def vid_embed(path, fps=1, n_fr=32):
#     vr = VideoReader(path, ctx=cpu(0)); tot = len(vr)
#     idx = np.linspace(0, tot-1, n_fr, dtype=int)
#     frames = [Image.fromarray(vr[i].asnumpy()) for i in idx]
#     inp = proc(video=frames, return_tensors="pt").to("cuda")
#     with torch.no_grad(): return model.get_video_features(**inp).cpu()

# def txt_embed(text):
#     inp = proc(text=[text], return_tensors="pt").to("cuda")
#     with torch.no_grad(): return model.get_text_features(**inp).cpu()

def vid_emb(path):
    inputs = proc(videos=get_frames(path), return_tensors="pt").to("cuda")
    with torch.no_grad():
        return model.get_video_features(**inputs).cpu()

def txt_emb(text):
    inputs = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        return model.get_text_features(**inputs).cpu()

# ① 인덱스 구축



paths = [
    "/purestorage/AILAB/AI_1/tyk/3_CUProjects/공항/20190917_Drop_11-04/20190917_07_Drop_00011_164_300/NIPA_17_07_20190917105959_Drop_00011_blur.mp4",

    "/purestorage/AILAB/AI_1/tyk/3_CUProjects/07.지능형_관제_서비스_CCTV_영상_데이터/3.개방데이터/1.데이터/Validation/01.원천데이터/E05_015.mp4",
    
    "/purestorage/AILAB/AI_1/tyk/3_CUProjects/078.스마트_제조_시설_안전_감시를_위한_데이터/3.개방데이터/1.데이터/Validation/01.원천데이터/intrusion/normal/rgb/intrusion_normal_rgb_0248_cctv3/intrusion_normal_rgb_0248_cctv3.mp4",
    
    "/purestorage/AILAB/AI_1/tyk/3_CUProjects/화재/불꽃/output.mp4"

]                       # 모든 MP4 경로
vecs  = torch.cat([vid_emb(p) for p in paths]).numpy()
index = faiss.IndexFlatIP(vecs.shape[1]); index.add(vecs)

# ② 검색
q = txt_emb("fire breaking out").numpy()
D,I = index.search(q, k=10)         # 상위 10개

print(I[0])
print(D[0])
# print([paths[i] for i in I[0]], D[0])
