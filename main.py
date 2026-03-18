from fastapi import FastAPI, UploadFile, File, Form
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

app = FastAPI()

# 서버가 켜질 때 AI 모델을 한 번만 메모리에 미리 올려둡니다.
print("🚀 FastAPI 서버 시작 중... AI 모델 로딩 (약 3~5초 소요)")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ AI 모델 로딩 완료!")

@app.get("/")
def read_root():
    return {"message": "StyleLens AI 벡터 서버가 정상 작동 중입니다!"}

# Member 2 (Spring) 가 호출할 핵심 API 주소
@app.post("/api/extract-vector")
async def extract_vector(
    text: str = Form(None),         # 자연어 텍스트 (선택사항)
    file: UploadFile = File(None)   # 이미지 파일 (선택사항)
):
    result = {}

    # 1. 텍스트가 들어왔을 때 처리
    if text:
        text_inputs = processor(text=[text], return_tensors="pt", padding=True)
        text_features = model.get_text_features(**text_inputs)
        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output
        
        # Spring 서버가 읽기 편하도록 파이썬 리스트로 변환하여 저장
        result["text_vector"] = text_features[0].tolist() 

    # 2. 이미지 파일이 들어왔을 때 처리
    if file:
        # 전송받은 파일을 읽어서 이미지로 변환
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        image_inputs = processor(images=image, return_tensors="pt")
        image_features = model.get_image_features(**image_inputs)
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output
            
        result["image_vector"] = image_features[0].tolist()

    return result