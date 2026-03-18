from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

print("AI 모델을 불러오는 중입니다...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. 준비한 옷 사진 열기
try:
    image = Image.open("sample.jpg")
except FileNotFoundError:
    print("❌ sample.jpg 파일을 찾을 수 없습니다. 폴더에 이미지를 넣어주세요!")
    exit()

# 2. 이미지를 모델이 이해할 수 있게 변환
inputs = processor(images=image, return_tensors="pt")

# 3. 이미지 벡터(숫자) 추출
image_features = model.get_image_features(**inputs)

# [안전장치] 객체로 반환될 경우 텐서만 추출
if not isinstance(image_features, torch.Tensor):
    image_features = image_features.pooler_output

print("=========================================")
print("📸 성공! 이미지 벡터의 모양(크기):", image_features.shape) 
print("📸 이미지 벡터 앞부분 5개 숫자:\n", image_features[0][:5])
print("=========================================")