from transformers import CLIPProcessor, CLIPModel
import torch

print("AI 모델을 불러오는 중입니다... 잠시만 기다려주세요!")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. 테스트용 텍스트 입력
text_input = "좀 더 두꺼운 겨울 재질로 찾아줘"

# 2. 텍스트를 모델이 이해할 수 있게 변환
inputs = processor(text=[text_input], return_tensors="pt", padding=True)

# 3. 텍스트 벡터(숫자) 추출 (가장 중요한 부분!)
# model.get_text_features를 사용하면 바로 텐서(배열)가 나옵니다.
text_features = model.get_text_features(**inputs)

# [안전장치] 만약 버전 문제로 '상자(객체)'가 반환되었다면, 알맹이(pooler_output)만 꺼내도록 처리
if not isinstance(text_features, torch.Tensor):
    text_features = text_features.pooler_output

# 4. 결과 출력
print("=========================================")
print("🎉 성공! 추출된 벡터의 모양(크기):", text_features.shape) 
print("🎉 벡터 앞부분 5개 숫자 확인:\n", text_features[0][:5])
print("=========================================")