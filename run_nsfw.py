from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch
import requests
from io import BytesIO

# presigned URL로 이미지 불러오기
image_url = "https://hansicbuffet.s3.ap-northeast-2.amazonaws.com/cuAJVstap9NEKLgh.webp"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# 모델 로딩
model_path = "./nsfw_image_detection"
extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

# 추론
inputs = extractor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

# 사람이 보기 쉽게 퍼센트로 변환
safe_prob = probs[0][0].item() * 100
nsfw_prob = probs[0][1].item() * 100

print(f"Safe 확률: {safe_prob:.2f}%")
print(f"NSFW 확률: {nsfw_prob:.2f}%")
