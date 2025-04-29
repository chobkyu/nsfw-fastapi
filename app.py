from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from io import BytesIO

app = FastAPI()

# 모델 로딩
model_path = './nsfw_image_detection'
extractor = AutoFeatureExtractor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)

class ImageRequest(BaseModel):
    image_url: str

@app.post("/nsfw")
async def nsfw_check(request: ImageRequest):
    image_url = request.image_url

    if not image_url:
        raise HTTPException(status_code=400, detail="No image_url provided")
    
    try:
        print(image_url)
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    
        image = image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load image: {str(e)}")
    
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    safe_prob = probs[0][0].item() * 100
    nsfw_prob = probs[0][1].item() * 100

    # 결과 반환
    return {
        "safe_probability": f"{safe_prob:.2f}%",
        "nsfw_probability": f"{nsfw_prob:.2f}%"
    }
