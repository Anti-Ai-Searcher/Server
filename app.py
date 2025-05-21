# etc module
from typing import Union
import detect_ai
import crawl

# AI module 
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

# fastapi module
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

print("Load Model")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'  # 학습할 때 사용한 모델이 'roberta-large'였는지 확인
model = RobertaForSequenceClassification.from_pretrained(model_name)

checkpoint = torch.load("ai_model/best-model.pt", map_location=device, weights_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_name) #tokenizer

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)

print("Load Model Done")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 실제 배포 시에는 * 대신 필요한 도메인만 열어주세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def index():
    return JSONResponse(content={"message": "Welcome to the AI Detection API!"})

@app.post("/check_ai")
async def check_ai(request: Request):
    try:
        data = await request.json()
        links = data.get("links", [])

        if not isinstance(links, list) or len(links) == 0:
            return JSONResponse({"error": "Invalid input. Expecting {'links': ['url1', 'url2', ...]}"})

        results = []
        for url in links:
            print(url)
            text = crawl.get_text_from_url(url)
            if not text:
                results.append({"url": url, "ai_probability": "텍스트 추출 실패"})
                continue
            if len(text) < 200:
                results.append({"url": url, "ai_probability": "텍스트 길이 부족"})
                continue

            ai_prob = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
            results.append({"url": url, "ai_probability": ai_prob if ai_prob else "판별 실패"})
            if ai_prob is None:
                results.append({"url": url, "ai_probability": "판별 실패"})
            else:
                results.append({"url": url, "ai_probability": ai_prob})

        return JSONResponse({"results": results})

    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/check_url")
async def check_url(url : str):
    try:
        if not url:
            return JSONResponse({"error": "Invalid input. Expecting {'url': 'url'}"})

        text = crawl.get_text_from_url(url)
        if not text:
            return JSONResponse({"error": "크롤링 실패"})
        
        print(text)
        return JSONResponse({"text": text})
        
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/check_str/")
async def check_str(s : str):
    try:
        if(len(s) < 200):
            return JSONResponse({"error": f"Invalid input. Expecting the string's length must longer than 200"})
        ai_prob = detect_ai.detect_ai_generated_text(s,tokenizer, model, device)
        result = {"input" : s, "result" : ai_prob if ai_prob else "판별 실패"}
        return result
    except:
        return JSONResponse({'error': 'Invalid input. Expecting a string'})
