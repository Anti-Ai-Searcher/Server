# etc module
from typing import Union
import detect_ai
import crawl

# AI module 
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# fastapi module
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import TransformerClassifier

print("Load Model")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'  # 학습할 때 사용한 모델이 'roberta-large'였는지 확인
model = RobertaForSequenceClassification.from_pretrained(model_name)

checkpoint = torch.load("ai_model/best-model.pt", map_location=device, weights_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_name) #tokenizer

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)

######
model_name_kor = "klue/roberta-base"
tokenizer_kor = AutoTokenizer.from_pretrained(model_name_kor)
checkpoint_kor = torch.load("ai_model/model_kor.pt", map_location=device, weights_only=True)

saved_args = checkpoint_kor.get('args')

if saved_args:
    config = vars(saved_args) if not isinstance(saved_args, dict) else saved_args
    d_model = config.get('d_model', 768)
    nhead = config.get('nhead', 12)
    num_layers = config.get('num_layers', 4)
    num_classes = config.get('num_classes', 2) 
    max_sequence_length = config.get('max_len', 128)
else:
    print("_____Warning: Model config not found in checkpoint, using hardcoded values._____")
    d_model = 768
    nhead = 12
    num_layers = 4
    num_classes = 2

model_kor = TransformerClassifier(
    vocab_size=tokenizer_kor.vocab_size,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    num_classes=num_classes,
    max_len=max_sequence_length
)

model_kor.load_state_dict(checkpoint_kor["model_state_dict"])
model_kor.to(device)
model_kor.eval()

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
            #ai_prob = detect_ai.detect_ai_generated_text_kor(text,tokenizer_kor, model_kor, device)
            # avg = ai_prob.get("average_probability")
            # max_p = ai_prob.get("max_probability")
            # num_chunks = ai_prob.get("chunk_count")
            # chunk_probs = ai_prob.get("chunk_probabilities")
            # result = {
            #     "average_probability": avg,
            #     "max_probability": max_p,
            #     "chunk_count": num_chunks
            # }
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
