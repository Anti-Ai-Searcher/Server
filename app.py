# etc module
from typing import Union
import detect_ai
import crawl
import os
import aiofiles
from io import BytesIO
from pdfminer.high_level import extract_text

# AI module 
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# fastapi module
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads" # 임시 경로
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

            ai_prob = detect_ai.detect_ai_generated_text(text,tokenizer, model, device, model_kor, tokenizer_kor)

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

@app.post("/check_str")
async def check_str(request: Request):
    try:
        data = await request.json()
        s = data.get("text", "")
        print(f'input : {s}')
        if len(s) < 200:
            return JSONResponse(
                status_code=400,
                content={"error": "텍스트의 길이가 200자 이상이어야 합니다."}
            )

        ai_prob = detect_ai.detect_ai_generated_text(s, tokenizer, model, device, model_kor, tokenizer_kor)
        result = {
            "input": s,
            "result": ai_prob if ai_prob else "판별 실패"
        }
        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"result": "Invalid input. Expecting a JSON object with a 'text' field"}
        )

@app.post("/check_pdf")
async def check_pdf(request: Request):
    try:
        form = await request.form()
        upload: UploadFile | None = form.get("upload")

        if upload is None:
            raise HTTPException(status_code=400, detail="파일이 전송되지 않았습니다. (필드명: upload)")

        if upload.content_type != "application/pdf" and not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="PDF 파일만 업로드할 수 있습니다.")

        pdf_bytes = await upload.read()
        text = extract_text(BytesIO(pdf_bytes)).strip()
        print(f'pdf text : {text}')
        if not text:
            return JSONResponse({"error": "PDF에서 텍스트를 추출할 수 없습니다."})
        elif len(text) < 200:
            return JSONResponse(
                status_code=400,
                content={"error": "텍스트의 길이가 200자 이상이어야 합니다."}
            )
        ai_prob = detect_ai.detect_ai_generated_text(text, tokenizer, model, device,model_kor,tokenizer_kor)
        result = {
            "input": text,
            "result": ai_prob if ai_prob else "판별 실패"
        }
        return JSONResponse(result)

    except HTTPException as e:
        return JSONResponse({"error": e.detail}, status_code=e.status_code)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)