# etc module
from typing import Union
import detect_ai
import read_contents.crawl as crawl
import os
from io import BytesIO

# fastapi module
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models.model import device
from models.eng_loader import model_eng, model_eng_tokenizer
from models.kor_loader import model_kor, model_kor_tokenizer
from models.img_loader import model_img

# FastAPI Apps

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Settings

UPLOAD_DIR = "./uploads" # 임시 경로
os.makedirs(UPLOAD_DIR, exist_ok=True)

## Paths

@app.get('/')
async def index():
    return JSONResponse(content={"message": "Welcome to the Anti AI Searcher API!"})

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
        ai_prob = detect_ai.detect_ai_generated_text_kor(text,model_kor_tokenizer,model_kor,device)
        print(ai_prob)
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

        ai_prob = detect_ai.detect_ai_generated_text(s, model_kor_tokenizer, model_kor, model_kor, model_kor_tokenizer)
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
        text = "asfsdafsad"
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