# etc module
from typing import Union
import detect_ai
import crawl

# fastapi module
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
            text = crawl.get_text_from_url(url)
            if not text:
                results.append({"url": url, "ai_probability": "크롤링 실패"})
                continue

            ai_prob = detect_ai.detect_ai_generated_text(text)
            if ai_prob is None:
                results.append({"url": url, "ai_probability": "판별 실패"})
            else:
                results.append({"url": url, "ai_probability": ai_prob})

        return JSONResponse({"results": results})

    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.post("/check_url/")
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