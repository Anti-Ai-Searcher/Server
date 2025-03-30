import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'  # 학습할 때 사용한 모델이 'roberta-large'였는지 확인
model = RobertaForSequenceClassification.from_pretrained(model_name)

checkpoint = torch.load("best-model.pt", map_location=device, weights_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)

app = Flask(__name__)

def detect_ai_generated_text(text):
    try:
        inputs = tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        ai_probability = probabilities[:, 1].item()
        #print(ai_probability)

        return round(ai_probability, 4)

    except Exception as e:
        print(f"AI 판별 오류: {e}")
        return None

def get_text_from_url(url):
    """ 주어진 URL에서 본문 텍스트 크롤링 """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # 봇 차단 우회
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = soup.find_all("p")  
        text = "\n".join([p.get_text() for p in paragraphs])

        return text.strip() if text else None
    except Exception as e:
        print(f"크롤링 오류: {e}")
        return None

@app.route('/')
def first():
    return render_template('index.html')


@app.route("/check_ai", methods=["POST"])
def check_ai():
    try:
        data = request.get_json()
        links = data.get("links", [])

        if not isinstance(links, list) or len(links) == 0:
            return jsonify({"error": "Invalid input. Expecting {'links': ['url1', 'url2', ...]}"})

        results = []
        for url in links:
            text = get_text_from_url(url)
            if not text:
                results.append({"url": url, "ai_probability": "크롤링 실패"})
                continue

            ai_prob = detect_ai_generated_text(text)
            results.append({"url": url, "ai_probability": ai_prob if ai_prob else "판별 실패"})

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=True)