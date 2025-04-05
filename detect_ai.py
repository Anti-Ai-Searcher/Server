# etc module
import requests
from bs4 import BeautifulSoup
import time

# AI module 
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
        log_file = open("log/log"+time.ctime()+".txt","w")
        log_file.write(url+"\n",e)
        return None