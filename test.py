# import our main modules
import app
import crawl
import detect_ai

# import the test client
from fastapi.testclient import TestClient
import pytest

# AI module 
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

print("Load Model")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'roberta-base'  # 학습할 때 사용한 모델이 'roberta-large'였는지 확인
model = RobertaForSequenceClassification.from_pretrained(model_name)

checkpoint = torch.load("ai_model/best-model.pt", map_location=device, weights_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_name) #tokenizer

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.to(device)

print("Load Model Done")

# Create a test client using the FastAPI app
client = TestClient(app.app)

# test_detect_ai.py
class TestDetectAI:
    def test_AI_texts_1(self):
        ai_text_link = "https://ai3886.tistory.com/2"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    def test_AI_texts_2(self):
        ai_text_link = "https://ai3886.tistory.com/3"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    def test_AI_texts_3(self):
        ai_text_link = "https://ai3886.tistory.com/4"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    def test_AI_texts_4(self):
        ai_text_link = "https://ai3886.tistory.com/5"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    
    def test_human_texts_1(self):
        human_text_link = "https://blog.naver.com/shinkyoungup/110092617672"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_human_texts_2(self):
        human_text_link = "https://blog.naver.com/skditjdqja12/140178347564"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_human_texts_3(self):
        human_text_link = "https://blog.naver.com/kkulmatapp/90084037882"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_human_texts_4(self):
        human_text_link = "https://m.blog.naver.com/junkigi11/20173492987"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4

    