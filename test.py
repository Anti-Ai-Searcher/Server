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
    # Test for the english text AI detection 
    def test_AI_texts_english_1(self):
        ai_text_link = "https://velog.io/@aiyaho123/%EB%B0%B1%EC%A4%80-1065%EB%B2%88-%ED%95%9C%EC%88%98-Python-%ED%92%80%EC%9D%B4-%EB%B0%8F-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_AI_texts_english_2(self):
        ai_text_link = "https://velog.io/@aiyaho123/Dyson-V12-Detect-Slim-Review-A-Laser-Powered-Cleaning-Experience-for-the-Data-Oriented-Mind#-final-thoughts"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_AI_texts_english_3(self):
        ai_text_link = "https://velog.io/@aiyaho123/Prison-Architect-Review-A-Simulation-Sandbox-for-Systems-Thinkers"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_AI_texts_english_4(self):
        ai_text_link = "https://velog.io/@aiyaho123/Spaghetti-Bolognese-for-Developers"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    
    def test_human_texts_english_1(self):
        human_text_link = "https://gaming.stackexchange.com/questions/276069/how-do-i-fight-sans-in-undertale"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.7
    def test_human_texts_english_2(self):
        human_text_link = "https://www.instructables.com/How-to-install-Linux-on-your-Windows/"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.7
    def test_human_texts_english_3(self):
        human_text_link = "https://medium.com/@ivan.mejia/c-development-using-visual-studio-code-cmake-and-lldb-d0f13d38c563"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.7
    def test_human_texts_english_4(self):
        human_text_link = "https://medium.com/@rjun07a/my-spring-2016-anime-list-23a226b2bc14"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.7

    # Test for the korean text AI detection 
    def test_AI_texts_korean_1(self):
        ai_text_link = "https://ai3886.tistory.com/2"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_AI_texts_korean_2(self):
        ai_text_link = "https://ai3886.tistory.com/3"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_AI_texts_korean_3(self):
        ai_text_link = "https://ai3886.tistory.com/4"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    def test_AI_texts_korean_4(self):
        ai_text_link = "https://ai3886.tistory.com/5"
        text = crawl.get_text_from_url(ai_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result < 0.4
    
    def test_human_texts_korean_1(self):
        human_text_link = "https://blog.naver.com/shinkyoungup/110092617672"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    def test_human_texts_korean_2(self):
        human_text_link = "https://blog.naver.com/skditjdqja12/140178347564"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    def test_human_texts_korean_3(self):
        human_text_link = "https://blog.naver.com/kkulmatapp/90084037882"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8
    def test_human_texts_korean_4(self):
        human_text_link = "https://m.blog.naver.com/junkigi11/20173492987"
        text = crawl.get_text_from_url(human_text_link)
        result = detect_ai.detect_ai_generated_text(text,tokenizer, model, device)
        assert result > 0.8

    