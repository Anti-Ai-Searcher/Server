# import our main modules
import app
import read_contents.crawl as crawl
from ai_detector import detector
from models import kor_loader, eng_loader
from models.model import device

# import the test client
from fastapi.testclient import TestClient
import pytest

# AI module 
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from models.model import TransformerClassifier

####


# Create a test client using the FastAPI app
client = TestClient(app.app)

# test_detect_ai.py
class TestDetectAI:
        # Test for the english text AI detection 
    def test_AI_texts_english_1(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/%EB%B0%B1%EC%A4%80-1065%EB%B2%88-%ED%95%9C%EC%88%98-Python-%ED%92%80%EC%9D%B4-%EB%B0%8F-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    def test_AI_texts_english_2(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/Dyson-V12-Detect-Slim-Review-A-Laser-Powered-Cleaning-Experience-for-the-Data-Oriented-Mind#-final-thoughts"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    def test_AI_texts_english_3(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/Prison-Architect-Review-A-Simulation-Sandbox-for-Systems-Thinkers"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    def test_AI_texts_english_4(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/Spaghetti-Bolognese-for-Developers"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    
    def test_human_texts_english_1(self, ai_models):
        human_text_link = "https://gaming.stackexchange.com/questions/276069/how-do-i-fight-sans-in-undertale"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4
    def test_human_texts_english_2(self, ai_models):
        human_text_link = "https://www.instructables.com/How-to-install-Linux-on-your-Windows/"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4
    def test_human_texts_english_3(self, ai_models):
        human_text_link = "https://medium.com/@ivan.mejia/c-development-using-visual-studio-code-cmake-and-lldb-d0f13d38c563"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4
    def test_human_texts_english_4(self, ai_models):
        human_text_link = "https://medium.com/@rjun07a/my-spring-2016-anime-list-23a226b2bc14"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4

    # Test for the korean text AI detection 
    def test_AI_texts_korean_1(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/2"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    def test_AI_texts_korean_2(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/3"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    def test_AI_texts_korean_3(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/4"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    def test_AI_texts_korean_4(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/5"
        text = crawl.get_text_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result > 0.7
    
    def test_human_texts_korean_1(self, ai_models):
        human_text_link = "https://ai3886.tistory.com/1"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4
    def test_human_texts_korean_2(self, ai_models):
        human_text_link = "https://aboooks.tistory.com/37"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4
    def test_human_texts_korean_3(self, ai_models):
        human_text_link = "https://www.ohmynews.com/NWS_Web/View/at_pg.aspx?CNTN_CD=A0000304103"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4
    def test_human_texts_korean_4(self, ai_models):
        human_text_link = "https://m.blog.naver.com/junkigi11/20173492987"
        text = crawl.get_text_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["tokenizer"],
            ai_models["eng"]["model"],
            ai_models["device"],
            ai_models["kor"]["model"],
            ai_models["kor"]["tokenizer"]
        )
        print(result)
        assert result < 0.4