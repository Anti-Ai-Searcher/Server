import pytest
import torch
from models import kor_loader, eng_loader, img_loader
from models.model import device

@pytest.fixture(scope="session")
def ai_models():
    print("\n모델 로딩 중...")
    
    model_kor, tokenizer_kor = kor_loader.get_korean_model()
    model_eng, tokenizer_eng = eng_loader.get_english_model()
    model_img, model_img_preprocess = img_loader.get_image_model()
    
    return {
        "kor": {"model": model_kor, "tokenizer": tokenizer_kor},
        "eng": {"model": model_eng, "tokenizer": tokenizer_eng},
        "img": {"model": model_img, "preprocessor": model_img_preprocess},
        "device": device
    }