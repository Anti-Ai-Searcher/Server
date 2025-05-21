# etc module
from bs4 import BeautifulSoup

# AI module 
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

def detect_ai_generated_text(text : str,tokenizer , model, device):
    try:
        inputs = tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        ai_probability = probabilities[:, 1].item()
        print(ai_probability)
        ai_probability = probabilities[:, 0].item()
        #print(ai_probability)

        return round(ai_probability, 4)

    except Exception as e:
        print(f"AI 판별 오류: {e}")
        return None