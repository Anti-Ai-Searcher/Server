from . import utils
import torch
from langdetect import detect
import torch.nn.functional as F

def detect_ai_generated_text(text: str, tokenizer, model, device, model_kor, tokenizer_kor):
    try:
        detected_lang = detect(text)
        if(detected_lang == 'ko'):
            #print(f"Detected language: Korean, {text}")
            prob = detect_ai_generated_text_kor(text, tokenizer_kor, model_kor, device)
            if prob['max_probability'] is None:
                prob = "error"
            else:
                prob = prob['max_probability']
            return prob
        else:
            #print(f"Detected language: English, {text}")
            return detect_ai_generated_text_eng(text, tokenizer, model, device)
    except:
        return detect_ai_generated_text_eng(text, tokenizer, model, device)
    
def detect_ai_generated_text_eng(text : str,tokenizer , model, device):
    try:
        inputs = tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        ai_probability = probabilities[:, 0].item()
        #print(ai_probability)

        return round(ai_probability, 4)

    except Exception as e:
        #print(f"AI 판별 오류: {e}")
        return None

def detect_ai_generated_text_kor(text: str, tokenizer, model, device, max_len=258):
    chunk_probabilities = []

    chunks=utils.chunk_text(text, tokenizer, max_len)

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            inputs = tokenizer(chunk, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(x=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            logits = outputs
            probabilities = F.softmax(logits, dim=1)
            ai_probability = probabilities[:, 0].item()
            chunk_probabilities.append(round(ai_probability, 4))
        except Exception as e:
            #print(f"청크 처리 오류: {e}")
            continue

    return utils.format_detection_results(chunk_probabilities)