# etc module
from langdetect import detect

# AI module
import torch
import torch.nn.functional as F
from models.model import device
import kss

def detect_ai_generated_text(text: str, tokenizer, model, model_kor, tokenizer_kor):
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

def remove_outliers_iqr(probs):
    sorted_probs = sorted(probs)
    q1 = sorted_probs[len(sorted_probs) // 4]
    q3 = sorted_probs[len(sorted_probs) * 3 // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    filtered = [p for p in probs if lower <= p <= upper]
    return filtered

def detect_ai_generated_text_kor(text: str, tokenizer, model, device, max_len=128):
    try:
        sentences = kss.split_sentences(text)
    except ImportError:
        #print("경고: 'kss' 미사용. 개행 또는 마침표 기준 분리.")
        sentences = [s.strip() for s in text.split('\n') if s.strip()]
        if len(sentences) <= 1:
            sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]

    chunks = []
    current_chunk_sentences = []
    current_length = 0
    max_len_for_chunking = max_len - 2

    for sentence in sentences:
        sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_token_ids)

        if sentence_length > max_len_for_chunking:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            chunks.append(sentence)
            current_chunk_sentences = []
            current_length = 0
            continue

        if current_length + sentence_length > max_len_for_chunking:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_length = sentence_length
        else:
            current_chunk_sentences.append(sentence)
            current_length += sentence_length

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))

    # 3. 각 청크별 AI 생성 확률 계산
    chunk_probabilities = []
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

    # 4. 최종 결과 반환
    if chunk_probabilities:
        filtered_probs = remove_outliers_iqr(chunk_probabilities)
        avg_prob = round(sum(filtered_probs) / len(filtered_probs), 4)
        max_prob = round(max(filtered_probs), 4)
        print(filtered_probs)
        
        return {
            "average_probability": avg_prob,
            "max_probability": max_prob,
            "chunk_probabilities": chunk_probabilities,
            "chunk_count": len(chunk_probabilities)
        }
    else:
        return {
            "average_probability": None,
            "max_probability": None,
            "chunk_probabilities": [],
            "chunk_count": 0
        }


