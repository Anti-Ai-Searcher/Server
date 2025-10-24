from . import utils
import torch
from langdetect import detect
import torch.nn.functional as F
import read_contents.crawl as crawl
from models.model import device
from models.eng_loader import model_eng_tokenizer as tokenizer_eng
from models.kor_loader import model_kor_tokenizer as tokenizer_kor
from models.img_loader import model_img_preprocess
from PIL import Image

def detect_ai_generated_text(text: str, model_eng, model_kor):
    try:
        detected_lang = detect(text)
        if(detected_lang == 'ko'):
            prob = detect_ai_generated_text_kor(text, model_kor)
        else:
            prob = detect_ai_generated_text_eng(text, model_eng)

        if prob['average_probability'] is None:
            prob = "error"
        else:
            prob = prob['average_probability']
        return prob
    except:
        return detect_ai_generated_text_eng(text, model_eng)
    
def detect_ai_generated_text_eng(text : str, model, max_len=256):
    chunk_probabilities = []

    chunk_ids_list = crawl.tokenize_text_eng(text, max_len)

    for chunk_ids in chunk_ids_list:
        if not chunk_ids:
            continue  
        try:
            tokens = chunk_ids
            sequence_length = len(tokens) + 2
            
            num_padding = max_len - sequence_length
            padding = [tokenizer_eng.pad_token_id] * num_padding

            final_tokens_list = [tokenizer_eng.bos_token_id] + tokens + [tokenizer_eng.eos_token_id] + padding
            input_ids = torch.tensor(final_tokens_list).unsqueeze(0).to(device)
            
            attention_mask = torch.zeros(max_len, dtype=torch.long)
            attention_mask[:sequence_length] = 1
            attention_mask = attention_mask.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)
            ai_probability = probabilities[:, 0].item() 
            chunk_probabilities.append(round(ai_probability, 4))
            
        except Exception as e:
            continue

    return utils.format_detection_results(chunk_probabilities)

def detect_ai_generated_text_kor(text: str, model, max_len=256):
    chunk_probabilities = []

    chunk_ids_list = crawl.tokenize_text_kor(text, max_len)

    for chunk_ids in chunk_ids_list:
        if not chunk_ids:
            continue
        
        try:

            tokens = chunk_ids
            sequence_length = len(tokens) + 2
            num_padding = max_len - sequence_length
            padding = [tokenizer_kor.pad_token_id] * num_padding

            final_tokens_list = [tokenizer_kor.bos_token_id] + tokens + [tokenizer_kor.eos_token_id] + padding
            input_ids = torch.tensor(final_tokens_list).unsqueeze(0).to(device)
            
            attention_mask = torch.zeros(max_len, dtype=torch.long)
            attention_mask[:sequence_length] = 1
            attention_mask = attention_mask.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(x=input_ids, attention_mask=attention_mask)
            
            logits = outputs
            probabilities = F.softmax(logits, dim=1)
            ai_probability = probabilities[:, 0].item()
            chunk_probabilities.append(round(ai_probability, 4))
            
        except Exception as e:
            continue

    return utils.format_detection_results(chunk_probabilities)

def detect_ai_generated_image(img: Image, model_img):
    img = model_img_preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_img(img)    
        probs  = torch.softmax(logits, dim=-1)[0].cpu().numpy()  
    return probs[0]