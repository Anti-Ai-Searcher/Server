import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models.model import device

def get_english_model():
    print("Load English Model")
    
    ckpt = torch.load('ai_model/model_eng.pt', map_location=device, weights_only=True)
    model_name = ckpt.get('model_name', 'roberta-base') # mocrosoft/deberta-v3-base OR roberta-base
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model.load_state_dict(state, strict = True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    model.eval()
    print("Done")
    return model, tokenizer

model_eng, model_eng_tokenizer = get_english_model()
