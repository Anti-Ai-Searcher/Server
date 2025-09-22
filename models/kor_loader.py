import torch
from transformers import AutoTokenizer
from models.model import TransformerClassifier, device

def get_korean_model():
    print("Load Korean model")
    model_name_kor = "klue/roberta-base"
    tokenizer_kor = AutoTokenizer.from_pretrained(model_name_kor)
    checkpoint_kor = torch.load("ai_model/model_kor.pt", map_location=device, weights_only=True)

    saved_args = checkpoint_kor.get('args')

    if saved_args:
        config = vars(saved_args) if not isinstance(saved_args, dict) else saved_args
        d_model = config.get('d_model')
        nhead = config.get('nhead')
        num_layers = config.get('num_layers')
        num_classes = config.get('num_classes', 2) 
        max_sequence_length = config.get('max_len', 258)

    model_kor = TransformerClassifier(
        vocab_size=tokenizer_kor.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        max_len=max_sequence_length
    )

    model_kor.load_state_dict(checkpoint_kor["model_state_dict"])
    model_kor.to(device)
    model_kor.eval()

    print("Done")

    return model_kor, tokenizer_kor

model_kor, model_kor_tokenizer = get_korean_model()
