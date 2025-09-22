import logging
from pdfminer.high_level import extract_text
from app import model_kor_tokenizer

# pdfminer 경고 억제
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def pdf_parse(path: str, tokenizer, max_token: int, overlap: int = 25):
    
    # 1. PDF 텍스트 추출
    text = extract_text(path)
    print(text)
    # 2. 토큰화
    enc = tokenizer(
        text,
        max_length=128,
        truncation=True,
        stride=overlap,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )

    input_id_chunks = enc["input_ids"]              # List[List[int]]
    attn_mask_chunks = enc["attention_mask"]        # List[List[int]]

    chunks = [
        {"input_ids": ids, "attention_mask": mask}
        for ids, mask in zip(input_id_chunks, attn_mask_chunks)
    ]

    # 디버깅: 얼마만큼 쪼개졌는지
    print(f"[pdf_parse] {len(chunks)} chunks (<= {128} tokens each, stride={overlap}).")

    print(tokenizer.batch_decode(input_id_chunks, skip_special_tokens=True))
    
    return chunks



if __name__ == "__main__":
    pdf_parse("uploads/test.pdf", model_kor_tokenizer, 256)
