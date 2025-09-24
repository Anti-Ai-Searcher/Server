import logging
from pdfminer.high_level import extract_text
from langdetect import detect
from nltk.tokenize import sent_tokenize

from models.eng_loader import model_eng_tokenizer as tokenizer_eng
from models.kor_loader import model_kor_tokenizer as tokenizer_kor

# pdfminer 경고 억제
logging.getLogger("pdfminer").setLevel(logging.ERROR)

def pdf_parse(path: str, max_len: int, overlap: int = 25):
    
    # 1. PDF 텍스트 추출
    text = extract_text(path)
    
    detected_lang = detect(text)
    if detected_lang == "kor":
        sentences = kss.split_sentences(text)
        tokenizer = tokenizer_kor
    else:
        sentences = sent_tokenize(text)
        tokenizer = tokenizer_eng    

    chunks = []
    current_chunk_sentences = []
    current_length = 0
    max_len_for_chunking = max_len - 2  # special token 자리

    for sentence in sentences:
        sentence_token_ids = tokenizer.encode(sentence, add_special_tokens=False)
        sentence_length = len(sentence_token_ids)

        # 문장 하나가 너무 긴 경우 → 그대로 청크로 넣기
        if sentence_length > max_len_for_chunking:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            chunks.append(sentence)
            current_chunk_sentences = []
            current_length = 0
            continue

        # 현재 청크에 넣으면 초과 → 새로운 청크 시작
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
    return chunks

if __name__ == "__main__":
    pdf_parse("uploads/test.pdf",  256)
