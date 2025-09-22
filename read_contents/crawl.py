# etc module
from readability import Document
import requests
from bs4 import BeautifulSoup

import kss
from nltk.tokenize import sent_tokenize

import time

from app import model_eng_tokenizer as tokenizer_eng
from app import model_kor_tokenizer as tokenizer_kor

def get_tokens_from_url(url):
    """ 주어진 URL에서 본문 텍스트 크롤링 """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # 봇 차단 우회
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None

        doc = Document(response.text)
        doc_content = doc.summary()

        soup = BeautifulSoup(doc_content, "html.parser")

        return soup.get_text()
    
    except Exception as e:
        print(f"크롤링 오류: {e}")
        log_file = open("log/log"+time.ctime()+".txt","w")
        log_file.write(url+"\n",e)
        log_file.close()
        return None

def tokenize_text_kor(text: str, max_len: int, overlap: int = 25):
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    max_len_for_chunking = max_len - 2

    sentences = kss.split_sentences(text)

    for sentence in sentences:
        sentence_token_ids = tokenizer_kor.encode(sentence, add_special_tokens=False)
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
    
    return chunks

def tokenize_text_eng(text: str,max_len: int, overlap: int = 25):
    chunks = []
    current_chunk_sentences = []
    current_length = 0
    max_len_for_chunking = max_len - 2 

    sentences = sent_tokenize(text)

    for sentence in sentences:
        sentence_token_ids = tokenizer_eng.encode(sentence, add_special_tokens=False)
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
    
    return chunks