# etc module
from readability import Document
import requests
from bs4 import BeautifulSoup
import time

def get_text_from_url(url):
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
