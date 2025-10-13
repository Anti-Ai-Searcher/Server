# import our main modules
import app
import read_contents.crawl as crawl
from ai_detector import detector
from PIL import Image

# import the test client
from fastapi.testclient import TestClient

# Create a test client using the FastAPI app
client = TestClient(app.app)

# test_detect_ai.py
class TestDetectAI:
        # Test for the english text AI detection 
    def test_AI_texts_english_1(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/%EB%B0%B1%EC%A4%80-1065%EB%B2%88-%ED%95%9C%EC%88%98-Python-%ED%92%80%EC%9D%B4-%EB%B0%8F-%EA%B0%9C%EB%85%90-%EC%A0%95%EB%A6%AC"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("engAI1",result)
        assert result > 0.7
    def test_AI_texts_english_2(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/Dyson-V12-Detect-Slim-Review-A-Laser-Powered-Cleaning-Experience-for-the-Data-Oriented-Mind#-final-thoughts"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("engAI2",result)
        assert result > 0.7
    def test_AI_texts_english_3(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/Prison-Architect-Review-A-Simulation-Sandbox-for-Systems-Thinkers"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("engAI3",result)
        assert result > 0.7
    def test_AI_texts_english_4(self, ai_models):
        ai_text_link = "https://velog.io/@aiyaho123/Spaghetti-Bolognese-for-Developers"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("engAI4",result)
        assert result > 0.7
    
    def test_human_texts_english_1(self, ai_models):
        human_text_link = "https://gaming.stackexchange.com/questions/276069/how-do-i-fight-sans-in-undertale"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("enghum1",result)
        assert result < 0.4
    def test_human_texts_english_2(self, ai_models):
        human_text_link = "https://www.instructables.com/How-to-install-Linux-on-your-Windows/"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("enghum2",result)
        assert result < 0.4
    def test_human_texts_english_3(self, ai_models):
        human_text_link = "https://medium.com/@ivan.mejia/c-development-using-visual-studio-code-cmake-and-lldb-d0f13d38c563"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("enghum3",result)
        assert result < 0.4
    def test_human_texts_english_4(self, ai_models):
        human_text_link = "https://medium.com/@rjun07a/my-spring-2016-anime-list-23a226b2bc14"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("enghum4",result)
        assert result < 0.4

    # Test for the korean text AI detection 
    def test_AI_texts_korean_1(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/2"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("koAI1",result)
        assert result > 0.7
    def test_AI_texts_korean_2(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/3"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("koAI2",result)
        assert result > 0.7
    def test_AI_texts_korean_3(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/4"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("koAI3",result)
        assert result > 0.7
    def test_AI_texts_korean_4(self, ai_models):
        ai_text_link = "https://ai3886.tistory.com/5"
        text = crawl.get_tokens_from_url(ai_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("koAI4",result)
        assert result > 0.7
    
    def test_human_texts_korean_1(self, ai_models):
        human_text_link = "https://ai3886.tistory.com/1"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("kohum1",result)
        assert result < 0.4
    def test_human_texts_korean_2(self, ai_models):
        human_text_link = "https://aboooks.tistory.com/37"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("kohum2",result)
        assert result < 0.4
    def test_human_texts_korean_3(self, ai_models):
        human_text_link = "https://www.ohmynews.com/NWS_Web/View/at_pg.aspx?CNTN_CD=A0000304103"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("kohum3",result)
        assert result < 0.4
    def test_human_texts_korean_4(self, ai_models):
        human_text_link = "https://m.blog.naver.com/junkigi11/20173492987"
        text = crawl.get_tokens_from_url(human_text_link)
        result = detector.detect_ai_generated_text(
            text, 
            ai_models["eng"]["model"],
            ai_models["kor"]["model"],
        )
        print("kohum4",result)
        assert result < 0.4

    def test_human_image_1(self, ai_models):
        img_path = "./testfile/IU.jpeg"
        img = Image.open(img_path).convert("RGB")

        ai_prob = detector.detect_ai_generated_image(img,ai_models["img"]["model"])

        print("imghum1",ai_prob)
        assert ai_prob < 0.6
    
    def test_human_image_2(self, ai_models):
        img_path = "./testfile/street.jpg"
        img = Image.open(img_path).convert("RGB")

        ai_prob = detector.detect_ai_generated_image(img,ai_models["img"]["model"])

        print("imghum2",ai_prob)
        assert ai_prob < 0.6
    
    def test_ai_image_1(self, ai_models):
        img_path = "./testfile/midjourney.jpeg"
        img = Image.open(img_path).convert("RGB")

        ai_prob = detector.detect_ai_generated_image(img,ai_models["img"]["model"])

        print("imgAI1",ai_prob)
        assert ai_prob >= 0.6
    
    def test_ai_image_2(self, ai_models):
        img_path = "./testfile/nanobanana.png"
        img = Image.open(img_path).convert("RGB")

        ai_prob = detector.detect_ai_generated_image(img,ai_models["img"]["model"])

        print("imgAI2",ai_prob)
        assert ai_prob >= 0.6