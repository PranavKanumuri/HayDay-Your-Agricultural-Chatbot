import streamlit as st
import torch
import re
import os
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from gtts import gTTS
import speech_recognition as sr
from googletrans import Translator

st.set_page_config(page_title="Multilingual Kissan Chatbot", layout="centered")


# ==============================
# 💡 MODEL CLASS
# ==============================

class MultilingualKissanAI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.supported_languages = {
            'en': 'en', 'hi': 'hi', 'bn': 'bn', 'ta': 'ta',
            'te': 'te', 'ml': 'ml', 'mr': 'mr', 'gu': 'gu',
            'kn': 'kn', 'pa': 'pa', 'or': 'or', 'as': 'as'
        }
        self.translator = Translator()
        self._load_models()

    def _load_models(self):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        print("🔁 Loading KissanAI model...")
        self.kissan_tokenizer = AutoTokenizer.from_pretrained("KissanAI/Dhenu2-In-Llama3.2-3B-Instruct")
        self.kissan_model = AutoModelForCausalLM.from_pretrained(
            "KissanAI/Dhenu2-In-Llama3.2-3B-Instruct",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def detect_language(self, text):
        patterns = {
            'hi': r'[\u0900-\u097F]', 'bn': r'[\u0980-\u09FF]', 'ta': r'[\u0B80-\u0BFF]',
            'te': r'[\u0C00-\u0C7F]', 'ml': r'[\u0D00-\u0D7F]', 'gu': r'[\u0A80-\u0AFF]',
            'kn': r'[\u0C80-\u0CFF]', 'pa': r'[\u0A00-\u0A7F]', 'or': r'[\u0B00-\u0B7F]'
        }
        for lang, pattern in patterns.items():
            if re.search(pattern, text):
                return lang
        return 'en'

    def translate_to_english(self, text, lang_code):
        if lang_code == 'en':
            return text
        return self.translator.translate(text, src=lang_code, dest='en').text

    def translate_from_english(self, text, tgt_lang_code):
        if tgt_lang_code == 'en':
            return text
        return self.translator.translate(text, src='en', dest=tgt_lang_code).text

    def generate_response(self, english_text):
        inputs = self.kissan_tokenizer(english_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.kissan_model.generate(
                inputs.input_ids,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        return self.kissan_tokenizer.decode(output[0], skip_special_tokens=True)

    def chat(self, user_input):
        import time
        start = time.time()
        lang = self.detect_language(user_input)
        eng_input = self.translate_to_english(user_input, lang) if lang != 'en' else user_input
        eng_response = self.generate_response(eng_input)
        final_response = self.translate_from_english(eng_response, lang) if lang != 'en' else eng_response
        return {
            "detected_language": lang,
            "english_input": eng_input if lang != 'en' else None,
            "english_response": eng_response,
            "translated_response": final_response if lang != 'en' else None,
            "final_response": final_response,
            "processing_time": time.time() - start
        }

# ==============================
# 🔧 UTILITY FUNCTION
# ==============================

def process_user_input(bot, text_input=None, audio_file=None):
    assert text_input or audio_file, "Either text or audio input required."
    tts_lang = None

    if audio_file:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            temp.write(audio_file.read())
            temp_path = temp.name

        with sr.AudioFile(temp_path) as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)

        try:
            input_text = recognizer.recognize_google(audio, language="hi-IN")
            tts_lang = "hi"
        except sr.UnknownValueError:
            try:
                input_text = recognizer.recognize_google(audio, language="te-IN")
                tts_lang = "te"
            except:
                return {"error": "Audio not recognized."}
        finally:
            os.remove(temp_path)

    else:
        input_text = text_input.strip()

    if not input_text:
        return {"error": "Empty input."}

    response = bot.chat(input_text)
    result = {
        "input_text": input_text,
        "detected_language": response["detected_language"],
        "translated_input": response.get("english_input"),
        "english_response": response["english_response"],
        "translated_response": response.get("translated_response"),
        "final_response": response["final_response"],
        "processing_time": response["processing_time"],
        "tts_lang": tts_lang
    }

    if tts_lang:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_audio:
            gTTS(response["final_response"], lang=tts_lang).save(tts_audio.name)
            result["tts_path"] = tts_audio.name

    return result

# ==============================
# 🖼️ STREAMLIT UI
# ==============================

@st.cache_resource
def load_model():
    return MultilingualKissanAI()

bot = load_model()

st.title("🌾 Multilingual Kissan AI")
st.markdown("Ask your agricultural question using text or audio.")

input_type = st.radio("Choose input type:", ["Text", "Audio"])

if input_type == "Text":
    user_input = st.text_area("📝 Enter your question:", height=100)
    if st.button("Submit Text"):
        with st.spinner("Generating response..."):
            result = process_user_input(bot, text_input=user_input)

elif input_type == "Audio":
    audio = st.file_uploader("🎤 Upload a .wav file", type=["wav"])
    if st.button("Submit Audio") and audio:
        with st.spinner("Processing audio..."):
            result = process_user_input(bot, audio_file=audio)

# Display result
if "result" in locals() and result:
    if "error" in result:
        st.error(result["error"])
    else:
        st.success("✅ Response Ready!")
        st.write(f"🧠 {result['final_response']}")
        with st.expander("🔍 Details"):
            st.write(f"Detected Language: {result['detected_language']}")
            if result["translated_input"]:
                st.write(f"Translated Input: {result['translated_input']}")
            st.write(f"English Response: {result['english_response']}")
            if result["translated_response"]:
                st.write(f"Back-translated: {result['translated_response']}")
            st.write(f"Processing Time: {result['processing_time']:.2f} sec")

        if "tts_path" in result:
            st.audio(result["tts_path"], format="audio/mp3")
