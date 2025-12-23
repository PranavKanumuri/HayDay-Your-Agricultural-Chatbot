# HayDay-Your-Agricultural-Chatbot

HayDay is an AI assistant built to help farmers and anyone interested in agriculture. It solves the language barrier by allowing you to ask questions in your own local language (like Hindi, Telugu, Tamil, etc.) using either text or your voice.

It uses the Dhenu2 model from KissanAI, which is specifically trained on agricultural topics.

## What It Does

* **Voice Support:** You can upload an audio file of your question, and the bot will listen, understand, and speak the answer back to you.
* **Multilingual:** Supports many Indian languages including Hindi, Bengali, Telugu, Tamil, Marathi, Gujarati, Kannada, Punjabi, Odia, and Assamese.
* **Smart AI:** Powered by the KissanAI Dhenu2 model (3 Billion parameters) to give accurate farming advice.
* **Auto-Translation:** You don't need to know English. The bot translates your question to English, gets the answer, and translates it back to your language automatically.

## How It Works

1.  **Input:** You provide text or an audio file.
2.  **Processing:** If it is audio, the system converts it to text using Google Speech Recognition.
3.  **Translation:** It detects your language and translates the question into English.
4.  **AI Response:** The Dhenu2 AI model thinks about the answer in English.
5.  **Final Output:** The answer is translated back to your language. If you used audio input, it also converts the text into speech so you can hear it.

## Tech Stack

* **Interface:** Streamlit
* **AI Model:** KissanAI/Dhenu2 (via Hugging Face Transformers)
* **Audio:** SpeechRecognition (Google) and gTTS (Google Text-to-Speech)
* **Translation:** Googletrans
* **Acceleration:** PyTorch (CUDA) and BitsAndBytes for 4-bit loading

## How to Run This

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/your-username/HayDay-Bot.git](https://github.com/your-username/HayDay-Bot.git)
    cd HayDay-Bot
    ```

2.  **Install the libraries**
    (Make sure you have Python installed)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```
