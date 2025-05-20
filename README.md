# 🎙️ Real-Time Voice and Emotion Processing

Welcome to the **Real-Time Voice and Emotion Processing** project — a complete real-time web application that converts your voice to text and analyzes your emotions live.

Using **OpenAI's Whisper** for high-accuracy speech-to-text and a **custom emotion classification model**, this system allows natural human-computer interaction, ideal for smart assistants, therapy bots, education tech, and real-time surveillance.

---

## 📸 Demo

### ▶️ Video Preview

https://user-images.githubusercontent.com/yourusername/demo.mp4

### 🖼️ Screenshots

| 🎤 Voice Input | 😊 Emotion Output |
|---------------|------------------|
| ![Voice Input](screenshots/voice_input.png) | ![Emotion Output](screenshots/emotion_result.png) |

---

## 🚀 Features

- 🎧 **Live microphone input** via web interface  
- 🔤 **Real-time transcription** using OpenAI's Whisper (base/medium/large selectable)  
- 😊 **Emotion recognition** (Happy, Sad, Angry, Neutral, etc.)  
- 📊 Graph-based emotion feedback (Matplotlib or Chart.js support)  
- 🌐 Flask-based lightweight server-side application  
- 🖥️ Clean and responsive frontend (HTML/CSS/JavaScript)  
- 📁 Optional saving of results as `.txt`, `.json`, or `.csv`  
- 🎛️ Model selection panel for advanced users

---

## 📚 Technologies Used

| Component       | Technology                         |
|------------------|-------------------------------------|
| Transcription    | Whisper (PyTorch)                   |
| Emotion Analysis | scikit-learn / TensorFlow           |
| Backend          | Python 3.11 + Flask                 |
| Frontend         | HTML + CSS + JavaScript             |
| Audio Handling   | ffmpeg + pydub                      |
| Visualization    | Matplotlib / Chart.js               |

---

## ⚙️ Installation & Setup

Make sure you have **Python 3.11+**, `ffmpeg`, and `pip` installed.

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/real-time-voice-emotion.git
cd real-time-voice-emotion

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py

Then open your browser and go to:
📍 http://127.0.0.1:5000
